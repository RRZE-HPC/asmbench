#!/usr/bin/env python3

import sys
import collections
import re
from pprint import pprint
import itertools
import argparse

import llvmlite.binding as llvm

import op
import bench


def translate_dag_to_py(raw):
    op, ramainer = raw.split(' ', 1)
    
    if remainder:
        recursive = False
        p_open = 0
        i = 0
        while i in range(len(remainder)):
            if raw[i] == '(':
                recursive = True
                p_open += 1
            elif raw[i] == ')':
                p_open -= 1
            if p_open == 0 and recursive:
                translate_dag_to_py(remainder[last_split:i])
            i += 1
           
    tuple([e.strip(',') for e in m.group(1).split(' ') if e is not None])
    return raw


def split_list(raw):
    last_split = 0
    p_open = 0
    i = 0
    while i in range(len(raw)):
        if raw[i] == '(':
            p_open += 1
        elif raw[i] == ')':
            p_open -= 1
        i += 1
        
        if p_open == 0 and raw[i:].startswith(', '):
            yield raw[last_split:i]
            if raw[i:].startswith(', '):
                i += 2
            last_split = i
        
        if i == len(raw)-1:
            yield raw[last_split:]


def translate_to_py(raw, type_, line=None):
    if raw == '':
        return
    # unknown
    if raw == '?':
        return '?'

    # string
    m = re.match(r'^["\'](.*)["\']$', raw)
    if type_ == 'string' and m:
        return m.group(1)
    # bit
    if type_ == 'bit':
        return bool(int(raw))
    # int
    if type_ == 'int':
        return int(raw)
    # dag
    m = re.match(r'^\((.*)\)$', raw)
    if type_ == 'dag' and m:
        return raw # translate_dag_to_py(m.group(1))
    # tuple
    m = re.match(r'^{(.*)}$', raw)
    if type_ in ['int', 'bit'] and m:
        return tuple([translate_to_py(e, type_, line) for e in m.group(1).split(',')])
    # list
    m = re.match(r'^\[(.*)\]$', raw)
    type_m = re.match(r'list<(.+)>', type_)
    if type_m and m:
        return m.group(1) # [translate_to_py(e, type_m.group(1), line) for e in split_list(m.group(1))]
    # function call(?)
    if re.match(r'^[a-zA-Z]+\(.*\)$', raw):
        return raw
    # code
    if type_ == 'code':
        # don't know what todo
        return raw
    # Register
    if type_ == 'Register':
        return raw
    # constant/reference(?)
    if re.match(r'^[A-Za-z0-9_]+$', raw.strip()):
        return raw.strip()
    
    print('unmachted type:', type_, file=sys.stderr)
    print('raw:', raw, file=sys.stderr)
    print('in line:', line, file=sys.stderr)
    sys.exit(1)


def evt_to_llvm_type(evt):
    # vN<evt>
    vector = None
    m = re.match(r'v([0-9]+)([fi].*)', evt)
    if m:
        vector = int(m.group(1))
        evt = m.group(2)
    type_ = None
    map_ = {'i1': 'i1',
            'i8': 'i8',
            'i16': 'i16',
            'i32': 'i32',
            'i64': 'i64',
            'f32': 'float',
            'f64': 'double',
            'f80': 'x86_fp80',
            'f128': 'fp128',
            'x86mmx': 'double',  # TODO
            'iPTR': '*TODO'}
    # i32, i64
    if evt in map_:
        type_ = map_[evt]
    
    if type_ is None and evt not in ['OtherVT', 'untyped']:
        raise ValueError("Unkown EVT type '{}' can not be converted to LLVM IR type.".format(evt))
    
    if vector is not None:
        return '<{} x {}>'.format(vector, type_)
    else:
        return type_


def convert_asm_to_att(asm):
    att = ''
    
    ignore_until = []
    for c in asm:
        if ignore_until and c != ignore_until[-1]:
            if c == '{':
                ignore_until.append('}')
            continue
        if ignore_until and c == ignore_until[-1]:
            ignore_until.pop()
            continue
        elif c == '{' or c == '}':
            # ignoring { and }
            continue
        elif c == '|':
            ignore_until.append('}')
            continue
        else:
            att += c
    return att


def rename_key(d, k_old, k_new, error=False):
    if k_old in d:
        d[k_new] = d[k_old]
        del d[k_old]
    else:
        if error:
            raise KeyError(k_old)

reg_class_conv_map = {
    'FR64': ('x', 'double'),
    'FR32': ('x', 'float'),
    'VR64': ('x', '<2 x float>'),
}

def convert_operands(operands, data):
    '''Take operand string from tablegen and convert it into dictionary.'''
    operands_dict = {}
    for m in re.finditer(r'(?P<reg_class>[a-zA-Z0-9_]+):(?P<reg_name>\$[a-z0-9]+)', operands):
        d = m.groupdict()
        constraint = None
        llvm_types = []
        operands_dict[d['reg_name']] = llvm_types
        
        if d['reg_class'] in reg_class_conv_map:
            llvm_types.append(reg_class_conv_map[d['reg_class']])
            continue
        
        reg_data = data[d['reg_class']]
        if ('string', 'OperandType') in reg_data:
            # Operands, but not registers
            op_type = reg_data[('string', 'OperandType')]
            if op_type == 'OPERAND_MEMORY':
                constraint = 'm'
                raise ValueError("due to memory operand")
            elif op_type == 'OPERAND_IMMEDIATE':
                constraint = 'i'
                vt = reg_data[('ValueType', 'Type')]
                llvm_types.append((constraint, evt_to_llvm_type(vt)))
                continue
            elif op_type == 'OPERAND_PCREL':
               raise ValueError("due to pcrel operand")
            elif op_type == "OPERAND_REGISTER":
                constraint = 'r'
                reg_data = data[reg_data[('RegisterClass', 'RegClass')]]
            else:
                raise ValueError("due to unknown operand type: {}".format(op_type))
        if ('list<ValueType>', 'RegTypes') in reg_data:
            # Registers
            for vt in reg_data[('list<ValueType>', 'RegTypes')].split(', '):
                lt = evt_to_llvm_type(vt)
                if ' x ' in lt:
                    constraint = 'x'
                else:
                    constraint = 'r'
                llvm_types.append((constraint, lt))
        
        if not llvm_types:
            raise ValueError("no operand types found")

    return operands_dict


def build_operand(op_constraint, op_type):
    if op_constraint in ['r', 'x']:
        return op.Register(op_type, op_constraint)
    elif op_constraint == 'i':
        return op.Immediate(op_type, '1')
    else:
        raise ValueError("unsupported llvm constraint")


def main():
    parser = argparse.ArgumentParser(description='Build benchmarks from TableGen output.')
    parser.add_argument('input', metavar='IN', type=argparse.FileType('r'), default=sys.stdin,
                        help='Input file (default stdin)')
    parser.add_argument("-v", "--verbosity", action="count", default=0,
                        help="increase output verbosity")
    args = parser.parse_args()
    
    data = collections.OrderedDict()

    cur_def = None
    for l in args.input.readlines():
        if cur_def is None:
            m = re.match(r'^def (?P<name>[A-Za-z0-9_]+) {', l)
            if m:
                cur_def = m.group(1)
                data[cur_def] = collections.OrderedDict()
        else:
            if l.startswith('}'):
                cur_def = None
                continue
            m = re.match(r'(?P<type>[A-Za-z<>]+) (?P<name>[A-Za-z]+) = (?P<value>.+);$', l.strip())
            if m:
                g = m.groupdict() 
                # Handle value types
                value = translate_to_py(g['value'], g['type'], l)
                data[cur_def][(g['type'], g['name'])] = value

    llvm.initialize()
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
    llvm.initialize_native_asmparser()
    
    init_value_by_llvm_type = {'i'+bits: '1' for bits in ['1', '8', '16', '32', '64']}
    init_value_by_llvm_type.update({fp_type: '1.0' for fp_type in ['float', 'double', 'fp128']})
    init_value_by_llvm_type.update(
        {'<{} x {}>'.format(vec, t): '<'+', '.join([t+' '+v]*vec)+'>'
         for t, v in init_value_by_llvm_type.items()
         for vec in [2, 4, 8, 16, 32, 64]})

    instr_data = collections.OrderedDict()
    instructions = collections.OrderedDict()

    run = False
    for instr_name, instr in data.items():
        #if instr_name != 'VPCLMULQDQYrr' and not run:
        #    continue
        #else:
        #    run = True

        # Filter non-instruction and uninteresting or unsupported ones
        if ('dag', 'OutOperandList') not in instr:
            if args.verbosity > 0:
                print('skipped', instr_name, 'due to missing OutOperandList', file=sys.stderr)
            continue
        if not instr[('string', 'AsmString')]:
            if args.verbosity > 0:
                print('skipped', instr_name, 'due to empty asm string', file=sys.stderr)
            continue
        if instr[('string', 'AsmString')].startswith('#') or re.match(r'^[A-Z]+', instr[('string', 'AsmString')]):
            if args.verbosity > 0:
                print('skipped', instr_name, 'due to strange asm string:', instr[('string', 'AsmString')],
                      file=sys.stderr)
            continue
        if '%' in instr[('string', 'AsmString')]:
            if args.verbosity > 0:
                print('skipped', instr_name, 'due to hardcoded register in asm string:',
                      instr[('string', 'AsmString')], file=sys.stderr)
            continue
        # TODO Automatically detext feature set
        if any([p in ['HasTBM', 'Not64BitMode', 'HasMPX', 'HasSSE4A', 'HasGFNI', 'HasDQI', 'HasBWI',
                      'HasAVX512', 'Has3DNow', 'HasSHA', 'HasVAES', 'HasVLX', 'HasERI', 'HasFMA4',
                      'HasXOP', 'HasVBMI2', 'HasCDI', 'HasVNNI', 'HasVBMI', 'HasIFMA',
                      'HasBITALG', 'HasVPOPCNTDQ'] 
                for p in instr[('list<Predicate>', 'Predicates')].split(', ')]):
            if args.verbosity > 0:
                print('skipped', instr_name, 'due to Not64BitMode', file=sys.stderr)
            continue
        if instr[('bit', 'isCodeGenOnly')]:
            if args.verbosity > 0:
                print('skipped', instr_name, 'due to isCodeGenOnly = True', file=sys.stderr)
            continue
        #if instr[('bit', 'isAsmParserOnly')]:
        #    if args.verbosity > 0:
        #        print('skipped', instr_name, 'due to isAsmParserOnly = True', file=sys.stderr)
        #    continue
        # FIXME
        if (instr_name in ['CMPPDrri', 'CMPPSrri', 'CMPSDrr', 'CMPSSrr', 'RDSSPD', 'RDSSPQ',
                           'VMREAD64rr', 'VMWRITE64rr', 'VPCLMULQDQYrr']
                or any([instr_name.startswith(p) for p in ['MMX_', 'VPCMP', 'VCMP']])):
            if args.verbosity > 0:
                print('skipped', instr_name, 'due to blacklisted instrunction name:', instr_name,
                      file=sys.stderr)
            continue
    
        # Build Instruction Info Dictionary
        instr_info = collections.OrderedDict(
            [('asm string', convert_asm_to_att(instr[('string', 'AsmString')])),
             ('source operands', {}),
             ('destination operands', {}),
             ('uses', instr[('list<Register>', 'Uses')]),
             ('defines', instr[('list<Register>', 'Defs')])])
        operands = instr[('dag', 'OutOperandList')]
        for m in re.finditer(r'(?P<reg_class>[a-zA-Z0-9_]+):(?P<reg_name>\$[a-z0-9]+)', operands):
            d = m.groupdict()
            llvm_types = []
            instr_info['destination operands'][d['reg_name']] = llvm_types
        
            reg_data = data[d['reg_class']]
            if ('ValueType', 'Type') in reg_data:
                vt = reg_data[('ValueType', 'Type')]
                llvm_types.append(evt_to_llvm_type(vt))
            elif ('list<ValueType>', 'RegTypes') in reg_data:
                for vt in reg_data[('list<ValueType>', 'RegTypes')].split(', '):
                    llvm_types.append(evt_to_llvm_type(vt))

        # Get opernad information and filter all unsupported operand types (e.g., memory references)
        try:
            instr_info['source operands'] = convert_operands(instr[('dag', 'InOperandList')], data)
            instr_info['destination operands'] = convert_operands(instr[('dag', 'OutOperandList')], data)
        except ValueError as e:
            if args.verbosity > 0:
                print('skipped {} {}'.format(instr_name, e), file=sys.stderr)
            continue
    
        # Parse Constraint string reduce number of variables
        for c in instr[('string', 'Constraints')].split(','):
            c = c.strip()
            m = re.match(r'(?P<r1>\$[a-zA-Z0-9_]+)\s*=\s*(?P<r2>\$[a-zA-Z0-9_]+)', c)
            if m:
                d = m.groupdict()
                rename_key(instr_info['source operands'], d['r1'], d['r2'])
                rename_key(instr_info['destination operands'], d['r1'], d['r2'])
                instr_info['asm string'] = instr_info['asm string'].replace(d['r1'], d['r2'])
            elif c and not c.startswith('@earlyclobber'):
                print('not machted:', c, m)
    
        instr_data[instr_name] = instr_info


        # Build op.Instruction
        # Build registers for source (in) and destination (out) operands
        source_operands = []
        try:
            for so_name, so_type in instr_info['source operands'].items():
                if len(so_type) != 1:
                    # FIXME which one to select?
                    pass
                source_operands.append(build_operand(so_type[0][0], so_type[0][1]))
        
            destination_operand = None
            if len(instr_info['destination operands']) < 1:
                # FIXME use "uses" and "defines"
                continue
            elif len(instr_info['destination operands']) > 1:
                raise ValueError("Multiple destination operands are not supported")
            for do_name, do_type in instr_info['destination operands'].items():
                if len(do_type) < 1:
                    continue
                elif len(do_type) > 1:
                    # FIXME which one to select?
                    pass
                if do_type[0][0] in ['r', 'x']:
                    destination_operand = op.Register(do_type[0][1], do_type[0][0])
                else:
                    raise ValueError('Destination operand is not a register')
        except ValueError as e:
            print("skipped", instr_name, str(e), file=sys.stderr)
            continue
        
        # Build instruction string from asm string
        # Sorting by var_name length is necessary to not replace "$src" in "$src1"
        instr_str = instr_info['asm string']
        for i, var_name in sorted(enumerate(
                itertools.chain(instr_info['destination operands'], instr_info['source operands'])),
                key=lambda x: len(x[1]), reverse=True):
            instr_str = instr_str.replace(var_name, '${}'.format(i))
        
        # Make Instruction object
        instr_op = op.Instruction(
            instruction=instr_str,
            destination_operand=destination_operand,
            source_operands=source_operands)
        instructions[instr_name] = instr_op
        
        # Filter instructions that can not be serialized easily
        if not any([so.llvm_type == instr_op.destination_operand.llvm_type and
                    type(so) == type(instr_op.destination_operand)
                    for so in instr_op.source_operands]):
            # TODO take also "uses" and "defs" into account
            continue
    
        s = op.Serialized([instr_op])
        s.build_ir()
        
        print('{:>12} '.format(instr_name), end='')
        sys.stdout.flush()
        
        # Choose init_value according to llvm_type
        init_values = {r.get_ir_repr(): init_value_by_llvm_type[r.llvm_type]
                       for r in s.get_source_registers()}
        b = bench.IntegerLoopBenchmark(s, init_values)
        #print(instr_name)
        #pprint(instr_data[instr_name])
        #print(i)
        #print(s.build_ir())
        #print(b.build_ir())
        #print(b.get_assembly())
        r = b.build_and_execute(repeat=1)
        print('{:>5.2f} cy/It'.format(min(r['runtimes'])*r['frequency']/r['iterations']))
        
        op.Register.reset()

if __name__ == '__main__':
    main()
