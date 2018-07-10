#!/usr/bin/env python3

import sys
import textwrap

import collections
import re
import itertools
import argparse
import random
from pprint import pprint

from asmjit import op, bench


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

        if i == len(raw) - 1:
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
        return raw
    # tuple
    m = re.match(r'^{(.*)}$', raw)
    if type_ in ['int', 'bit'] and m:
        return tuple([translate_to_py(e, type_, line) for e in m.group(1).split(',')])
    # list
    m = re.match(r'^\[(.*)\]$', raw)
    type_m = re.match(r'list<(.+)>', type_)
    if type_m and m:
        return m.group(1)
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
    """Take operand string from tablegen and convert it into dictionary."""
    operands_dict = {}
    for m in re.finditer(r'(?P<reg_class>[a-zA-Z0-9_]+):(?P<reg_name>\$[a-z0-9]+)', operands):
        d = m.groupdict()
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
                # constraint = 'm'
                raise ValueError("due to memory operand")
            elif op_type == 'OPERAND_IMMEDIATE':
                constraint = 'i'
                vt = reg_data[('ValueType', 'Type')]
                llvm_types.append((constraint, evt_to_llvm_type(vt)))
                continue
            elif op_type == 'OPERAND_PCREL':
                raise ValueError("due to pcrel operand")
            elif op_type == "OPERAND_REGISTER":
                # constraint = 'r'
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
    if re.match(r'[0-9]+', op_constraint):
        return op.Register(op_type, op_constraint)
    if op_constraint in ['r', 'x']:
        return op.Register(op_type, op_constraint)
    elif op_constraint == 'i':
        return op.Immediate(op_type, '1')
    else:
        raise ValueError("unsupported llvm constraint")


def read_tablegen_output(f):
    data = collections.OrderedDict()

    cur_def = None
    for l in f.readlines():
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
    return data


def extract_instruction_information(data, verbosity=0):
    instr_data = collections.OrderedDict()

    for instr_name, instr in data.items():
        # Filter non-instruction and uninteresting or unsupported ones
        if ('dag', 'OutOperandList') not in instr:
            if verbosity > 0:
                print('skipped', instr_name, 'due to missing OutOperandList', file=sys.stderr)
            continue
        if not instr[('string', 'AsmString')]:
            if verbosity > 0:
                print('skipped', instr_name, 'due to empty asm string', file=sys.stderr)
            continue
        if (instr[('string', 'AsmString')].startswith('#') or
                re.match(r'^[A-Z]+', instr[('string', 'AsmString')])):
            if verbosity > 0:
                print('skipped', instr_name, 'due to strange asm string:',
                      instr[('string', 'AsmString')],
                      file=sys.stderr)
            continue
        if '%' in instr[('string', 'AsmString')]:
            if verbosity > 0:
                print('skipped', instr_name, 'due to hardcoded register in asm string:',
                      instr[('string', 'AsmString')], file=sys.stderr)
            continue
        if instr[('bit', 'isCodeGenOnly')]:
            if verbosity > 0:
                print('skipped', instr_name, 'due to isCodeGenOnly = True', file=sys.stderr)
            continue

        # TODO is this necessary?
        # if instr[('bit', 'isAsmParserOnly')]:
        #    if args.verbosity > 0:
        #        print('skipped', instr_name, 'due to isAsmParserOnly = True', file=sys.stderr)
        #    continue

        # Build Instruction Info Dictionary
        instr_info = collections.OrderedDict(
            [('asm string', convert_asm_to_att(instr[('string', 'AsmString')])),
             ('source operands', {}),
             ('destination operands', {}),
             ('uses', instr[('list<Register>', 'Uses')]),
             ('defines', instr[('list<Register>', 'Defs')]),
             ('predicates', instr[('list<Predicate>', 'Predicates')].split(', '))])
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

        # Get operand information and filter all unsupported operand types (e.g., memory references)
        try:
            instr_info['source operands'] = convert_operands(instr[('dag', 'InOperandList')], data)
            instr_info['destination operands'] = convert_operands(instr[('dag', 'OutOperandList')],
                                                                  data)
        except ValueError as e:
            if verbosity > 0:
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
    return instr_data


def filter_relevant_instructions_from_info(instruction_data, verbosity=0):
    """
    Return name of instructions that can be run on this architecture and do not have other issues
    """
    for instr_name, instr_info in instruction_data.items():
        # TODO Automatically detext feature set
        if any([p in ['HasTBM', 'Not64BitMode', 'HasMPX', 'HasSSE4A', 'HasGFNI', 'HasDQI', 'HasBWI',
                      'HasAVX512', 'Has3DNow', 'HasSHA', 'HasVAES', 'HasVLX', 'HasERI', 'HasFMA4',
                      'HasXOP', 'HasVBMI2', 'HasCDI', 'HasVNNI', 'HasVBMI', 'HasIFMA',
                      'HasBITALG', 'HasVPOPCNTDQ']
                for p in instr_info['predicates']]):
            if verbosity > 0:
                print('skipped', instr_name, 'due to predicates', file=sys.stderr)
            continue
        # FIXME
        if (instr_name in ['CMPPDrri', 'CMPPSrri', 'CMPSDrr', 'CMPSSrr', 'RDSSPD', 'RDSSPQ',
                           'VMREAD64rr', 'VMWRITE64rr', 'VPCLMULQDQYrr']
                or any([instr_name.startswith(p) for p in ['MMX_', 'VPCMP', 'VCMP']])):
            if verbosity > 0:
                print('skipped', instr_name, 'due to blacklisted instruction name:', instr_name,
                      file=sys.stderr)
            continue
        yield instr_name


def filter_relevant_instruction_from_operation(instructions, verbosity=0):
    for instr_name, instr_op in instructions.items():
        # Filter instructions that can not be serialized easily
        if not can_serialize(instr_op):
            if verbosity > 0:
                print("skipped", instr_name, " will not serialize.")
            continue
        yield instr_name


def build_instruction_objects(instruction_data, instruction_names=None, verbosity=0):
    if instruction_names is None:
        instruction_names = instruction_data.keys()

    for instr_name in instruction_names:
        instr_info = instruction_data[instr_name]
        # Build op.Instruction
        # Build registers for source (in) and destination (out) operands
        source_operands = []
        destination_operand = None
        try:
            if len(instr_info['destination operands']) < 1:
                # FIXME use "uses" and "defines"
                raise ValueError('Missing destination operand(s)')
            elif len(instr_info['destination operands']) > 1:
                raise ValueError("Multiple destination operands are not supported")
            for do_name, do_type in instr_info['destination operands'].items():
                if len(do_type) < 1:
                    raise ValueError('No destination operand type')
                elif len(do_type) > 1:
                    # FIXME which one to select?
                    pass
                if do_type[0][0] in ['r', 'x']:
                    destination_operand = op.Register(do_type[0][1], do_type[0][0])
                else:
                    raise ValueError('Destination operand is not a register')

            for so_name, so_type in instr_info['source operands'].items():
                if len(so_type) != 1:
                    # FIXME which one to select?
                    pass
                if so_name in instr_info['destination operands']:
                    # If this operand is both source AND destination, the source constraint string
                    # needs to reference the destination
                    if len(instr_info['destination operands']) > 1:
                        raise ValueError("Multiple destination operands are not supported")
                    # FIXME if multiple destinations are supported, the intere needs to match dst
                    # order
                    constraint = '0'
                else:
                    constraint = so_type[0][0]
                source_operands.append(build_operand(constraint, so_type[0][1]))
        except ValueError as e:
            if verbosity > 0:
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

        yield (instr_name, instr_op)


# noinspection PyUnusedLocal
def main():
    parser = argparse.ArgumentParser(description='Build benchmarks from TableGen output.')
    parser.add_argument('input', metavar='IN', type=argparse.FileType('r'), default=sys.stdin,
                        help='Input file (default stdin)')
    parser.add_argument("-v", "--verbosity", action="count", default=0,
                        help="increase output verbosity")
    args = parser.parse_args()

    data = read_tablegen_output(args.input)
    instruction_data = extract_instruction_information(data, verbosity=args.verbosity)
    rel_instruction_names = list(
        filter_relevant_instructions_from_info(instruction_data, verbosity=args.verbosity))
    instructions = collections.OrderedDict(build_instruction_objects(
            instruction_data, rel_instruction_names, verbosity=args.verbosity))
    rel_instruction_names = [
        iname
        for iname in filter_relevant_instruction_from_operation(instructions, verbosity=args.verbosity)
        if iname in rel_instruction_names]

    # Setup LLVM environment
    bench.setup_llvm()

    # Benchmark TP and Lat for each instruction
    # for instr_name, instr_op in instructions.items():
    #     tp, lat = bench.bench_instructions([instr_op])
    #     print("{:>12} {:>5.2f} {:>5.2f}".format(instr_name, tp, lat))

    # Benchmark TP and Lat for all valid instruction pairs
    # for a,b in itertools.combinations(instructions, 2):
    #    ia = instructions[a]
    #    ib = instructions[b]
    #    if not (can_serialize(ia) and can_serialize(ib) and
    #            ia.get_destination_registers()[0].llvm_type
    #            == ib.get_destination_registers()[0].llvm_type):
    #        continue
    #    print("{:>12} {:<12} ".format(a,b), end="")
    #    tp, lat = bench.bench_instructions([instructions[a], instructions[b]]#)
    #    print("{:>5.2f} {:>5.2f}".format(tp, lat))

    if args.verbosity > 0:
        print('instructions:', len(instructions))
        print('2-combinations:', len(list(combined_instructions(instructions, 2))))

    # Build subgroups for each return type
    instructions_ret_type = collections.defaultdict(collections.OrderedDict)
    if args.verbosity > 0:
        for ret_type in rel_instruction_names:
            print(ret_type, 'has', len(instructions_ret_type[ret_type]), 'instructions')

    # Benchmark random instruction sequences
    for instr_name, instr_op in instructions.items():
        instructions_ret_type[instr_op.get_destination_registers()[0].llvm_type][
            instr_name] = (instr_name, instr_op)
    # Constructing random benchmarks, one for each return type
    #random.seed(42)
    #parallel_factor = 8
    #for t in sorted(instructions_ret_type):
    #    valid = False
    #    while not valid:
    #        selected_names, selected_instrs = zip(
    #            *[random.choice(list(instructions_ret_type[t].values())) for i in range(10)])
    #
    #        if not all([can_serialize(i) for i in selected_instrs]):
    #            continue
    #        else:
    #            valid = True
    #
    #        serial = op.Serialized(selected_instrs)
    #        p = op.Parallelized([serial] * parallel_factor)
    #
    #        init_values = [op.init_value_by_llvm_type[reg.llvm_type] for reg in
    #                       p.get_source_registers()]
    #        b = bench.IntegerLoopBenchmark(p, init_values)
    #        print('## Selected Instructions')
    #        print(', '.join(selected_names))
    #        print('## Generated Assembly ({}x parallel)'.format(parallel_factor))
    #        print(b.get_assembly())
    #        #pprint(selected_instrs)
    #        r = b.build_and_execute(repeat=4, min_elapsed=0.1, max_elapsed=0.2)
    #        r['parallel_factor'] = parallel_factor
    #        print('## Detailed Results')
    #        pprint(r)
    #        print("minimal throughput: {:.2f} cy".format(
    #            min(r['runtimes'])/r['iterations']*r['frequency']/parallel_factor))

    # Reduce to 100 instructions:
    #instructions = dict(list(instructions.items())[:100])

    # Reduce to set of instructions used in Stream Triad:
    instructions = {k: v for k,v in instructions.items() if k in ['ADD32ri', 'ADD64ri32', 'INC64r', 'SUB32ri', 'VADDPDYrr', 'VADDSDrr', 'VADDSSrr', 'VCVTSI642SSrr', 'VFMADD213PDYr', 'VFMADD213PDr', 'VFMADD213PSYr', 'VFMADD213PSr', 'VFMADD213SDr', 'VFMADD213SSr', 'VINSERTF128rr', 'VMULPDYrr', 'VMULSDrr_Int', 'VMULSSrr_Int', 'VSUBSDrr_Int', 'VSUBSSrr_Int']}

    random.seed(23)
    instructions_per_run = 3
    parallel_factor = 4
    print(textwrap.dedent("""
        # This file contains example data measured on an Intel I7-6700HQ with 2.6GHz with Turbo mode
        # disabled.
        
        # Comments are possible everywhere after hash symbols.
        
        # This part contains necessary configuration information.
        configuration:
            model: three-level   # we assume that instructions are decomposed into uops
            num_ports: 7         # our hardware has 4 execution ports
            num_uops_per_insn: 4 # the maximal number of uops into which an instruction can be decomposed
            slack_limit: 0.0     # relative margin of error for cycle measurements
        
        
        # Here follows a list of experiments.
        """))
    for i in range(100):
        selected_names, selected_instrs = zip(*[random.choice(list(instructions.items()))
                                                for i in range(instructions_per_run)])
        print("experiment:")
        p = op.Parallelized(selected_instrs*parallel_factor)
        b = bench.IntegerLoopBenchmark(p)
        print('    instructions:')
        print('        '+('\n        '.join(selected_names)))
        if args.verbosity > 0:
            print('    ir:')
            print(textwrap.indent(b.build_ir(), ' '*8))
            print('    asm:')
            print(textwrap.indent(b.get_assembly(), ' '*8))
        r = b.build_and_execute(repeat=4, min_elapsed=0.1, max_elapsed=0.2)
        r['parallel_factor'] = parallel_factor
        if args.verbosity > 0:
            print('    detailed_result:')
            pprint(r, indent=8)
        print("    cycles: {:.2f}".format(
            min(r['runtimes'])/r['iterations']*r['frequency']/parallel_factor))

def can_serialize(instr):
    if not any([so.llvm_type == instr.destination_operand.llvm_type and
                isinstance(so, instr.destination_operand.__class__)
                for so in instr.source_operands]):
        # TODO take also "uses" and "defs" into account
        return False
    return True


def combined_instructions(instructions, length):
    for instr_names in itertools.combinations(instructions, length):
        instrs = [instructions[n] for n in instr_names]
        dst_types = list([i.get_destination_registers()[0].llvm_type for i in instrs])
        if not all([can_serialize(i) for i in instrs]) and dst_types[1:] == dst_types[:-1]:
            continue
        yield instrs


if __name__ == '__main__':
    main()
