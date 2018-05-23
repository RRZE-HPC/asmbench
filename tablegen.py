#!/usr/bin/env python3

import sys
import collections
import re
from pprint import pprint
import itertools

import op


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
    # i32, i64
    if re.fullmatch(r'i[0-9]+', evt):
        type_ = evt
    # f32 -> float
    elif evt == 'f32':
        type_ = 'float'
    # f64 -> double
    elif evt == 'f64':
        type_ = 'double'
    elif evt == 'f80':
        type_ = 'x86_fp80'
    elif evt == 'f128':
        type_ = 'fp128'
    elif evt == 'x86mmx':
        # TODO
        type_ = 'fp64'
    elif evt == 'iPTR':
        # TODO
        type_ = '*TODO'
    
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


def convert_operands(operands, data):
    '''Take operand string from tablegen and convert it into dictionary.'''
    operands_dict = {}
    for m in re.finditer(r'(?P<reg_class>[a-zA-Z0-9_]+):(?P<reg_name>\$[a-z0-9]+)', operands):
        d = m.groupdict()
        constraint = None
        llvm_types = []
        operands_dict[d['reg_name']] = llvm_types
        
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
            constraint = 'r'
            for vt in reg_data[('list<ValueType>', 'RegTypes')].split(', '):
                llvm_types.append((constraint, evt_to_llvm_type(vt)))
        
        if not llvm_types:
            raise ValueError("no operand types found")

    return operands_dict

data = collections.OrderedDict()

cur_def = None
for l in sys.stdin.readlines():
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

instr_data = collections.OrderedDict()

for instr_name, instr in data.items():
    # Filter non-instruction and uninteresting ones
    if ('dag', 'OutOperandList') not in instr:
        print('skipped', instr_name, 'due to missing OutOperandList', file=sys.stderr)
        continue
    if not instr[('string', 'AsmString')]:
        print('skipped', instr_name, 'due to empty asm string', file=sys.stderr)
        continue
    if instr[('string', 'AsmString')].startswith('#') or re.match(r'^[A-Z]+', instr[('string', 'AsmString')]):
        print('skipped', instr_name, 'due to strange asm string:', instr[('string', 'AsmString')],
              file=sys.stderr)
        continue
    if '%' in instr[('string', 'AsmString')]:
        print('skipped', instr_name, 'due to hardcoded register in asm string:',
              instr[('string', 'AsmString')], file=sys.stderr)
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

pprint(instr_data)
print(len(instr_data), 'relevant instructions found', file=sys.stderr)

def build_operand(op_constraint, op_type):
    if op_constraint == 'r':
        return op.Register(op_type)
    elif op_constraint == 'i':
        return op.Immediate(op_type, '1')
    else:
        raise ValueError("unsupported llvm constraint")


# Build op.Instruction
instructions = []
for instr_name, instr in instr_data.items():
    # Build registers for source (in) and destination (out) operands
    source_operands = []
    try:
        for so_name, so_type in instr['source operands'].items():
            if len(so_type) != 1:
                # FIXME which one to select?
                pass
            source_operands.append(build_operand(so_type[0][0], so_type[0][1]))
        
        destination_operand = None
        if len(instr['destination operands']) < 1:
            # FIXME use "uses" and "defines"
            continue
        elif len(instr['destination operands']) > 1:
            raise ValueError("Multiple destination operands are not supported")
        for do_name, do_type in instr['destination operands'].items():
            if len(do_type) < 1:
                continue
            elif len(do_type) > 1:
                # FIXME which one to select?
                pass
            if do_type[0][0] == 'r':
                destination_operand = op.Register(do_type[0][1])
            else:
                raise ValueError('Destination operand is not a register')
    except ValueError as e:
        print("skipped", instr_name, str(e), file=sys.stderr)
        continue
    
    # Build instruction string from asm string
    instr_str = instr['asm string']
    i = 0
    for var_name in itertools.chain(instr['destination operands'], instr['source operands']):
        instr_str = instr_str.replace(var_name, '${}'.format(i))
        i += 1
    
    # Make Instruction object
    instructions.append(op.Instruction(
        instruction=instr_str,
        destination_operand=destination_operand,
        source_operands=source_operands))

pprint(instructions)
