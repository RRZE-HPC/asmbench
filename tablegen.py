#!/usr/bin/env python3

import sys
import collections
import re
from pprint import pprint


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
    
    print('unmachted type:', type_)
    print('raw:', raw)
    print('in line:', line)
    sys.exit(1)

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
            #if cur_def is not None:
            #    print(cur_def)
            #    pprint(data[cur_def])
            #v = data[cur_def]
            #try:
            #    print('ASM', v[('string', 'AsmString')])
            #    print('OUT', v[('dag', 'OutOperandList')])
            #    print('IN ', v[('dag', 'InOperandList')])
            #except KeyError:
            #    # ignoring non-instructions
            #    pass
            cur_def = None
            continue
        m = re.match(r'(?P<type>[A-Za-z<>]+) (?P<name>[A-Za-z]+) = (?P<value>.+);$', l.strip())
        if m:
            g = m.groupdict() 
            # Handle value types
            value = translate_to_py(g['value'], g['type'], l)
            data[cur_def][(g['type'], g['name'])] = value


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


for instr_name in [k for k, v in data.items() if ('dag', 'OutOperandList') in v]:
    if 'm' in instr_name:
        print('skipped', instr_name, 'due to memref')
        continue
    instr = data[instr_name]
    print(instr_name, instr[('string', 'AsmString')])
    print('   Uses', instr[('list<Register>', 'Uses')])
    print('   Defines', instr[('list<Register>', 'Defs')])
    operands = instr[('dag', 'OutOperandList')] + instr[('dag', 'InOperandList')]
    print('  ', operands)
    for m in re.finditer(r'(?P<reg_class>[a-zA-Z0-9_]+):(?P<reg_name>\$[a-z0-9]+)', operands):
        d = m.groupdict()
        print('    ', d['reg_class'], d['reg_name'])
        reg_data = data[d['reg_class']]
        if ('ValueType', 'Type') in reg_data:
            vt = reg_data[('ValueType', 'Type')]
            print('      ', vt)
            print('      ', vt, evt_to_llvm_type(vt))
        elif ('list<ValueType>', 'RegTypes') in reg_data:
            print('      ', reg_data[('list<ValueType>', 'RegTypes')])
            for vt in reg_data[('list<ValueType>', 'RegTypes')].split(', '):
                print('      ', vt, evt_to_llvm_type(vt))

target = {
    'VADDPDYrm': {
        'asm_string': 'vaddpd $2 $0',
        'source operands': ['4 x double', '4 x double'],
        'destination operands': ['4 x double'],
    }
}

#op_types = set()
#for v in data.values():
#    ops = v.get(('dag', 'OutOperandList'), [None,])[1:] + \
#          v.get(('dag', 'InOperandList'), [None,])[1:]
#    for o in ops:
#        op_types.add(o.split(':')[0])
#print(op_types)
