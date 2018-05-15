#!/usr/bin/env python3

import sys
import collections
import re
from pprint import pprint


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
        print(raw, m.groups())
        return tuple([e.strip(',') for e in m.group(1).split(' ') if e is not None])
    # tuple
    m = re.match(r'^{(.*)}$', raw)
    if type_ in ['int', 'bit'] and m:
        return tuple([translate_to_py(e, type_, line) for e in m.group(1).split(',')])
    # list
    m = re.match(r'^\[(.*)\]$', raw)
    type_m = re.match(r'list<(.+)>', type_)
    if type_m and m:
        return [translate_to_py(e, type_m.group(1), line) for e in m.group(1).split(',')]
    # constant/reference(?)
    if re.match(r'^[A-Za-z0-9_]+$', raw):
        return raw
    # function call(?)
    if re.match(r'^[a-zA-Z]+\(.*\)$', raw):
        return raw
    # code
    if type_ == 'code':
        # don't know what todo
        return raw
    
    print('unmachted type:', raw)
    print('in line:', line)
    sys.exit(1)

data = collections.OrderedDict()
cur_def = None
for l in sys.stdin.readlines():
    if cur_def is None:
        m = re.match(r'^def (?P<name>[A-Za-z0-9]+) {', l)
        if m:
            cur_def = m.group(1)
            data[cur_def] = collections.OrderedDict()
    else:
        if l.startswith('}'):
            if cur_def is not None:
                print(cur_def)
                pprint(data[cur_def])
            cur_def = None
            continue
        m = re.match(r'(?P<type>[A-Za-z]+) (?P<name>[A-Za-z]+) = (?P<value>.+);$', l.strip())
        if m:
            g = m.groupdict()
            # Handle value types
            value = translate_to_py(g['value'], g['type'], l)
            data[cur_def][(g['type'], g['name'])] = value