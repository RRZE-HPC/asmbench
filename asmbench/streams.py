#!/usr/bin/env python3

import collections
import itertools
import socket
import textwrap

import numpy
import matplotlib.pyplot as plt
import matplotlib as mpl

from asmbench import op, bench
from asmbench import oldjit


type_size = {
    'i32': 4,
    'i64': 8,
    'f32': 4,
    'float': 4,
    'f64': 8,
    'double': 8,
}


class StreamsBenchmark(bench.Benchmark):
    def __init__(self,
                 read_streams=0, read_write_streams=0, write_streams=0,
                 stream_byte_length=0,
                 element_type='i64'):
        super().__init__()
        self.read_streams = read_streams
        self.read_write_streams = read_write_streams
        self.write_streams = write_streams
        self.stream_byte_length = stream_byte_length
        self.element_type = element_type

    def build_ir(self, iaca_marker=False):
        if iaca_marker:
            iaca_start_marker = textwrap.dedent('''\
                call void asm "movl    $$111,%ebx", ""()
                call void asm ".byte   100,103,144", ""()''')
            iaca_stop_marker = textwrap.dedent('''\
                call void asm "movl    $$222,%ebx", ""()
                call void asm ".byte   100,103,144", ""()''')
        else:
            iaca_start_marker = ''
            iaca_stop_marker = ''

        ir = textwrap.dedent('''\
            define i64 @"test"(i64 %"N"{pointer_arguments})
            {{
            entry:
              %"loop_cond" = icmp slt i64 0, %"N"
              br i1 %"loop_cond", label %"loop", label %"end"

            loop:
              %"loop_counter" = phi i64 [0, %"entry"], [%"loop_counter.1", %"loop"]
            {iaca_start_marker}
            {loop_body}
              %"loop_counter.1" = add i64 %"loop_counter", 1
              %"loop_cond.1" = icmp slt i64 %"loop_counter.1", %"N"
              br i1 %"loop_cond.1", label %"loop", label %"end"

            end:
              %"ret" = phi i64 [0, %"entry"], [%"loop_counter", %"loop"]
            {iaca_stop_marker}
              ret i64 %"ret"
            }}
            ''').format(
            pointer_arguments='',
            loop_body='',
            iaca_start_marker=iaca_start_marker,
            iaca_stop_marker=iaca_stop_marker)

        return ir

if __name__ == '__main__':
    bench.setup_llvm()
    sb = StreamsBenchmark()
    print(sb.build_and_execute())

