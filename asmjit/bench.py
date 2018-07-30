#!/usr/bin/env python3
import ctypes
import time
import textwrap
import itertools
import re
from pprint import pprint
import tempfile
import subprocess
import sys

import llvmlite.binding as llvm
import psutil
try:
    from kerncraft import iaca
except ImportError:
    iaca = None

from . import op


def setup_llvm():
    llvm.initialize()
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
    llvm.initialize_native_asmparser()


def uniquify(l):
    # Uniquify list while preserving order
    seen = set()
    return [x for x in l if x not in seen and not seen.add(x)]


class Benchmark:
    def __repr__(self):
        return '{}({})'.format(
            self.__class__.__name__,
            ', '.join(['{}={!r}'.format(k, v) for k, v in self.__dict__.items()
                       if not k.startswith('_')]))

    @staticmethod
    def prepare_arguments(previous_args=None, time_factor=1.0):
        """Build argument tuple, to be passed to low level function."""
        if previous_args is None:
            return 10000000,
        else:
            try:
                return int(previous_args[0] * time_factor),
            except OverflowError:
                return previous_args[0]*10,

    @staticmethod
    def get_iterations(args) -> int:
        """Return number of iterations performed, based on lower level function arguments."""
        return args[0]

    def build_ir(self):
        raise NotImplementedError()

    def get_llvm_module(self, iaca_marker=False):
        """Build and return LLVM module from LLVM IR code."""
        ir = self.build_ir(iaca_marker=iaca_marker)
        return llvm.parse_assembly(ir)

    def get_target_machine(self):
        """Instantiate and return target machine."""
        features = llvm.get_host_cpu_features().flatten()
        cpu = '' # llvm.get_host_cpu_name()  # Work around until ryzen problems are fixed
        return llvm.Target.from_default_triple().create_target_machine(
             cpu=cpu, features=features, opt=3)

    def get_assembly(self, iaca_marker=False):
        """Compile and return assembly from LLVM module."""
        tm = self.get_target_machine()
        tm.set_asm_verbosity(0)
        asm = tm.emit_assembly(self.get_llvm_module(iaca_marker=iaca_marker))
        # Remove double comments
        asm = re.sub(r'## InlineAsm End\n\s*## InlineAsm Start\n\s*', '', asm)
        return asm

    def get_function_ctype(self):
        return ctypes.CFUNCTYPE(ctypes.c_int64, ctypes.c_int64)

    def get_iaca_analysis(self, arch):
        """Compile and return IACA analysis."""
        if iaca is None:
            raise ValueError("kerncraft not installed. IACA analysis is not supported.")
        tm = self.get_target_machine()
        tmpf = tempfile.NamedTemporaryFile("wb")
        tmpf.write(tm.emit_object(self.get_llvm_module(iaca_marker=True)))
        tmpf.flush()
        return iaca.iaca_analyse_instrumented_binary(tmpf.name, arch)

    def build_and_execute(self, repeat=10, min_elapsed=0.1, max_elapsed=0.3):
        # Compile the module to machine code using MCJIT
        tm = self.get_target_machine()
        runtimes = []
        return_values = []
        args = self.prepare_arguments()
        with llvm.create_mcjit_compiler(self.get_llvm_module(), tm) as ee:
            ee.finalize_object()

            # Obtain a pointer to the compiled 'sum' - it's the address of its JITed
            # code in memory.
            cfptr = ee.get_function_address('test')

            # To convert an address to an actual callable thing we have to use
            # CFUNCTYPE, and specify the arguments & return type.
            cfunc = self.get_function_ctype()(cfptr)

            # Now 'cfunc' is an actual callable we can invoke
            # TODO replace time.clock with a C implemententation for less overhead
            # TODO return result in machine readable format
            fixed_args = False
            for i in range(repeat):
                tries = 0
                while True:
                    if tries > 10:
                        raise RuntimeError("Unable to measure non-zero runtime.")
                    tries += 1
                    start = time.perf_counter()
                    ret = cfunc(*args)
                    end = time.perf_counter()
                    elapsed = end - start
                    if ret != args[0]-1:
                        raise RuntimeError(
                            "Return value {} is invalid, should have been {}.".format(ret, args[0]-1))
                    if not fixed_args and (elapsed < min_elapsed or elapsed > max_elapsed):
                        target_elapsed = 2 / 3 * min_elapsed + 1 / 3 * max_elapsed
                        factor = target_elapsed / elapsed
                        args = self.prepare_arguments(previous_args=args, time_factor=factor)
                        continue
                    else:
                        # After we have the right argument choice, we keep it.
                        fixed_args = True
                        break
                return_values.append(ret)
                runtimes.append(elapsed)

        return {'iterations': self.get_iterations(args),
                'arguments': args,
                'runtimes': runtimes,
                'frequency': psutil.cpu_freq().current * 1e6,
                'returned': return_values}


class LoopBenchmark(Benchmark):
    def __init__(self, root_synth, init_values=None, loop_carried_dependencies=True):
        super().__init__()
        self.root_synth = root_synth
        self.init_values = init_values or root_synth.get_default_init_values()
        self.loop_carried_dependencies = loop_carried_dependencies

        if len(root_synth.get_source_registers()) != len(self.init_values):
            raise ValueError("Number of init values and source registers do not match.")

    def get_source_names(self):
        return ['%in.{}'.format(i) for i in range(len(self.root_synth.get_source_registers()))]

    def get_destination_names(self):
        return ['%out.{}'.format(i) for i in
                range(len(self.root_synth.get_destination_registers()))]

    def get_phi_code(self):
        if not self.loop_carried_dependencies:
            return ''
        # Compile loop carried dependencies
        lcd = []
        # Change in naming (src <-> dst) is on purpose!
        srcs = self.root_synth.get_destination_registers()
        dsts = self.root_synth.get_source_registers()
        # cycle iterator is used to not only reuse a single destination, but go through all of them
        srcs_it = itertools.cycle(enumerate(srcs))
        matched = False
        last_match_idx = len(srcs) - 1
        for dst_idx, dst in enumerate(dsts):
            for src_idx, src in srcs_it:
                if src.llvm_type == dst.llvm_type:
                    lcd.append([dst,
                                self.get_source_names()[dst_idx],
                                self.init_values[dst_idx],
                                src,
                                self.get_destination_names()[src_idx]])
                    matched = True
                    last_match_idx = src_idx
                    break
                # since srcs_it is an infinity iterator, we need to abort after a complete cycle
                if src_idx == last_match_idx:
                    break
        if not matched:
            raise ValueError("Unable to match source to any destination.")

        code = ''
        for dst_reg, dst_name, init_value, src_reg, src_name in lcd:
            assert dst_reg.llvm_type == src_reg.llvm_type, \
                "Source and destination types do not match"
            code += ('{dst_name} = phi {llvm_type} [{init_value}, %"entry"], '
                     '[{src_name}, %"loop"]\n').format(
                llvm_type=dst_reg.llvm_type,
                dst_name=dst_name,
                init_value=init_value,
                src_name=src_name)

        # Add extra phi for constant values. Assuming LLVM will optimize them "away"
        for dst_idx, dst in enumerate(dsts):
            if dst not in [d for d, dn, i, s, sn in lcd]:
                code += ('{dst_reg} = phi {llvm_type} [{init_value}, %"entry"], '
                         '[{init_value}, %"loop"]\n').format(
                    llvm_type=dst.llvm_type,
                    dst_reg=self.get_source_names()[dst_idx],
                    init_value=self.init_values[dst_idx])

        return code

    def build_ir(self):
        raise NotImplementedError()


class IntegerLoopBenchmark(LoopBenchmark):
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
            define i64 @"test"(i64 %"N")
            {{
            entry:
              %"loop_cond" = icmp slt i64 0, %"N"
              br i1 %"loop_cond", label %"loop", label %"end"

            loop:
              %"loop_counter" = phi i64 [0, %"entry"], [%"loop_counter.1", %"loop"]
            {phi}
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
            loop_body=textwrap.indent(
                self.root_synth.build_ir(self.get_destination_names(),
                                         self.get_source_names()), '  '),
            phi=textwrap.indent(self.get_phi_code(), '  '),
            iaca_start_marker=iaca_start_marker,
            iaca_stop_marker=iaca_stop_marker)

        return ir


def bench_instructions(instructions, serial_factor=8, parallel_factor=4, throughput_serial_factor=8,
                       serialize=False, verbosity=0, iaca_comparison=None,
                       repeat=4, min_elapsed=0.1, max_elapsed=0.2):
    not_serializable = False
    try:
        # Latency Benchmark
        if verbosity > 0:
            print('## Latency Benchmark')
        p_instrs = []
        if not serialize:
            for i in instructions:
                p_instrs.append(op.Serialized([i] * serial_factor))
        else:
            p_instrs = [op.Serialized(instructions * serial_factor)]
        p = op.Parallelized(p_instrs)
        b = IntegerLoopBenchmark(p)
        if verbosity >= 3:
            print('### LLVM IR')
            print(b.build_ir())
        if verbosity >= 2:
            print('### Assembly')
            print(b.get_assembly())
        if verbosity >= 3:
            print('### IACA Analysis')
            print(b.get_iaca_analysis('SKL')['output'])
        result = b.build_and_execute(
            repeat=repeat, min_elapsed=min_elapsed, max_elapsed=max_elapsed)
        lat = min(*[(t / serial_factor) * result['frequency'] / result['iterations']
                    for t in result['runtimes']])
        result['latency'] = lat
        if verbosity > 0:
            print('### Detailed Results')
            pprint(result)
            print()
    except op.NotSerializableError as e:
        print("Latency measurement not possible:", e)
        not_serializable = True

    if not_serializable:
        throughput_serial_factor = 1
        print("WARNING: throughput_serial_factor has be set to 1.")

    # Throughput Benchmark
    if verbosity > 0:
        print('## Throughput Benchmark')
    p_instrs = []
    if not serialize:
        for i in instructions:
            p_instrs.append(op.Serialized([i] * throughput_serial_factor))
    else:
        p_instrs = [op.Serialized(instructions * throughput_serial_factor)]
    p = op.Parallelized(p_instrs * parallel_factor)
    b = IntegerLoopBenchmark(p)
    if verbosity >= 3:
        print('### LLVM IR')
        print(b.build_ir())
    if verbosity >= 2:
        print('### Assembly')
        print(b.get_assembly())
    if verbosity >= 3:
        print('### IACA Analysis')
        print(b.get_iaca_analysis('SKL')['output'])
    result = b.build_and_execute(
        repeat=repeat, min_elapsed=min_elapsed, max_elapsed=max_elapsed)
    tp = min(
        [(t / throughput_serial_factor / parallel_factor) * result['frequency'] / result['iterations']
         for t in result['runtimes']])
    result['throughput'] = tp
    if iaca_comparison is not None:
        iaca_analysis = b.get_iaca_analysis(iaca_comparison)
        result['iaca throughput'] = iaca_analysis['throughput']/(
                parallel_factor * throughput_serial_factor)
    if verbosity > 0:
        print('### Detailed Results')
        pprint(result)
        print()
    if verbosity > 1 and iaca_comparison is not None:
        print('### IACA Results')
        print(iaca_analysis['output'])
        print('!!! throughput_serial_factor={} and parallel_factor={}'.format(
            throughput_serial_factor, parallel_factor))

    # Result compilation
    return lat, tp


if __name__ == '__main__':
    setup_llvm()

    i1 = op.Instruction(
        instruction='add $2, $0',
        destination_operand=op.Register('i64', 'r'),
        source_operands=[op.Register('i64', '0'), op.Immediate('i64', '1')])
    i2 = op.Instruction(
        instruction='sub $2, $0',
        destination_operand=op.Register('i64', 'r'),
        source_operands=[op.Register('i64', '0'), op.Immediate('i64', '1')])
    s = op.Serialized([i1, i2])
    i3 = op.Instruction(
        instruction='add $2, $0',
        destination_operand=op.Register('i64', 'r'),
        source_operands=[op.Register('i64', '0'), op.Register('i64', 'r')])
    i4 = op.Instruction(
        instruction='sub $2, $0',
        destination_operand=op.Register('i64', 'r'),
        source_operands=[op.Register('i64', '0'), op.Immediate('i64', '23')])
    i5 = op.Instruction(
        instruction='add $2, $0',
        destination_operand=op.Register('i64', 'r'),
        source_operands=[op.Register('i64', '0'), op.Immediate('i64', '23')])
    i6 = op.Instruction(
        instruction='add $2, $0',
        destination_operand=op.Register('i64', 'r'),
        source_operands=[op.Register('i64', '0'), op.Register('i64', 'r')])
    s1 = op.Serialized([i1, i2])
    s2 = op.Serialized([s1, i3])
    s3 = op.Serialized([i4, i5])
    p1 = op.Parallelized([i6, s2, s3])
    init_values = ['1' for r in p1.get_source_registers()]
    b = IntegerLoopBenchmark(p1, init_values)
    print(b.build_ir())
    print(b.get_assembly())
    print(b.build_and_execute())

    print(bench_instructions([op.Instruction(
        instruction='add $2, $0',
        destination_operand=op.Register('i64', 'r'),
        source_operands=[op.Register('i64', '0'), op.Immediate('i64', '1')])]))

    # if len(s.get_source_operand_types())
    # b = IntegerLoopBenchmark(loop_body,
    #                          [(type_, dst_reg, '1', src_reg)
    #                           # for type_, dst_reg, src_reg in zip(s.get_last_destination_type(), )])
    # print(b.get_ir())
