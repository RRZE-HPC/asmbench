#!/usr/bin/env python3
import ctypes
import time
import textwrap
import itertools

import llvmlite.binding as llvm
import psutil

from asmjit import op


def uniquify(l):
    # Uniquify list while preserving order
    seen = set()
    return [x for x in l if x not in seen and not seen.add(x)]


class Benchmark:
    def __init__(self):
        self._tm = None
        self._llvm_module = None

    def __repr__(self):
        return '{}({})'.format(
            self.__class__.__name__,
            ', '.join(['{}={!r}'.format(k, v) for k, v in self.__dict__.items()
                       if not k.startswith('_')]))

    @staticmethod
    def prepare_arguments(previous_args=None, time_factor=1.0):
        """Build argument tuple, to be passed to low level function."""
        if previous_args is None:
            return 100,
        else:
            return int(previous_args[0] * time_factor),

    @staticmethod
    def get_iterations(args):
        """Return number of iterations performed, based on lower level function arguments."""
        return args[0]

    def build_ir(self):
        raise NotImplementedError()

    def get_llvm_module(self):
        """Build and return LLVM module from LLVM IR code."""
        if self._llvm_module is None:
            self._llvm_module = llvm.parse_assembly(self.build_ir())
            self._llvm_module.verify()
        return self._llvm_module

    def get_target_machine(self):
        """Instantiate and return target machine."""
        if self._tm is None:
            features = llvm.get_host_cpu_features().flatten()
            cpu = llvm.get_host_cpu_name()
            self._tm = llvm.Target.from_default_triple().create_target_machine(
                cpu=cpu, features=features, opt=3)
        return self._tm

    def get_assembly(self):
        """Compile and return assembly from LLVM module."""
        tm = self.get_target_machine()
        tm.set_asm_verbosity(0)
        return tm.emit_assembly(self.get_llvm_module())

    def get_function_ctype(self):
        return ctypes.CFUNCTYPE(ctypes.c_int64, ctypes.c_int64)

    def build_and_execute(self, repeat=10, min_elapsed=0.1, max_elapsed=0.3):
        # Compile the module to machine code using MCJIT
        tm = self.get_target_machine()
        runtimes = []
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
            results = []
            for i in range(repeat):
                while True:
                    start = time.perf_counter()
                    results.append(cfunc(*args))
                    end = time.perf_counter()
                    elapsed = end - start
                    if not fixed_args and (elapsed < min_elapsed or elapsed > max_elapsed):
                        target_elapsed = 2 / 3 * min_elapsed + 1 / 3 * max_elapsed
                        factor = target_elapsed / elapsed
                        args = self.prepare_arguments(previous_args=args, time_factor=factor)
                        continue
                    else:
                        # After we have the right argument choice, we keep it.
                        fixed_args = True
                        break
                runtimes.append(elapsed)

        return {'iterations': self.get_iterations(args),
                'arguments': args,
                'runtimes': runtimes,
                'frequency': psutil.cpu_freq().current * 1e6,
                'results': results}


class LoopBenchmark(Benchmark):
    def __init__(self, root_synth, init_values=None):
        super().__init__()
        self.root_synth = root_synth
        self.init_values = init_values or []

        if len(root_synth.get_source_registers()) != len(self.init_values):
            raise ValueError("Number of init values and source registers do not match.")

    def get_source_names(self):
        return ['%in.{}'.format(i) for i in range(len(self.root_synth.get_source_registers()))]

    def get_destination_names(self):
        return ['%out.{}'.format(i) for i in
                range(len(self.root_synth.get_destination_registers()))]

    def get_phi_code(self):
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

        # Add extra phi for constant values. Assuming LLVM will optimiz them "away"
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
    def build_ir(self):
        return textwrap.dedent('''\
            define i64 @"test"(i64 %"N")
            {{
            entry:
              %"loop_cond" = icmp slt i64 0, %"N"
              br i1 %"loop_cond", label %"loop", label %"end"

            loop:
              %"loop_counter" = phi i64 [0, %"entry"], [%"loop_counter.1", %"loop"]
            {phi}
            {loop_body}
              %"loop_counter.1" = add i64 %"loop_counter", 1
              %"loop_cond.1" = icmp slt i64 %"loop_counter.1", %"N"
              br i1 %"loop_cond.1", label %"loop", label %"end"

            end:
              %"ret" = phi i64 [0, %"entry"], [%"loop_counter", %"loop"]
              ret i64 %"ret"
            }}
            ''').format(
            loop_body=textwrap.indent(
                self.root_synth.build_ir(self.get_destination_names(),
                                         self.get_source_names()), '  '),
            phi=textwrap.indent(self.get_phi_code(), '  '))


def bench_instruction(instruction, serial_factor=8, parallel_factor=8, parallel_serial_factor=8):
    # Latency Benchmark
    s = op.Serialized([instruction] * serial_factor)
    init_values = [op.init_value_by_llvm_type[reg.llvm_type] for reg in s.get_source_registers()]
    b = IntegerLoopBenchmark(s, init_values)
    result = b.build_and_execute(repeat=4, min_elapsed=0.1, max_elapsed=0.2)
    lat = min(*[(t / serial_factor) * result['frequency'] / result['iterations']
                for t in result['runtimes']])

    # Throughput Benchmark
    p = op.Parallelized([op.Serialized([instruction] * parallel_serial_factor)] * parallel_factor)
    init_values = [op.init_value_by_llvm_type[reg.llvm_type] for reg in
                   p.get_source_registers()]
    b = IntegerLoopBenchmark(p, init_values)
    result = b.build_and_execute(repeat=4, min_elapsed=0.1, max_elapsed=0.2)
    tp = min(
        [(t / parallel_serial_factor / parallel_factor) * result['frequency'] / result['iterations']
         for t in result['runtimes']])

    # Result compilation
    return lat, tp


def bench_instructions(instructions, serial_factor=8, parallel_factor=4, parallel_serial_factor=8):
    # Latency Benchmark
    p_instrs = []
    for i in instructions:
        p_instrs.append(op.Serialized([i] * serial_factor))
    p = op.Parallelized(p_instrs)
    init_values = [op.init_value_by_llvm_type[reg.llvm_type] for reg in p.get_source_registers()]
    b = IntegerLoopBenchmark(p, init_values)
    result = b.build_and_execute(repeat=4, min_elapsed=0.1, max_elapsed=0.2)
    lat = min(*[(t / serial_factor) * result['frequency'] / result['iterations']
                for t in result['runtimes']])

    # Throughput Benchmark
    p_instrs = []
    for i in instructions:
        p_instrs.append(op.Serialized([i] * parallel_serial_factor))
    p = op.Parallelized(p_instrs * parallel_factor)
    init_values = [op.init_value_by_llvm_type[reg.llvm_type] for reg in
                   p.get_source_registers()]
    b = IntegerLoopBenchmark(p, init_values)
    result = b.build_and_execute(repeat=4, min_elapsed=0.1, max_elapsed=0.2)
    tp = min(
        [(t / parallel_serial_factor / parallel_factor) * result['frequency'] / result['iterations']
         for t in result['runtimes']])

    # Result compilation
    return lat, tp


if __name__ == '__main__':
    llvm.initialize()
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
    llvm.initialize_native_asmparser()

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

    print(bench_instruction(op.Instruction(
        instruction='add $2, $0',
        destination_operand=op.Register('i64', 'r'),
        source_operands=[op.Register('i64', '0'), op.Immediate('i64', '1')])))

    # if len(s.get_source_operand_types())
    # b = IntegerLoopBenchmark(loop_body,
    #                          [(type_, dst_reg, '1', src_reg)
    #                           # for type_, dst_reg, src_reg in zip(s.get_last_destination_type(), )])
    # print(b.get_ir())