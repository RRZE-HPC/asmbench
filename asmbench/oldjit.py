#!/usr/bin/env python3
import ctypes
import sys
import time
import textwrap
import itertools
import random
import collections
import pprint
import math

import llvmlite.binding as llvm
import psutil


# TODOs
# * API to create test scenarios
#   * DSL?
# * Test cases:
#   * Instructions:
#     * [x] arithmetics \w reg and/or imm.
#       * scalar
#       * packed
#     * [x] lea
#     * [x] LOAD / mov \w mem
#     * [TODO] STORE / mov to mem
#   * [x] Single Latency
#   * [x] Single Throughput
#   * [TODO] Combined Throughput
#   * [TODO] Random Throughput
# * [TODO] Automated TP, Lat, #pipeline analysis
# * [TODO] IACA marked binary output generation
# * [TODO] Fuzzing algorithm
# * [TODO] CLI
# * C based timing routine? As an extension?
# * make sanity checks during runtime, check for fixed frequency and pinning

def floor_harmonic_fraction(n, error=0.1):
    """
    Finds closest floored integer or inverse integer and returns error.

    (numerator, denominator, relative error) where either numerator or denominator is exactly one.
    """
    floor_n = math.floor(n)
    if floor_n > 0:
        return floor_n, 1, 1 - floor_n / n
    else:
        i = 2
        while (1 / i) > n:
            i += 1

        return 1, i, 1 - (1 / i) / n


class Benchmark:
    def __init__(self, parallel=1, serial=5):
        self._function_ctype = ctypes.CFUNCTYPE(ctypes.c_int64, ctypes.c_int64)
        self.parallel = parallel
        self.serial = serial

        # Do interesting work
        self._loop_body = textwrap.dedent('''\
            %"checksum" = phi i64 [0, %"entry"], [%"checksum.1", %"loop"]
            %"checksum.1" = call i64 asm sideeffect "
                add $1, $0",
                "=r,i,r" (i64 1, i64 %"checksum")\
            ''')

    def __repr__(self):
        return '{}({})'.format(
            self.__class__.__name__,
            ', '.join(['{}={!r}'.format(k, v) for k, v in self.__dict__.items()
                       if not k.startswith('_')]))

    def get_ir(self):
        # FP add loop - may have issues
        # return textwrap.dedent('''\
        #    define i64 @"test"(i64 %"N")
        #    {{
        #    entry:
        #      %"N.fp" = sitofp i64 %"N" to double
        #      %"loop_cond" = fcmp olt double 0.0, %"N.fp"
        #      br i1 %"loop_cond", label %"loop", label %"end"
        #
        #    loop:
        #      %"loop_counter" = phi double [0.0, %"entry"], [%"loop_counter.1", %"loop"]
        #    {loop_body}
        #      %"loop_counter.1" = fadd double %"loop_counter", 1.0
        #      %"loop_cond.1" = fcmp olt double %"loop_counter.1", %"N.fp"
        #      br i1 %"loop_cond.1", label %"loop", label %"end"
        #
        #    end:
        #      %"ret.fp" = phi double [0.0, %"entry"], [%"loop_counter", %"loop"]
        #      %"ret" = fptosi double %"ret.fp" to i64
        #      ret i64 %"ret"
        #    }}
        #    ''').format(
        #        loop_body=textwrap.indent(self._loop_body, '  '))
        return textwrap.dedent('''\
            define i64 @"test"(i64 %"N")
            {{
            entry:
              %"loop_cond" = icmp slt i64 0, %"N"
              br i1 %"loop_cond", label %"loop", label %"end"

            loop:
              %"loop_counter" = phi i64 [0, %"entry"], [%"loop_counter.1", %"loop"]
            {loop_body}
              %"loop_counter.1" = add i64 %"loop_counter", 1
              %"loop_cond.1" = icmp slt i64 %"loop_counter.1", %"N"
              br i1 %"loop_cond.1", label %"loop", label %"end"

            end:
              %"ret" = phi i64 [0, %"entry"], [%"loop_counter", %"loop"]
              ret i64 %"ret"
            }}
            ''').format(
            loop_body=textwrap.indent(self._loop_body, '  '))

    def prepare_arguments(self, previous_args=None, time_factor=1.0):
        """Build argument tuple, to be passed to low level function."""
        if previous_args is None:
            return 100,
        else:
            return int(previous_args[0] * time_factor),

    def get_iterations(self, args):
        """Return number of iterations performed, based on lower level function arguments."""
        return args[0]

    def get_llvm_module(self):
        """Build and return LLVM module from LLVM IR code."""
        if not hasattr(self, '_llvm_module'):
            self._llvm_module = llvm.parse_assembly(self.get_ir())
            self._llvm_module.verify()
        return self._llvm_module

    def get_target_machine(self):
        """Instantiate and return target machine."""
        if not hasattr(self, '_llvm_module'):
            features = llvm.get_host_cpu_features().flatten()
            cpu = llvm.get_host_cpu_name()
            self._tm = llvm.Target.from_default_triple().create_target_machine(
                cpu=cpu, features=features, opt=1)
        return self._tm

    def get_assembly(self):
        """Compile and return assembly from LLVM module."""
        tm = self.get_target_machine()
        tm.set_asm_verbosity(0)
        return tm.emit_assembly(self.get_llvm_module())

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
            cfunc = self._function_ctype(cfptr)

            # Now 'cfunc' is an actual callable we can invoke
            # TODO replace time.clock with a C implemententation for less overhead
            # TODO return result in machine readable format
            fixed_args = False
            for i in range(repeat):
                while True:
                    start = time.perf_counter()
                    res = cfunc(*args)
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
                'frequency': psutil.cpu_freq().current * 1e6}

    @classmethod
    def get_latency(cls, max_serial=6, print_table=False, **kwargs):
        if print_table:
            print(' s |' + ''.join([' {:^5}'.format(i) for i in range(1, max_serial)]))
            print('   | ', end='')
        serial_runs = []
        for s in range(1, max_serial):
            m = cls(serial=s, parallel=1, **kwargs)
            r = m.build_and_execute(repeat=1)
            cy_per_it = min(r['runtimes']) * r['frequency'] / (
                        r['iterations'] * m.parallel * m.serial)
            if print_table:
                print('{:.3f} '.format(cy_per_it), end='')
            sys.stdout.flush()

            serial_runs.append((cy_per_it, floor_harmonic_fraction(cy_per_it), m))

        if print_table:
            print()
            print('LAT: {lat[0]}/{lat[1]}cy (min. error {lat[2]:.1%})'.format(
                lat=min(serial_runs)[1]))

        return min(serial_runs)[1]

    @classmethod
    def get_throughput(cls, max_serial=6, max_parallel=17, print_table=False, **kwargs):
        if print_table:
            print('s\p |' + ''.join([' {:^5}'.format(i) for i in range(2, max_parallel)]))
        parallel_runs = []
        for s in range(1, max_serial):
            if print_table:
                print('{:>3} | '.format(s), end='')
            for p in range(2, max_parallel):
                m = cls(serial=s, parallel=p, **kwargs)
                r = m.build_and_execute(repeat=1)
                cy_per_it = min(r['runtimes']) * r['frequency'] / (
                            r['iterations'] * m.parallel * m.serial)
                if print_table:
                    print('{:.3f} '.format(cy_per_it), end='')
                sys.stdout.flush()
                parallel_runs.append((cy_per_it, floor_harmonic_fraction(cy_per_it), m))
            if print_table:
                print()

        if print_table:
            print('TP: {tp[0]}/{tp[1]}cy (min. error {tp[2]:.1%});'.format(
                tp=min(parallel_runs)[1]))

        return min(parallel_runs)[1]


class InstructionBenchmark(Benchmark):
    def __init__(self, instruction='addq $1, $0',
                 dst_operands=(),
                 dstsrc_operands=(('r', 'i64', '0'),),
                 src_operands=(('i', 'i64', '1'),),
                 parallel=10,
                 serial=4):
        """
        Build LLVM IR for arithmetic instruction benchmark without memory references.

        Currently only one destination (dst) or combined destination and source (dstsrc) operand
        is allowed. Only instruction's operands ($N) refer to the order of opernads found in
        dst + dstsrc + src.
        """
        Benchmark.__init__(self, parallel=parallel, serial=serial)
        self.instruction = instruction
        self.dst_operands = dst_operands
        self.dstsrc_operands = dstsrc_operands
        self.src_operands = src_operands
        self._loop_body = ''
        if len(dst_operands) + len(dstsrc_operands) != 1:
            raise NotImplemented("Must have exactly one dst or dstsrc operand.")
        if not all([op[0] in 'irx'
                    for op in itertools.chain(dst_operands, dstsrc_operands, src_operands)]):
            raise NotImplemented("This class only supports register and immediate operands.")

        # Part 1: PHI functions and initializations
        for i, dstsrc_op in enumerate(dstsrc_operands):
            # constraint code, llvm type string, initial value
            if dstsrc_op[0] in 'rx':
                # register operand
                for p in range(self.parallel):
                    self._loop_body += (
                        '%"dstsrc{index}_{p}" = phi {type} '
                        '[{initial}, %"entry"], [%"dstsrc{index}_{p}.out", %"loop"]\n').format(
                        index=i, type=dstsrc_op[1], initial=dstsrc_op[2], p=p)
            else:
                raise NotImplemented("Operand type in {!r} is not yet supported.".format(dstsrc_op))

        # Part 2: Inline ASM call
        # Build constraint string from operands
        constraints = ','.join(
            ['=' + dop[0] for dop in itertools.chain(dst_operands, dstsrc_operands)] +
            [sop[0] for sop in itertools.chain(src_operands)] +
            ['{}'.format(i + len(dst_operands)) for i in range(len(dstsrc_operands))])

        for i, dstsrc_op in enumerate(dstsrc_operands):
            # Build instruction from instruction and operands
            # TODO support multiple dstsrc operands
            # TODO support dst and dstsrc operands at the same time
            for p in range(self.parallel):
                operands = ['{type} {val}'.format(type=sop[1], val=sop[2]) for sop in src_operands]
                for j, dop in enumerate(dstsrc_operands):
                    operands.append('{type} %dstsrc{index}_{p}'.format(type=dop[1], index=j, p=p))
                args = ', '.join(operands)

                self._loop_body += (
                    '%"dstsrc{index}_{p}.out" = call {dst_type} asm sideeffect'
                    ' "{instruction}", "{constraints}" ({args})\n').format(
                    index=i,
                    dst_type=dstsrc_op[1],
                    instruction='\n'.join([instruction] * self.serial),
                    constraints=constraints,
                    args=args,
                    p=p)

        for i, dst_op in enumerate(dst_operands):
            # Build instruction from instruction and operands
            # TODO support multiple dst operands
            # TODO support dst and dstsrc operands at the same time
            if self.serial != 1:
                raise NotImplemented("Serial > 1 and dst operand is not supported.")
            for p in range(self.parallel):
                operands = ['{type} {val}'.format(type=sop[1], val=sop[2]) for sop in src_operands]
                args = ', '.join(operands)

                self._loop_body += (
                    '%"dst{index}_{p}.out" = call {dst_type} asm sideeffect'
                    ' "{instruction}", "{constraints}" ({args})\n').format(
                    index=i,
                    dst_type=dst_op[1],
                    instruction=instruction,
                    constraints=constraints,
                    args=args,
                    p=p)


class AddressGenerationBenchmark(Benchmark):
    def __init__(self,
                 offset=('i', 'i64', '0x42'),
                 base=('r', 'i64', '0'),
                 index=('r', 'i64', '0'),
                 width=('i', None, '4'),
                 destination='base',
                 parallel=10,
                 serial=4):
        """
        Benchmark for address generation modes.

        Arguments may be None or (arg_type, reg_type, initial_value), with arg_type 'r' (register)
        or 'i' (immediate) and initial_value a string.
        E.g., ('r', 'i64', '0') or ('i', None, '4')

        +--------------------------------+-----------------------------+
        | Mode                           | AT&T                        |
        +--------------------------------+-----------------------------+
        | Offset                         | leal           0x0100, %eax | <- no latency support
        | Base                           | leal           (%esi), %eax |
        | Offset + Base                  | leal         -8(%ebp), %eax |
        | Offset + Index*Width           | leal   0x100(,%ebx,4), %eax |
        | Offset + Base + Index*Width    | leal 0x8(%edx,%ebx,4), %eax |
        +--------------------------------+-----------------------------+
        OFFSET(BASE, INDEX, WIDTH) -> offset + base + index*width
        offset: immediate integer (+/-)
        base: register
        index: register
        width: immediate 1,2,4 or 8
        """
        Benchmark.__init__(self, parallel=parallel, serial=serial)
        self.offset = offset
        self.base = base
        self.index = index
        self.width = width
        self.destination = destination
        self.parallel = parallel
        # Sanity checks:
        if bool(index) ^ bool(width):
            raise ValueError("Index and width both need to be set, or be None.")
        elif index and width:
            if width[0] != 'i' or int(width[2]) not in [1, 2, 4, 8]:
                raise ValueError("Width may only be immediate 1,2,4 or 8.")
            if index[0] != 'r':
                raise ValueError("Index must be a register.")

        if offset and offset[0] != 'i':
            raise ValueError("Offset must be an immediate.")
        if base and base[0] != 'r':
            raise ValueError("Offset must be a register.")

        if not index and not width and not offset and not base:
            raise ValueError("Must provide at least an offset or base.")

        if destination == 'base' and not base:
            raise ValueError("Destination may only be set to 'base' if base is set.")
        elif destination == 'index' and not index:
            raise ValueError("Destination may only be set to 'index' if index is set.")
        elif destination not in ['base', 'index']:
            raise ValueError("Destination must be set to 'base' or 'index'.")

        if not base and not index:
            raise ValueError("Either base or index must be set for latency test to work.")

        if serial != 1 and not (base or index):
            raise ValueError("Serial > 1 only works with index and/or base in use.")

        self._loop_body = ''

        ops = ''
        if offset:
            ops += offset[2]
        if base:
            ops += '($0'
            if width and index:
                ops += ',$1,{}'.format(width[2])
            ops += ')'

            if destination == 'base':
                ops += ', $0'
            else:  # destination == 'index'
                ops += ', $1'
        else:
            if width and index:
                ops += '(,$0,{}), $0'.format(width[2])
        ops += ' '

        if destination == 'base':
            destination_reg = base
        else:  # destination == 'index'
            destination_reg = index

        # Part 1: PHI function for destination
        for p in range(parallel):
            self._loop_body += (
                '%"{name}_{p}.0" = '
                'phi {type} [{initial}, %"entry"], [%"{name}_{p}.{s}", %"loop"]\n').format(
                name=destination, type=destination_reg[1], initial=destination_reg[2], p=p,
                s=self.serial)

        for p in range(parallel):
            for s in range(self.serial):
                constraints = '=r,r'
                if base and index:
                    constraints += ',r'
                    if destination == 'base':
                        args = '{base_type} %"{base_name}_{p}.{s_in}", {index_type} {index_value}'.format(
                            base_type=base[1], base_name=destination,
                            index_type=index[1], index_value=index[2], p=p, s_in=s)
                    else:  # destination == 'index':
                        args = '{base_type} {base_value}, {index_type} %"{index_name}_{p}.{s_in}"'.format(
                            base_type=base[1], base_value=base[2],
                            index_type=index[1], index_name=destination, p=p, s_in=s)
                else:
                    args = '{type} %"{name}_{p}.{s_in}"'.format(
                        type=destination_reg[1], name=destination, p=p, s_in=s)

                self._loop_body += (
                    '%"{name}_{p}.{s_out}" = call {type} asm sideeffect'
                    ' "lea {ops}", "{constraints}" ({args})\n').format(
                    name=destination,
                    type=destination_reg[1],
                    ops=ops,
                    constraints=constraints,
                    args=args,
                    p=p,
                    s_out=s + 1)


class LoadBenchmark(Benchmark):
    def __init__(self, chain_length=2048, structure='linear', parallel=6, serial=4):
        """
        Benchmark for L1 load using pointer chasing.

        *chain_length* is the number of pointers to place in memory.
        *structure* may be 'linear' (1-offsets) or 'random'.
        """
        Benchmark.__init__(self, parallel=parallel, serial=serial)
        self._loop_body = ''
        element_type = ctypes.POINTER(ctypes.c_int)
        self._function_ctype = ctypes.CFUNCTYPE(
            ctypes.c_int, ctypes.POINTER(element_type), ctypes.c_int)
        self.chain_length = chain_length
        self.parallel = parallel
        self.structure = structure
        self._pointer_field = (element_type * chain_length)()
        if chain_length % serial != 0:
            raise ValueError(
                "chain_length ({}) needs to be divisible by serial factor ({}).".format(
                    chain_length, serial))

        # Initialize pointer field
        # Field must represent a ring of pointers
        if structure == 'linear':
            for i in range(chain_length):
                self._pointer_field[i] = ctypes.cast(
                    ctypes.pointer(self._pointer_field[(i + 1) % chain_length]), element_type)
        elif structure == 'random':
            shuffled_indices = list(range(chain_length))
            random.shuffle(shuffled_indices)
            for i in range(chain_length):
                self._pointer_field[shuffled_indices[i]] = ctypes.cast(
                    ctypes.pointer(self._pointer_field[shuffled_indices[(i + 1) % chain_length]]),
                    element_type)
        else:
            raise ValueError("Given structure is not supported. Supported are: "
                             "linear and random.")

    def prepare_arguments(self, previous_args=None, time_factor=1.0):
        """Build argument tuple, to be passed to low level function."""
        if previous_args is None:
            return self._pointer_field, 100
        else:
            return previous_args[0], int(previous_args[1] * time_factor)

    def get_iterations(self, args):
        """Return number of iterations performed, based on lower level function arguments."""
        return self.chain_length * args[1]

    def get_ir(self):
        """
        Return LLVM IR equivalent of (in case of parallel == 1 and serial == 1):

        int test(int** ptrf, int repeat) {
            int** p0 = (int**)ptrf[0];
            int i = 0;
            while(i < N) {
                int** p = (int**)*p0;
                while(p != p0) {
                    p = (int**)*p;
                }
                i++;
            }
            return i;
        }
        """
        ret = textwrap.dedent('''
        define i32 @test(i32** %"ptrf_0", i32 %"repeats") {
        entry:
        ''')
        # Load pointer to ptrf[p] and p0
        for p in range(self.parallel):
            if p > 0:
                ret += '  %"ptrf_{p}" = getelementptr i32*, i32** %"ptrf_0", i64 {p}\n'.format(p=p)
            ret += (
                '  %"pp0_{p}" = bitcast i32** %"ptrf_{p}" to i32***\n'
                '  %"p0_{p}" = load i32**, i32*** %"pp0_{p}", align 8\n').format(p=p)

        ret += textwrap.dedent('''
            %"cmp.entry" = icmp sgt i32 %"repeats", 0
            br i1 %"cmp.entry", label %"loop0", label %"end"

        loop0:
            br label %"loop1"

        loop1:
            %"i" = phi i32 [ %"i.1", %"loop3" ], [ 0, %"loop0" ]
            br label %"loop2"

        loop2:\n''')

        for p in range(self.parallel):
            ret += ('  %"p_{p}.0" = phi i32** '
                    '[ %"p0_{p}", %"loop1" ], [ %"p_{p}.{s_max}", %"loop2" ]\n').format(
                p=p, s_max=self.serial)

        # load p, compare to p0 and or-combine results
        for p in range(self.parallel):
            for s in range(self.serial):
                ret += ('  %"pp_{p}.{s}" = bitcast i32** %"p_{p}.{s_prev}" to i32***\n'
                        '  %"p_{p}.{s}" = load i32**, i32*** %"pp_{p}.{s}", align 8\n').format(
                    p=p, s=s + 1, s_prev=s)

            # Compare is needed for all registers, for llvm not to remove unused 
            # instructions:
            ret += '  %"cmp_{p}.loop2" = icmp eq i32** %"p_{p}.{s_max}", %"p0_{p}"\n'.format(
                p=p, s_max=self.serial)

        # TODO tree reduce cmp to make use of all cmp_* values

        # It is sufficient to use only one compare, all others will be eliminated
        ret += '  br i1 %"cmp_0.loop2", label %"loop3", label %"loop2"\n'

        ret += textwrap.dedent('''
        loop3:
            %"i.1" = add i32 %"i", 1
            %"cmp.loop3" = icmp eq i32 %"i.1", %"repeats"
            br i1 %"cmp.loop3", label %"end", label %"loop1"

        end:
            %"ret" = phi i32 [ 0, %"entry" ], [ %"repeats", %"loop3" ]
            ret i32 %"ret"
        }''')
        return ret


if __name__ == '__main__':
    llvm.initialize()
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
    llvm.initialize_native_asmparser()

    modules = collections.OrderedDict()

    # immediate source
    modules['add i64 r64 LAT'] = InstructionBenchmark(
        instruction='addq $1, $0',
        dst_operands=(),
        dstsrc_operands=(('r', 'i64', '0'),),
        src_operands=(('i', 'i64', '1'),),
        parallel=1,
        serial=5)

    # register source
    modules['add r64 r64 LAT'] = InstructionBenchmark(
        instruction='addq $1, $0',
        dst_operands=(),
        dstsrc_operands=(('r', 'i64', '0'),),
        src_operands=(('r', 'i64', '1'),),
        parallel=1,
        serial=5)

    # multiple instructions
    modules['4xadd i64 r64 LAT'] = InstructionBenchmark(
        instruction='addq $1, $0\naddq $1, $0\naddq $1, $0\naddq $1, $0',
        dst_operands=(),
        dstsrc_operands=(('r', 'i64', '0'),),
        src_operands=(('i', 'i64', '1'),),
        parallel=1,
        serial=5)

    # immediate source
    modules['add i64 r64 TP'] = InstructionBenchmark(
        instruction='addq $1, $0',
        dst_operands=(),
        dstsrc_operands=(('r', 'i64', '0'),),
        src_operands=(('i', 'i64', '1'),),
        parallel=10,
        serial=5)

    # register source
    modules['add r64 r64 TP'] = InstructionBenchmark(
        instruction='addq $1, $0',
        dst_operands=(),
        dstsrc_operands=(('r', 'i64', '0'),),
        src_operands=(('r', 'i64', '1'),),
        parallel=10,
        serial=5)

    # multiple instructions
    modules['4xadd i64 r64 TP'] = InstructionBenchmark(
        instruction='addq $1, $0\naddq $1, $0\naddq $1, $0\naddq $1, $0',
        dst_operands=(),
        dstsrc_operands=(('r', 'i64', '0'),),
        src_operands=(('i', 'i64', '1'),),
        parallel=10,
        serial=1)

    modules['lea base LAT'] = AddressGenerationBenchmark(
        offset=None,
        base=('r', 'i64', '666'),
        index=None,
        width=None,
        destination='base',
        parallel=1,
        serial=5)

    modules['lea base+offset LAT'] = AddressGenerationBenchmark(
        offset=('i', None, '23'),
        base=('r', 'i64', '666'),
        index=None,
        width=None,
        destination='base',
        parallel=1,
        serial=5)

    modules['lea index*width LAT'] = AddressGenerationBenchmark(
        offset=None,
        base=None,
        index=('r', 'i64', '1'),
        width=('i', None, '4'),
        destination='index',
        parallel=1,
        serial=5)

    modules['lea offset+index*width LAT'] = AddressGenerationBenchmark(
        offset=('i', 'i64', '-0x8'),
        base=None,
        index=('r', 'i64', '51'),
        width=('i', None, '4'),
        destination='index',
        parallel=1,
        serial=5)

    modules['lea base+index*width LAT'] = AddressGenerationBenchmark(
        offset=None,
        base=('r', 'i64', '23'),
        index=('r', 'i64', '12'),
        width=('i', None, '4'),
        destination='base',
        parallel=1,
        serial=5)

    modules['lea base+offset+index*width LAT'] = AddressGenerationBenchmark(
        offset=('i', None, '42'),
        base=('r', 'i64', '23'),
        index=('r', 'i64', '12'),
        width=('i', None, '4'),
        destination='base',
        parallel=1,
        serial=5)

    modules['lea base TP'] = AddressGenerationBenchmark(
        offset=None,
        base=('r', 'i64', '666'),
        index=None,
        width=None,
        destination='base',
        parallel=10,
        serial=1)

    modules['lea base+offset TP'] = AddressGenerationBenchmark(
        offset=('i', None, '23'),
        base=('r', 'i64', '666'),
        index=None,
        width=None,
        destination='base',
        parallel=10,
        serial=1)

    modules['lea index*width TP'] = AddressGenerationBenchmark(
        offset=None,
        base=None,
        index=('r', 'i64', '1'),
        width=('i', None, '4'),
        destination='index',
        parallel=10,
        serial=1)

    modules['lea offset+index*width TP'] = AddressGenerationBenchmark(
        offset=('i', 'i64', '-0x8'),
        base=None,
        index=('r', 'i64', '51'),
        width=('i', None, '4'),
        destination='index',
        parallel=10,
        serial=1)

    modules['lea base+index*width TP'] = AddressGenerationBenchmark(
        offset=None,
        base=('r', 'i64', '23'),
        index=('r', 'i64', '12'),
        width=('i', None, '4'),
        destination='base',
        parallel=10,
        serial=1)

    modules['lea base+offset+index*width TP'] = AddressGenerationBenchmark(
        offset=('i', None, '42'),
        base=('r', 'i64', '23'),
        index=('r', 'i64', '12'),
        width=('i', None, '4'),
        destination='base',
        parallel=10,
        serial=1)

    modules['LD linear LAT'] = LoadBenchmark(
        chain_length=2048,  # 2048 * 8B = 16kB
        structure='linear',
        parallel=1,
        serial=8)

    modules['LD random LAT'] = LoadBenchmark(
        chain_length=2048,  # 2048 * 8B = 16kB
        structure='random',
        parallel=1,
        serial=8)

    modules['LD linear TP'] = LoadBenchmark(
        chain_length=2048,  # 2048 * 8B = 16kB
        structure='linear',
        parallel=6,
        serial=8)

    modules['LD random TP'] = LoadBenchmark(
        chain_length=2048,  # 2048 * 8B = 16kB
        structure='random',
        parallel=6,
        serial=8)
    # TODO check that this does what it's supposed to do...
    print(modules['LD linear TP'].get_assembly())

    modules['vaddpd x<4 x double> x<4 x double> x<4 x double> LAT'] = InstructionBenchmark(
        instruction='vaddpd $1, $0, $0',
        dst_operands=(),
        dstsrc_operands=(('x', '<4 x double>', '<{}>'.format(', '.join(['double 1.23e-10'] * 4))),),
        src_operands=(('x', '<4 x double>', '<{}>'.format(', '.join(['double 3.21e-10'] * 4))),),
        parallel=1,
        serial=5)

    modules['vmulpd x<4 x double> x<4 x double> x<4 x double> (dstsrc) LAT'] = InstructionBenchmark(
        instruction='vmulpd $1, $0, $0',
        dst_operands=(),
        dstsrc_operands=(('x', '<4 x double>', '<{}>'.format(', '.join(['double 1.23e-10'] * 4))),),
        src_operands=(('x', '<4 x double>', '<{}>'.format(', '.join(['double 3.21e-10'] * 4))),),
        parallel=1,
        serial=5)

    # This is actually a TP benchmark with parallel=1, because there are no inter-loop depencies:
    modules['vmulpd x<4 x double> x<4 x double> x<4 x double> (dstsrc) TP'] = InstructionBenchmark(
        instruction='vmulpd $1, $2, $0',
        dst_operands=(),
        dstsrc_operands=(('x', '<4 x double>', '<{}>'.format(', '.join(['double 1.23e-10'] * 4))),),
        src_operands=(('x', '<4 x double>', '<{}>'.format(', '.join(['double 3.21e-10'] * 4))),),
        parallel=10,
        serial=1)

    modules = collections.OrderedDict([(k, v) for k,v in modules.items() if k.startswith('LD ')])

    verbose = 2 if '-v' in sys.argv else 0
    for key, module in modules.items():
        if verbose > 0:
            print("=== Benchmark")
            print(repr(module))
            print("=== LLVM")
            print(module.get_ir())
            print("=== Assembly")
            print(module.get_assembly())
        r = module.build_and_execute(repeat=3)
        if verbose > 0:
            print("=== Result")
            pprint.pprint(r)

        cy_per_it = min(r['runtimes']) * r['frequency'] / (
                    r['iterations'] * module.parallel * module.serial)
        print('{key:<32} {cy_per_it:.3f} cy/It with {runtime_sum:.4f}s'.format(
            key=key,
            module=module,
            cy_per_it=cy_per_it,
            runtime_sum=sum(r['runtimes'])))

    # InstructionBenchmark.get_latency(
    #    instruction='vmulpd $1, $0, $0',
    #    dst_operands=(),
    #    dstsrc_operands=(('x','<4 x double>', '<{}>'.format(', '.join(['double 1.23e-10']*4))),),
    #    src_operands=(('x','<4 x double>', '<{}>'.format(', '.join(['double 3.21e-10']*4))),
    #                  ('x','<4 x double>', '<{}>'.format(', '.join(['double 2.13e-10']*4))),),
    #    print_table=True)
    # InstructionBenchmark.get_throughput(
    #    instruction='vmulpd $1, $0, $0',
    #    dst_operands=(),
    #    dstsrc_operands=(('x','<4 x double>', '<{}>'.format(', '.join(['double 1.23e-10']*4))),),
    #    src_operands=(('x','<4 x double>', '<{}>'.format(', '.join(['double 3.21e-10']*4))),
    #                  ('x','<4 x double>', '<{}>'.format(', '.join(['double 2.13e-10']*4))),),
    #    print_table=True)
    #
    # InstructionBenchmark.get_latency(
    #    instruction='nop',
    #    dst_operands=(),
    #    dstsrc_operands=(('r','i8', '0'),),
    #    src_operands=(),
    #    print_table=True)
    # InstructionBenchmark.get_throughput(
    #    instruction='nop',
    #    dst_operands=(),
    #    dstsrc_operands=(('r','i8', '0'),),
    #    src_operands=(),
    #    print_table=True)
