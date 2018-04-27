#!/usr/bin/env python3
import ctypes
import sys
import time
import textwrap
import itertools
import random
import collections

import llvmlite.binding as llvm
import psutil

# TODOs
# * API to create test scenarios
#   * DSL?
# * Test cases:
#   * Instructions:
#     * arithmetics \w reg and/or imm.
#       * scalar
#       * packed
#     * lea
#     * LOAD / mov \w mem
#   * Single Latency
#   * Single Throughput
#   * Combined Throughput
#   * Random Throughput
# * IACA marked binary output generation
# * Fuzzing algorithm
# * CLI
# * C based timing routine? As an extension?
# * make sanity checks during runtime, check for fixed frequency and pinning
        

class Benchmark:
    LLVM2CTYPE = {
        'i8': ctypes.c_int8,
        'i16': ctypes.c_int16,
        'i32': ctypes.c_int32,
        'i64': ctypes.c_int64,
        'float': ctypes.c_float,
        'double': ctypes.c_double,
        'i8*': ctypes.POINTER(ctypes.c_int8),
        'i16*': ctypes.POINTER(ctypes.c_int16),
        'i32*': ctypes.POINTER(ctypes.c_int32),
        'i64*': ctypes.POINTER(ctypes.c_int64),
        'float*': ctypes.POINTER(ctypes.c_float),
        'double*': ctypes.POINTER(ctypes.c_double),
    }
    def __init__(self):
        self._loop_init = ''
        self._ret_llvmtype = 'i64'
        self._ret_ctype = self.LLVM2CTYPE[self._ret_llvmtype]
        self._function_ctype = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int)
        self._iterations = 100000000
        
        # Do interesting work
        self._loop_body = textwrap.dedent('''\
            %"checksum" = phi i64 [0, %"entry"], [%"checksum.1", %"loop"]
            %"checksum.1" = call i64 asm sideeffect "
                add $1, $0",
                "=r,i,r" (i64 1, i64 %"checksum")\
            ''')
        
        # Set %"ret" to something, needs to be a constant or phi function
        self._loop_tail = textwrap.dedent('''\
            %"ret" = phi i64 [0, %"entry"], [%"checksum.1", %"loop"]\
            ''')
    
    def __repr__(self):
        return '{}({})'.format(
            self.__class__.__name__,
            ', '.join(['{}={!r}'.format(k,v) for k,v in self.__dict__.items()
                       if not k.startswith('_')]))
    
    def get_ir(self):
        return textwrap.dedent('''\
            define {ret_type} @"test"(i64 %"N")
            {{
            entry:
              %"loop_cond" = icmp slt i64 0, %"N"
            {loop_init}
              br i1 %"loop_cond", label %"loop", label %"end"

            loop:
              %"loop_counter" = phi i64 [0, %"entry"], [%"loop_counter.1", %"loop"]
            {loop_body}
              %"loop_counter.1" = add i64 %"loop_counter", 1
              %"loop_cond.1" = icmp slt i64 %"loop_counter.1", %"N"
              br i1 %"loop_cond.1", label %"loop", label %"end"

            end:
            {loop_tail}
              ret {ret_type} %"ret"
            }}
            ''').format(
                ret_type=self._ret_llvmtype,
                loop_init=textwrap.indent(self._loop_init, '  '),
                loop_body=textwrap.indent(self._loop_body, '  '),
                loop_tail=textwrap.indent(self._loop_tail, '  '))
    
    def prepare_arguments(self):
        '''Build argument tuple, to be passed to low level function.'''
        return (self._iterations,)
    
    def get_llvm_module(self):
        '''Build and return LLVM module from LLVM IR code.'''
        if not hasattr(self, '_llvm_module'):
            self._llvm_module = llvm.parse_assembly(self.get_ir())
            self._llvm_module.verify()
        return self._llvm_module
    
    def get_target_machine(self):
        '''Instantiate and return target machine.'''
        if not hasattr(self, '_llvm_module'):
            features=llvm.get_host_cpu_features().flatten()
            cpu=llvm.get_host_cpu_name()
            self._tm = llvm.Target.from_default_triple().create_target_machine(
                cpu=cpu, features=features)
        return self._tm
    
    def get_assembly(self):
        '''Compile and return assembly from LLVM module.'''
        tm = self.get_target_machine()
        tm.set_asm_verbosity(0)
        return tm.emit_assembly(self.get_llvm_module())
    
    def build_and_execute(self, repeat=10):
        # Compile the module to machine code using MCJIT
        tm = self.get_target_machine()
        runtimes = []
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
            args = self.prepare_arguments()
            for i in range(repeat):
                start = time.perf_counter()
                res = cfunc(*args)
                end = time.perf_counter()
                runtimes.append(end-start)
        
        return {'iterations': self._iterations,
                'runtimes': runtimes,
                'frequency': psutil.cpu_freq().current*1e6}


class InstructionBenchmark(Benchmark):
    def __init__(self, instruction='addq $1, $0',
                 dst_operands=(),
                 dstsrc_operands=(('r','i64', '0'),),
                 src_operands=(('i','i64', '1'),),
                 parallelism=10):
        '''
        Build LLVM IR for arithmetic instruction benchmark without memory references.
                 
        Currently only one destination (dst) or combined destination and source (dstsrc) operand
        is allowed. Only instruction's operands ($N) refer to the order of opernads found in
        dst + dstsrc + src.
        '''
        Benchmark.__init__(self)
        self.instruction = instruction
        self.dst_operands = dst_operands
        self.dstsrc_operands = dstsrc_operands
        self.src_operands = src_operands
        self.parallelism = parallelism
        self._loop_init = ''
        self._loop_body = ''
        if len(dst_operands) + len(dstsrc_operands) != 1:
            raise NotImplemented("Must have exactly one dst or dstsrc operand.")
        if not all([op[0] in 'irx'
                    for op in itertools.chain(dst_operands, dstsrc_operands, src_operands)]):
            raise NotImplemented("This class only supports register and immediate operands.")

        self._ret_llvmtype = dst_operands[0][1] if dst_operands else dstsrc_operands[0][1]
        
        # Part 1: PHI functions and initializations
        for i, dstsrc_op in enumerate(itertools.chain(dstsrc_operands)):
            # constraint code, llvm type string, initial value
            if dstsrc_op[0] in 'rx':
                # register operand
                for p in range(self.parallelism):
                    self._loop_body += (
                        '%"dstsrc{index}_{p}" = phi {type} '
                        '[{initial}, %"entry"], [%"dstsrc{index}_{p}.out", %"loop"]\n').format(
                            index=i, type=dstsrc_op[1], initial=dstsrc_op[2], p=p)
            else:
                raise NotImplemented("Operand type in {!r} is not yet supported.".format(dstsrc_op))
        
        for i, dst_op in enumerate(itertools.chain(dst_operands)):
            # No phi functions necessary
            # TODO build phi function to switch between source and destination from one iteration 
            # to next
            raise NotImplemented("Destination operand is not yet implemented")
        
        # Part 2: Inline ASM call
        for i, dstsrc_op in enumerate(itertools.chain(dstsrc_operands)):
            # Build instruction from instruction and operands
            # TODO support multiple dstsrc operands
            # TODO support dst and dstsrc operands at the same time
            # Build constraint string from operands
            constraints = ','.join(
                ['='+dop[0] for dop in itertools.chain(dst_operands, dstsrc_operands)] +
                [sop[0] for sop in itertools.chain(src_operands, dstsrc_operands)])
            
            for p in range(self.parallelism):
                operands = ['{type} {val}'.format(type=sop[1], val=sop[2]) for sop in src_operands]
                for i, dop in enumerate(dstsrc_operands):
                    operands.append('{type} %dstsrc{index}_{p}'.format(type=dop[1], index=i, p=p))
                args = ', '.join(operands)
                
                self._loop_body += (
                    '%"dstsrc{index}_{p}.out" = call {dst_type} asm sideeffect'
                    ' "{instruction}", "{constraints}" ({args})\n').format(
                        index=i,
                        dst_type=dstsrc_op[1],
                        instruction=instruction,
                        constraints=constraints,
                        args=args,
                        p=p)
            
        for i, dst_op in enumerate(dst_operands):
            # FIXME support dst operands
            # TODO support dst and dstsrc operands at the same time
            raise NotImplemented("Destination operand is not yet implemented")
        
        # Set %"ret" to something, needs to be a constant or phi function
        self._loop_tail = textwrap.dedent('''\
            %"ret" = phi {type} [{}, %"entry"], [%"dstsrc0_0.out", %"loop"]\
            '''.format(dstsrc_operands[0][2], type=dstsrc_operands[0][1]))


class AddressGenerationBenchmark(Benchmark):
    def __init__(self,
                 offset=('i', 'i64', '0x42'),
                 base=('r', 'i64', '0'),
                 index=('r', 'i64', '0'),
                 width=('i', None, '4'),
                 destination='base',
                 parallelism=10):
        '''
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
        '''
        Benchmark.__init__(self)
        self.offset = offset
        self.base = base
        self.index = index
        self.width = width
        self.destination = destination
        self.parallelism = parallelism
        # Sanity checks:
        if bool(index) ^ bool(width):
            raise ValueError("Index and width both need to be set, or be None.")
        elif index and width:
            if width[0] != 'i' or int(width[2]) not in [1,2,4,8]:
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
        
        self._loop_init = ''
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
            self._ret_llvmtype = base[1]
            destination_reg = base
        else:  # destination == 'index'
            self._ret_llvmtype = index[1]
            destination_reg = index
        
        # Part 1: PHI function for destination
        for p in range(parallelism):
            self._loop_body += (
                '%"{name}_{p}" = '
                'phi {type} [{initial}, %"entry"], [%"{name}_{p}.out", %"loop"]\n').format(
                    name=destination, type=destination_reg[1], initial=destination_reg[2], p=p)
        

        for p in range(parallelism):
            constraints = '=r,r'
            if base and index:
                constraints += ',r'
                if destination == 'index':
                    args = '{base_type} %"{base_name}_{p}", {index_type} {index_value}'.format(
                        base_type=base[1], base_name=destination,
                        index_type=index[1], index_value=index[2], p=p)
                else:  # destination == 'index':
                    args ='{base_type} {base_value}, {index_type} %"{index_name}_{p}"'.format(
                        base_type=base[1], base_value=base[2],
                        index_type=index[1], index_name=destination, p=p)
            else:
                args = '{type} %"{name}_{p}"'.format(type=destination_reg[1], name=destination, p=p)
            
            self._loop_body += (
                '%"{name}_{p}.out" = call {type} asm sideeffect'
                ' "lea {ops}", "{constraints}" ({args})\n').format(
                    name=destination,
                    type=destination_reg[1],
                    ops=ops,
                    constraints=constraints,
                    args=args,
                    p=p)

        # Set %"ret" to something, needs to be a constant or phi function
        self._loop_tail = textwrap.dedent('''\
            %"ret" = phi {type} [{initial_value}, %"entry"], [%"{name}_0.out", %"loop"]\
            '''.format(name=destination, initial_value=destination_reg[2], type=destination_reg[1]))


class LoadBenchmark(Benchmark):
    def __init__(self, chain_length=2048, repeat=100000, structure='linear', parallelism=6):
        '''
        Benchmark for L1 load using pointer chasing.
        
        *chain_length* is the number of pointers to place in memory.
        *repeat* is the number of iterations the chain run through.
        *structure* may be 'linear' (1-offsets) or 'random'.
        '''
        Benchmark.__init__(self)
        self._loop_init = ''
        self._loop_body = ''
        self._loop_tail = ''
        element_type = ctypes.POINTER(ctypes.c_int)
        self._function_ctype = ctypes.CFUNCTYPE(
            ctypes.c_int, ctypes.POINTER(element_type), ctypes.c_int)
        self.chain_length = chain_length
        self.repeat = repeat
        self._iterations = chain_length*repeat
        self.parallelism = parallelism
        self.structure = structure
        self._pointer_field = (element_type * chain_length)()

        # Initialize pointer field
        # Field must represent a ring of pointers
        if structure == 'linear':
            for i in range(chain_length):
                self._pointer_field[i] = ctypes.cast(
                    ctypes.pointer(self._pointer_field[(i+1)%chain_length]), element_type)
        elif structure == 'random':
            shuffled_indices = list(range(chain_length))
            random.shuffle(shuffled_indices)
            for i in range(chain_length):
                self._pointer_field[shuffled_indices[i]] = ctypes.cast(
                    ctypes.pointer(self._pointer_field[shuffled_indices[(i+1)%chain_length]]),
                    element_type)
        else:
            raise ValueError("Given structure is not supported. Supported are: "
                             "linear and random.")
    
    def prepare_arguments(self):
        return (self._pointer_field, self.repeat)
    
    def get_ir(self):
        '''
        Return LLVM IR equivalent of (in case of parallelism == 1):
        
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
        '''
        ret = textwrap.dedent('''
        define i32 @test(i32** %"ptrf_0", i32 %"repeats") {
        entry:
        ''')
        # Load pointer to ptrf[p] and p0
        for p in range(self.parallelism):
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
        
        for p in range(self.parallelism):
            ret += ('  %"p_{p}" = phi i32** '
                    '[ %"p0_{p}", %"loop1" ], [ %"p_{p}.1", %"loop2" ]\n').format(p=p)
        
        # load p, compare to p0 and or-combine results
        for p in range(self.parallelism):
            ret += ('  %"pp_{p}" = bitcast i32** %"p_{p}" to i32***\n'
                    '  %"p_{p}.1" = load i32**, i32*** %"pp_{p}", align 8\n'
                    '  %"cmp_{p}.loop2" = icmp eq i32** %"p_{p}.1", %"p0_{p}"\n').format(p=p)
            if p == 1:
                ret += ('  %"cmp__{p}.loop2" = '
                        'or i1 %"cmp_{p_before}.loop2", %"cmp_{p}.loop2"\n').format(
                        p=p, p_before=p-1)
            elif p > 1:
                ret += ('  %"cmp__{p}.loop2" = '
                        'or i1 %"cmp__{p_before}.loop2", %"cmp_{p}.loop2"\n').format(
                        p=p, p_before=p-1)
        
        if self.parallelism == 1:
            ret += '  br i1 %"cmp_0.loop2", label %"loop3", label %"loop2"\n'
        else:
            ret += '  br i1 %"cmp__{p_last}.loop2", label %"loop3", label %"loop2"\n'.format(
                p_last=self.parallelism-1)
        
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
        dstsrc_operands=(('r','i64', '0'),),
        src_operands=(('i','i64', '1'),),
        parallelism=1)

    # register source
    modules['add r64 r64 LAT'] = InstructionBenchmark(
        instruction='addq $1, $0',
        dst_operands=(),
        dstsrc_operands=(('r','i64', '0'),),
        src_operands=(('r','i64', '1'),),
        parallelism=1)
    
    # multiple instructions
    modules['4xadd i64 r64 LAT'] = InstructionBenchmark(
        instruction='addq $1, $0\naddq $1, $0\naddq $1, $0\naddq $1, $0',
        dst_operands=(),
        dstsrc_operands=(('r','i64', '0'),),
        src_operands=(('i','i64', '1'),),
        parallelism=1)
    
    # vector add
    # TODO
    
    # immediate source
    modules['add i64 r64 TP'] = InstructionBenchmark(
        instruction='addq $1, $0',
        dst_operands=(),
        dstsrc_operands=(('r','i64', '0'),),
        src_operands=(('i','i64', '1'),),
        parallelism=10)

    # register source
    modules['add r64 r64 TP'] = InstructionBenchmark(
        instruction='addq $1, $0',
        dst_operands=(),
        dstsrc_operands=(('r','i64', '0'),),
        src_operands=(('r','i64', '1'),),
        parallelism=10)
    
    # multiple instructions
    modules['4xadd i64 r64 TP'] = InstructionBenchmark(
        instruction='addq $1, $0\naddq $1, $0\naddq $1, $0\naddq $1, $0',
        dst_operands=(),
        dstsrc_operands=(('r','i64', '0'),),
        src_operands=(('i','i64', '1'),),
        parallelism=10)
    
    modules['lea base LAT'] = AddressGenerationBenchmark(
        offset=None,
        base=('r', 'i64', '666'),
        index=None,
        width=None,
        destination='base',
        parallelism=1)
    
    modules['lea index*width LAT'] = AddressGenerationBenchmark(
        offset=None,
        base=None,
        index=('r', 'i64', '1'),
        width=('i', None, '4'),
        destination='index',
        parallelism=1)
    
    modules['lea offset+index*width LAT'] = AddressGenerationBenchmark(
        offset=('i', 'i64', '-0x8'),
        base=None,
        index=('r', 'i64', '51'),
        width=('i', None, '4'),
        destination='index',
        parallelism=1)
    
    modules['lea base+index*width LAT'] = AddressGenerationBenchmark(
        offset=None,
        base=('r', 'i64', '23'),
        index=('r', 'i64', '12'),
        width=('i', None, '4'),
        destination='base',
        parallelism=1)
    
    modules['lea base+offset+index*width LAT'] = AddressGenerationBenchmark(
        offset=('i', None, '42'),
        base=('r', 'i64', '23'),
        index=('r', 'i64', '12'),
        width=('i', None, '4'),
        destination='base',
        parallelism=1)
    
    modules['lea base TP'] = AddressGenerationBenchmark(
        offset=None,
        base=('r', 'i64', '666'),
        index=None,
        width=None,
        destination='base',
        parallelism=10)
    
    modules['lea index*width TP'] = AddressGenerationBenchmark(
        offset=None,
        base=None,
        index=('r', 'i64', '1'),
        width=('i', None, '4'),
        destination='index',
        parallelism=10)
    
    modules['lea offset+index*width TP'] = AddressGenerationBenchmark(
        offset=('i', 'i64', '-0x8'),
        base=None,
        index=('r', 'i64', '51'),
        width=('i', None, '4'),
        destination='index',
        parallelism=10)
    
    modules['lea base+index*width TP'] = AddressGenerationBenchmark(
        offset=None,
        base=('r', 'i64', '23'),
        index=('r', 'i64', '12'),
        width=('i', None, '4'),
        destination='base',
        parallelism=10)
    
    modules['lea base+offset+index*width TP'] = AddressGenerationBenchmark(
        offset=('i', None, '42'),
        base=('r', 'i64', '23'),
        index=('r', 'i64', '12'),
        width=('i', None, '4'),
        destination='base',
        parallelism=10)
    
    modules['LD linear LAT'] = LoadBenchmark(
        chain_length=2048,  # 2048 * 8B = 16kB
        repeat=100000,
        structure='linear',
        parallelism=1)
        
    modules['LD random LAT'] = LoadBenchmark(
        chain_length=2048,  # 2048 * 8B = 16kB
        repeat=100000,
        structure='random',
        parallelism=1)
                
    modules['LD linear TP'] = LoadBenchmark(
        chain_length=2048,  # 2048 * 8B = 16kB
        repeat=100000,
        structure='linear',
        parallelism=10)
        
    modules['LD random TP'] = LoadBenchmark(
        chain_length=2048,  # 2048 * 8B = 16kB
        repeat=100000,
        structure='random',
        parallelism=10)
        
    modules = {}
    modules['vaddpd x<4 x double> x<4 x double> x<4 x double> LAT'] = InstructionBenchmark(
        instruction='vaddpd $1, $0, $0',
        dst_operands=(),
        dstsrc_operands=(('x','<4 x double>', '<{}>'.format(', '.join(['double 1.23e-10']*4))),),
        src_operands=(('x','<4 x double>', '<{}>'.format(', '.join(['double 3.21e-10']*4))),),
        parallelism=1)
    
    modules['vmulpd x<4 x double> x<4 x double> x<4 x double> LAT'] = InstructionBenchmark(
        instruction='vmulpd $1, $0, $0',
        dst_operands=(),
        dstsrc_operands=(('x','<4 x double>', '<{}>'.format(', '.join(['double 1.23e-10']*4))),),
        src_operands=(('x','<4 x double>', '<{}>'.format(', '.join(['double 3.21e-10']*4))),),
        parallelism=1)
    
    verbose = 0
    for key, module in modules.items():
        if verbose > 0:
            print("=== LLVM")
            print(module.get_ir())
            print("=== Assembly")
            print(module.get_assembly())
        r = module.build_and_execute()
        
        if module.parallelism > 1:
            cy_per_it = min(r['runtimes'])/r['iterations']*r['frequency']/module.parallelism
        else:
            cy_per_it = min(r['runtimes'])/r['iterations']*r['frequency']
        print('{key:<32} {cy_per_it:.2f} cy/It with {runtime_sum:.4f}s'.format(
            key=key,
            module=module,
            cy_per_it=cy_per_it,
            runtime_sum=sum(r['runtimes'])))
    