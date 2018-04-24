#!/usr/bin/env python3
import ctypes
import sys
import time
import textwrap
import itertools
import random

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
        

class InstructionTest:
    LLVM2CTYPE = {
        'i8': ctypes.c_int8,
        'i16': ctypes.c_int16,
        'i32': ctypes.c_int32,
        'i64': ctypes.c_int64,
        'f32': ctypes.c_float,
        'f64': ctypes.c_double,
        'i8*': ctypes.POINTER(ctypes.c_int8),
        'i16*': ctypes.POINTER(ctypes.c_int16),
        'i32*': ctypes.POINTER(ctypes.c_int32),
        'i64*': ctypes.POINTER(ctypes.c_int64),
        'f32*': ctypes.POINTER(ctypes.c_float),
        'f64*': ctypes.POINTER(ctypes.c_double),
    }
    def __init__(self):
        self.loop_init = ''
        self.ret_llvmtype = 'i64'
        self.ret_ctype = self.LLVM2CTYPE[self.ret_llvmtype]
        self.function_ctype = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int)
        self.N = 100000000
        
        # Do interesting work
        self.loop_body = textwrap.dedent('''\
            %"checksum" = phi i64 [0, %"entry"], [%"checksum.1", %"loop"]
            %"checksum.1" = call i64 asm sideeffect "
                add $1, $0",
                "=r,i,r" (i64 1, i64 %"checksum")\
            ''')
        
        # Set %"ret" to something, needs to be a constant or phi function
        self.loop_tail = textwrap.dedent('''\
            %"ret" = phi i64 [0, %"entry"], [%"checksum.1", %"loop"]\
            ''')
    
    def get_ir(self):
        return textwrap.dedent('''\
            define {ret_type} @"test"(i64 %"N")
            {{
            entry:
              %"loop_cond" = icmp slt i64 0, %"N"
            {loop_init}
              br i1 %"loop_cond", label %"loop", label %"end"

            loop:
              %"loop_counter" = phi {ret_type} [0, %"entry"], [%"loop_counter.1", %"loop"]
            {loop_body}
              %"loop_counter.1" = add i64 %"loop_counter", 1
              %"loop_cond.1" = icmp slt i64 %"loop_counter.1", %"N"
              br i1 %"loop_cond.1", label %"loop", label %"end"

            end:
            {loop_tail}
              ret {ret_type} %"ret"
            }}
            ''').format(
                ret_type=self.ret_llvmtype,
                loop_init=textwrap.indent(self.loop_init, '  '),
                loop_body=textwrap.indent(self.loop_body, '  '),
                loop_tail=textwrap.indent(self.loop_tail, '  '))
    
    def prepare_arguments(self):
        '''Build argument tuple, to be passed to low level function.'''
        return (self.N,)
    
    def build_and_execute(self, print_assembly=True):
        llvm_module = llvm.parse_assembly(self.get_ir())
        llvm_module.verify()
    
        # Compile the module to machine code using MCJIT
        tm = llvm.Target.from_default_triple().create_target_machine()
        tm.set_asm_verbosity(100)
        with llvm.create_mcjit_compiler(llvm_module, tm) as ee:
            ee.finalize_object()
            print('=== Assembly')
            print(tm.emit_assembly(llvm_module))
        
            # Obtain a pointer to the compiled 'sum' - it's the address of its JITed
            # code in memory.
            cfptr = ee.get_function_address('test')
    
            # To convert an address to an actual callable thing we have to use
            # CFUNCTYPE, and specify the arguments & return type.
            cfunc = self.function_ctype(cfptr)
    
            # Now 'cfunc' is an actual callable we can invoke
            # TODO replace time.clock with a C implemententation for less overhead
            # TODO return result in machine readable format
            args = self.prepare_arguments()
            for i in range(10):
                start = time.perf_counter()
                res = cfunc(*args)
                end = time.perf_counter()
                benchtime = end-start
                cur_freq = psutil.cpu_freq().current*1e6
                print('The result ({}) in {:.6f} cy / it, {:.6f}s'.format(
                    res, benchtime/self.N*cur_freq, benchtime))


class ArithmeticLatencytTest(InstructionTest):
    def __init__(self, instruction='addq $1, $0',
                 dst_operands=(),
                 dstsrc_operands=(('r','i64', '0'),),
                 src_operands=(('i','i64', '1'),)):
        '''
        Build LLVM IR for arithmetic instruction latency benchmark without memory references.
                 
        Currently only one destination (dst) or combined destination and source (dstsrc) operand
        is allowed. Only 
        instruction's operands ($N) refer to the order of opernads found in dst + dstsrc + src.
        '''
        InstructionTest.__init__(self)
        self.loop_init = ''
        self.loop_body = ''
        if len(dst_operands) + len(dstsrc_operands) != 1:
            raise NotImplemented("Must have exactly one dst or dstsrc operand.")
        if not all([op[0] in 'ir'
                    for op in itertools.chain(dst_operands, dstsrc_operands, src_operands)]):
            raise NotImplemented("This class only supports register and immediate operands.")

        self.ret_llvmtype = dst_operands[0][1] if dst_operands else dstsrc_operands[0][1]
        
        # Part 1: PHI functions and initializations
        for i, dstsrc_op in enumerate(itertools.chain(dstsrc_operands)):
            # constraint code, llvm type string, initial value
            if dstsrc_op[0] == 'r':
                # register operand
                self.loop_body += (
                    '%"dstsrc{index}" = '
                    'phi {type} [{initial}, %"entry"], [%"dstsrc{index}.out", %"loop"]\n').format(
                        index=i, type=dstsrc_op[1], initial=dstsrc_op[2])
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
            
            operands = ['{type} {val}'.format(type=sop[1], val=sop[2]) for sop in src_operands]
            for i, dop in enumerate(dstsrc_operands):
                operands.append('{type} %dstsrc{index}'.format(type=dop[1], index=i))
            args = ', '.join(operands)
            self.loop_body += ('%"dstsrc{index}.out" = call {dst_type} asm sideeffect'
                               ' "{instruction}", "{constraints}" ({args})\n').format(
                index=i,
                dst_type=dstsrc_op[1],
                instruction=instruction,
                constraints=constraints,
                args=args
            )
            
        for i, dst_op in enumerate(dst_operands):
            # FIXME support dst operands
            # TODO support dst and dstsrc operands at the same time
            raise NotImplemented("Destination operand is not yet implemented")
        
        # Set %"ret" to something, needs to be a constant or phi function
        self.loop_tail = textwrap.dedent('''\
            %"ret" = phi {type} [{}, %"entry"], [%"dstsrc0.out", %"loop"]\
            '''.format(dstsrc_operands[0][2], type=dstsrc_operands[0][1]))


class AddressGenerationLatencytTest(InstructionTest):
    def __init__(self,
                 offset=('i', 'i64', '0x42'),
                 base=('r', 'i64', '0'),
                 index=('r', 'i64', '0'),
                 width=('i', None, '4'),
                 destination='base'):
        '''
        Test for address generation modes.
        
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
        InstructionTest.__init__(self)
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
        
        self.loop_init = ''
        self.loop_body = ''
        
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
            self.ret_llvmtype = base[1]
            destination_reg = base
        else:  # destination == 'index'
            self.ret_llvmtype = index[1]
            destination_reg = index
        
        # Part 1: PHI function for destination
        self.loop_body += (
            '%"{name}" = '
            'phi {type} [{initial}, %"entry"], [%"{name}.out", %"loop"]\n').format(
                name=destination, type=destination_reg[1], initial=destination_reg[2])
        
        constraints = '=r,r'
        if base and index:
            constraints += ',r'
            if destination == 'index':
                args = '{base_type} %"{base_name}", {index_type} {index_value}'.format(
                    base_type=base[1], base_name=destination,
                    index_type=index[1], index_value=index[2])
            else:  # destination == 'index':
                args ='{base_type} {base_value}, {index_type} %"{index_name}"'.format(
                    base_type=base[1], base_value=base[2],
                    index_type=index[1], index_name=destination)
        else:
            args = '{type} %"{name}"'.format(type=destination_reg[1], name=destination)
        
        self.loop_body += ('%"{name}.out" = call {type} asm sideeffect'
                           ' "lea {ops}", "{constraints}" ({args})\n').format(
            name=destination,
            type=destination_reg[1],
            ops=ops,
            constraints=constraints,
            args=args
        )

        # Set %"ret" to something, needs to be a constant or phi function
        self.loop_tail = textwrap.dedent('''\
            %"ret" = phi {type} [{initial_value}, %"entry"], [%"{name}.out", %"loop"]\
            '''.format(name=destination, initial_value=destination_reg[2], type=destination_reg[1]))


class LoadLatencyTest(InstructionTest):
    def __init__(self, chain_length=1024, repeat=100, structure='ring'):
        '''
        Build LLVM IR benchmark to test L1 load latency using pointer chasing.
        
        *chain_length* is the number of pointers to place in memory.
        *repeat* is the number of iterations the chain run through.
        *structure* may be 'linear' (1-offsets) or 'reverse' (jump back and forth) or 'random'.
        '''
        InstructionTest.__init__(self)
        self.loop_init = ''
        self.loop_body = ''
        self.loop_tail = ''
        element_type = ctypes.POINTER(ctypes.c_int)
        self.function_ctype = ctypes.CFUNCTYPE(
            ctypes.c_int, ctypes.POINTER(element_type), ctypes.c_int)
        self.chain_length = chain_length
        self.repeat = repeat
        self.N = chain_length*repeat
        self.pointer_field = (element_type * chain_length)()

        # Initialize pointer field
        # Field must represent a ring of pointers
        if structure == 'linear':
            for i in range(chain_length):
                self.pointer_field[i] = ctypes.cast(
                    ctypes.pointer(self.pointer_field[(i+1)%chain_length]), element_type)
        elif structure == 'jump':
            for i in range(chain_length):
                self.pointer_field[i] = ctypes.cast(
                    ctypes.pointer(self.pointer_field[(chain_length-i)%chain_length]), element_type)
        elif structure == 'random':
            shuffled_indices = list(range(chain_length))
            random.shuffle(shuffled_indices)
            for i in range(chain_length):
                self.pointer_field[i] = ctypes.cast(
                    ctypes.pointer(self.pointer_field[shuffled_indices[i]]), element_type)
    
    def prepare_arguments(self):
        return (self.pointer_field, self.repeat)
    
    def get_ir(self):
        '''
        Return LLVM IR equivalent of:
        
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
        return textwrap.dedent('''
        define i32 @test(i32**, i32) {
          %3 = bitcast i32** %0 to i32***
          %4 = load i32**, i32*** %3, align 8
          %5 = icmp sgt i32 %1, 0
          br i1 %5, label %6, label %17
        
          br label %7
        
          %8 = phi i32 [ %15, %14 ], [ 0, %6 ]
          br label %9
        
          %10 = phi i32** [ %4, %7 ], [ %12, %9 ]
          %11 = bitcast i32** %10 to i32***
          %12 = load i32**, i32*** %11, align 8
          %13 = icmp eq i32** %12, %4
          br i1 %13, label %14, label %9
        
          %15 = add i32 %8, 1
          %16 = icmp eq i32 %15, %1
          br i1 %16, label %17, label %7
        
          %18 = phi i32 [ 0, %2 ], [ %1, %14 ]
          ret i32 %18
        }
        ''')

if __name__ == '__main__':
    llvm.initialize()
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
    llvm.initialize_native_asmparser()
    
    modules = []
    
    # Construct with home grown IR builder
    modules.append(InstructionTest())
    # immediate source
    modules.append(ArithmeticLatencytTest(
        instruction='addq $1, $0',
        dst_operands=(),
        dstsrc_operands=(('r','i64', '0'),),
        src_operands=(('i','i64', '1'),)))

    # register source
    modules.append(ArithmeticLatencytTest(
        instruction='addq $1, $0',
        dst_operands=(),
        dstsrc_operands=(('r','i64', '0'),),
        src_operands=(('r','i64', '1'),)))
    
    # multiple instructions
    modules.append(ArithmeticLatencytTest(
        instruction='addq $1, $0\naddq $1, $0\naddq $1, $0\naddq $1, $0',
        dst_operands=(),
        dstsrc_operands=(('r','i64', '0'),),
        src_operands=(('i','i64', '1'),)))
    
    modules.append(AddressGenerationLatencytTest())
    
    modules.append(AddressGenerationLatencytTest(
        offset=None,
        base=('r', 'i64', '666'),
        index=None,
        width=None,
        destination='base'))
    
    modules.append(AddressGenerationLatencytTest(
        offset=None,
        base=None,
        index=('r', 'i64', '1'),
        width=('i', None, '4'),
        destination='index'))
    
    modules.append(AddressGenerationLatencytTest(
        offset=('i', 'i64', '-0x8'),
        base=None,
        index=('r', 'i64', '51'),
        width=('i', None, '4'),
        destination='index'))
    
    modules.append(AddressGenerationLatencytTest(
        offset=None,
        base=('r', 'i64', '23'),
        index=('r', 'i64', '12'),
        width=('i', None, '4'),
        destination='base'))
    
    modules = []
    modules.append(LoadLatencyTest(
        chain_length=2048,
        repeat=100000,
        structure='linear'))
    modules.append(LoadLatencyTest(
        chain_length=2048,
        repeat=100000,
        structure='random'))
    modules.append(LoadLatencyTest(
        chain_length=2048,
        repeat=100000,
        structure='reverse'))
    
    for module in modules:
        print("=== LLVM")
        print(module.get_ir())
        module.build_and_execute()
    