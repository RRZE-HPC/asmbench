#!/usr/bin/env python3
import ctypes
import sys
import time
import textwrap
import itertools

import llvmlite.binding as llvm
import psutil

# TODOs
# * API to create test scenarios
#   * DSL?
# * Test cases:
#   * Instructions:
#     * scalar
#     * packed
#     * memory references
#   * Single Latency
#   * Single Throughput
#   * Combined Latency
#   * Combined throughput
#   * Random throughput
# * IACA marked binary output generation
# * Fuzzing algorithm
# * CLI
# * C based timing routine? As an extension?
# * make sanity checks during runtime, check for fixed frequency and pinning

def construct_test_module():
    import llvmlite.ir as ll
    # Create a new module with a function implementing this:
    #
    # int test(int N) {
    #   int i=0
    #   while(i<N) {
    #       // ASM things go here
    #       i++;
    #   }
    #   return i;  // replace by checksum like value
    # }
    
    # Module configuration
    counter_type = ll.IntType(64)
    checksum_type = ll.IntType(64)  # ll.DoubleType()
    checksum_init = checksum_type(0)
    
    # Module
    module = ll.Module()
    func_ty = ll.FunctionType(counter_type, [counter_type])
    func = ll.Function(module, func_ty, name='test')
    func.args[0].name = 'N'
    
    # Code
    bb_entry = func.append_basic_block('entry')
    irbuilder = ll.IRBuilder(bb_entry)
    bb_loop = irbuilder.append_basic_block('loop')
    bb_end = irbuilder.append_basic_block('end')
    counter_init = counter_type(0)
    loop_cond = irbuilder.icmp_signed('<', counter_init, func.args[0], name="loop_cond")
    irbuilder.cbranch(loop_cond, bb_loop, bb_end)
    
    with irbuilder.goto_block(bb_loop):
        # Loop mechanics & Checksum (1)
        loop_counter_phi = irbuilder.phi(counter_type, name="loop_counter")
        checksum_phi = irbuilder.phi(checksum_type, name="checksum")
        loop_counter_phi.add_incoming(counter_init, bb_entry)
        loop_counter = irbuilder.add(loop_counter_phi, ll.Constant(counter_type, 1), name="loop_counter")
        loop_counter_phi.add_incoming(loop_counter, bb_loop)
        checksum_phi.add_incoming(checksum_init, bb_entry)
        
        # Insert assembly here:
        # IRBuilder.asm(ftype, asm, constraint, args, side_effect, name='')
        asm_ftype = ll.FunctionType(counter_type, [counter_type])
        checksum = irbuilder.asm(
            asm_ftype,
            "add $2, $0\n",
            "=r,r,i",
            (checksum_phi, checksum_type(1)),
            side_effect=True, name="asm")
        
        # Loop mechanics & Checksum (2)
        checksum_phi.add_incoming(checksum, bb_loop)
        loop_cond = irbuilder.icmp_signed('<', loop_counter, func.args[0], name="loop_cond")
        irbuilder.cbranch(loop_cond, bb_loop, bb_end)
    with irbuilder.goto_block(bb_end):
        ret_phi = irbuilder.phi(counter_type, name="ret")
        ret_phi.add_incoming(checksum_init, bb_entry)
        ret_phi.add_incoming(checksum, bb_loop)
        irbuilder.ret(ret_phi)
    
    return module
        

class InstructionTest:
    LLVM2CTYPE = {
        'i8': ctypes.c_int8,
        'i16': ctypes.c_int16,
        'i32': ctypes.c_int32,
        'i64': ctypes.c_int64,
        'f32': ctypes.c_float,
        'f64': ctypes.c_double,
    }
    def __init__(self):
        self.loop_init = ''
        self.ret_llvmtype = 'i64'
        self.ret_ctype = self.LLVM2CTYPE[self.ret_llvmtype]
        
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
    
    def __str__(self):
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


class LatencytTest(InstructionTest):
    def __init__(self, instruction='addq $1, $0',
                 dst_operands=(),
                 dstsrc_operands=(('r','i64', '0'),),
                 src_operands=(('i','i64', '1'),),
                 instruction_count=1):
        '''
        instruction's operands ($N) refer to the order of opernads found in dst + dstsrc + src.
        '''
        self.loop_init = ''
        self.loop_body = ''
        if len(dst_operands) + len(dstsrc_operands) > 1:
            raise NotImplemented("Currently only none or exactly one dst or dstsrc operand is "
                                 "supported.")

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
            # TODO add support for memory operands
            elif dstsrc_op[0] == 'm':
                # memory operand
                self.loop_init += (
                    '%"dstsrc{index}" = alloca {type}\n'
                    'store {type} {initial}, {type}* %"dstsrc{index}"\n').format(
                        index=i, type=dstsrc_op[1], initial=dstsrc_op[2])
            else:
                raise NotImplemented("Operand type in {!r} is not yet supported.".format(dstsrc_op))
        
        for i, dst_op in enumerate(itertools.chain(dst_operands)):
            # No phi functions necessary
            pass
        
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
                type_ = dop[1]
                if dop[0] == 'm':
                    # Make pointer if it is a memory reference
                    type_ += '*'
                operands.append('{type} %dstsrc{index}'.format(type=type_, index=i))
            args = ', '.join(operands)
            dst_type = dstsrc_op[1]
            if dstsrc_op[0] == 'm':
                dst_type += '*'
            self.loop_body += ('%"dstsrc{index}.out" = call {dst_type} asm sideeffect'
                               ' "{instruction}", "{constraints}" ({args})\n').format(
                index=i,
                dst_type=dst_type,
                instruction=instruction,
                constraints=constraints,
                args=args
            )
            
        for i, dst_op in enumerate(dst_operands):
            # FIXME support dst operands
            # TODO support dst and dstsrc operands at the same time
            raise NotImplemented("Destination operands are not yet implemented")
        
        # Set %"ret" to something, needs to be a constant or phi function
        if dstsrc_operands[0][0] == 'm':
            self.loop_tail = textwrap.dedent('''\
                %"ret" = load {type}, {type}* %"dstsrc0"\
                '''.format(type=dstsrc_operands[0][1]))
        else:
            self.loop_tail = textwrap.dedent('''\
                %"ret" = phi {type} [{}, %"entry"], [%"dstsrc0.out", %"loop"]\
                '''.format(dstsrc_operands[0][2], type=dstsrc_operands[0][1]))


if __name__ == '__main__':
    llvm.initialize()
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
    llvm.initialize_native_asmparser()
    
    # Option 1: Construct using llvmlite's irbuilder
    module = construct_test_module()
    
    # Option 2: Use raw LLVM IR file
    with open('dev_test/x86_add_ir64_lat1.ll') as f:
        module = f.read()
    
    # Option 3: Construct with home grown IR builder
    # module = str(InstructionTest())
    # immediate source
    module = str(LatencytTest(instruction='addq $1, $0',
                              dst_operands=(),
                              dstsrc_operands=(('r','i64', '0'),),
                              src_operands=(('i','i64', '1'),)))

    # register source
    module = str(LatencytTest(instruction='addq $1, $0',
                              dst_operands=(),
                              dstsrc_operands=(('r','i64', '0'),),
                              src_operands=(('r','i64', '1'),)))
    
    # mem ref source
    module = str(LatencytTest(instruction='addq $1, $0\naddq $1, $0\naddq $1, $0\naddq $1, $0',
                              dst_operands=(),
                              dstsrc_operands=(('r','i64', '0'),),
                              src_operands=(('m','i64', '1'),)))
    
    # multiple instructions
    module = str(LatencytTest(instruction='addq $1, $0\naddq $1, $0\naddq $1, $0\naddq $1, $0',
                              dst_operands=(),
                              dstsrc_operands=(('r','i64', '0'),),
                              src_operands=(('i','i64', '1'),)))
    
    # TODO mem ref source destination
    #module = str(LatencytTest(instruction='addq $1, $0',
    #                          dst_operands=(),
    #                          dstsrc_operands=(),
    #                          src_operands=(('m','i64', '0'), ('i','i64', '1'))))
    print('=== LLVM IR')
    print(module)
    
    # Convert textual LLVM IR into in-memory representation.
    llvm_module = llvm.parse_assembly(str(module))
    llvm_module.verify()
    
    # Compile the module to machine code using MCJIT
    tm = llvm.Target.from_default_triple().create_target_machine()
    tm.set_asm_verbosity(100)
    with llvm.create_mcjit_compiler(llvm_module, tm) as ee:
        ee.finalize_object()
        print('=== Assembly')
        print(tm.emit_assembly(llvm_module))
        
        # ??? ee.run_static_constructors()
        # Obtain a pointer to the compiled 'sum' - it's the address of its JITed
        # code in memory.
        cfptr = ee.get_function_address('test')
    
        # To convert an address to an actual callable thing we have to use
        # CFUNCTYPE, and specify the arguments & return type.
        cfunc = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int)(cfptr)
    
        # Now 'cfunc' is an actual callable we can invoke
        # TODO replace time.clock with a C implemententation for less overhead
        N = 100000000
        for i in range(10):
            start = time.perf_counter()
            res = cfunc(N)
            end = time.perf_counter()
            benchtime = end-start
            cur_freq = psutil.cpu_freq().current*1e6
            print('The result ({}) in {:.6f} cy / it, {:.6f}s'.format(res, benchtime/N*cur_freq, benchtime))
    