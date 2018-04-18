#!/usr/bin/env python3
import ctypes
import sys
import time
import textwrap
import functools

import llvmlite.binding as llvm
import psutil

# TODOs
# * API to create test scenarios
#   * DSL?
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
    def __init__(self, instruction='addq $1, $0', dst_operands=[], dstsrc_operands=[('r','i64', '0')],
                 src_operands=[('i','i64', '1')]):
        '''
        instruction's operands ($N) refer to the order of opernads found in dst + dstsrc + src.
        '''
        self.loop_init = ''
        self.loop_body = ''
        if len(dst_operands) + len(dstsrc_operands) != 1:
            raise ValueError("There must be exactly one dst or dstsrc oprand. Future versions "
                            "might support multiple or no outputs.")

        # Part 1: PHI functions
        for i, dstsrc_op in enumerate(functools.chain(dstsrc_operands)):
            # constraint code, llvm type string, initial value
            if dstsrc_op[0] == 'r':
                # register operand
                self.loop_body += \
                    '%"dstsrc{index}" = phi {type} [{initial}, %"entry"], [%"dst{index}.out", %"loop"]'.format(
                        index=i, type=dstsrc_op[1], initial=dstsrc_op[2])
            # TODO add support for memory operands
            #elif dst_op[0] == 'm':
            #    # memory operand
            else:
                raise ValueError("Operand type in {!r} is not supported here.".format(dstsrc_op))
        
        for i, dst_op in enumerate(functools.chain(dst_operands)):
            # No phi functions necessary
            pass
        
        for i, src_op in enumerate(functools.chain(src_operands)):
            # No phi functions necessary
            pass
        
        # Part 2: Inline ASM call
        for i, dstsrc_op in enumerate(functools.chain(dstsrc_operands)):
            # Build instruction from instruction and operands
            # TODO either document order of operands in instruction string or replace by DSL
            # Build constraint string from operands
            constraints = ','.join(
                ['='+dop[0] for dop in functools.chain(dst_operands, dstsrc_operands)] +
                [sop[0] for sop in functools.chain(src_operands, dstsrc_operands)])
            self.loop_body += '%"dstsrc" = call {dst_type} asm sideeffect "{instruction}", "{constraints}", ({args})'.format(
                dst_type=dstsrc_op[1],
                instruction=instruction,
                constraints=constraints,
                
            )
            
        
        for i, dst_op in enumerate(functools.chain(dst_operands)):
            # No phi functions necessary
            pass
        
        for i, src_op in enumerate(functools.chain(src_operands)):
            # No phi functions necessary
            pass
                
        textwrap.dedent('''\
            %"checksum" = phi i64 [0, %"entry"], [%"checksum.1", %"loop"]
            %"checksum.1" = call i64 asm sideeffect "
                add $1, $0",
                "=r,i,r" (i64 1, i64 %"checksum")\
            ''')
        
        # Set %"ret" to something, needs to be a constant or phi function
        self.loop_tail = textwrap.dedent('''\
            %"ret" = phi i64 [{}, %"entry"], [%"dstsrc0.1", %"loop"]\
            '''.format(dstsrc_operand[0][2]))


if __name__ == '__main__':
    llvm.initialize()
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
    llvm.initialize_native_asmparser()
    
    module = construct_test_module()
    
    # Alternative, use raw LLVM IR:
    module = '''
    define i64 @"test"(i64 %"N")
    {
    entry:
      %"loop_cond" = icmp slt i64 0, %"N"
      br i1 %"loop_cond", label %"loop", label %"end"
    loop:
      %"loop_counter" = phi i64 [0, %"entry"], [%"loop_counter.1", %"loop"]
      %"checksum" = phi i64 [0, %"entry"], [%"checksum.1", %"loop"]
      %"extra_regs_1" = phi i64 [0, %"entry"], [%"extra_regs_1.1", %"loop"]
      %"extra_regs_2" = phi i64 [0, %"entry"], [%"extra_regs_2.1", %"loop"]
      %"extra_regs_3" = phi i64 [0, %"entry"], [%"extra_regs_3.1", %"loop"]
      %"extra_regs_4" = phi i64 [0, %"entry"], [%"extra_regs_4.1", %"loop"]
      %"extra_regs_5" = phi i64 [0, %"entry"], [%"extra_regs_5.1", %"loop"]
      %"extra_regs_6" = phi i64 [0, %"entry"], [%"extra_regs_6.1", %"loop"]
      %"extra_regs_7" = phi i64 [0, %"entry"], [%"extra_regs_7.1", %"loop"]
      %"extra_regs_8" = phi i64 [0, %"entry"], [%"extra_regs_8.1", %"loop"]
      %"extra_regs_9" = phi i64 [0, %"entry"], [%"extra_regs_9.1", %"loop"]
      %"extra_regs_10" = phi i64 [0, %"entry"], [%"extra_regs_10.1", %"loop"]
      %"extra_regs_11" = phi i64 [0, %"entry"], [%"extra_regs_11.1", %"loop"]
      %"asm" = call { i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64 } asm sideeffect "
          add $12, $0
          add $12, $1
          add $12, $2
          add $12, $3
          add $12, $4
          add $12, $5
          add $12, $6
          add $12, $7
          add $12, $8
          add $12, $9
          add $12, $10
          add $12, $11",
          "=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,i,r,r,r,r,r,r,r,r,r,r,r,r"
          (i64 1, i64 %"checksum", i64 %"extra_regs_1", i64 %"extra_regs_2", i64 %"extra_regs_3",
           i64 %"extra_regs_4", i64 %"extra_regs_5", i64 %"extra_regs_6", i64 %"extra_regs_7",
           i64 %"extra_regs_8", i64 %"extra_regs_9", i64 %"extra_regs_10", i64 %"extra_regs_11")
      %"checksum.1" = extractvalue { i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64 } %"asm", 0
      %"extra_regs_1.1" = extractvalue { i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64 } %"asm", 1
      %"extra_regs_2.1" = extractvalue { i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64 } %"asm", 2
      %"extra_regs_3.1" = extractvalue { i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64 } %"asm", 3
      %"extra_regs_4.1" = extractvalue { i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64 } %"asm", 4
      %"extra_regs_5.1" = extractvalue { i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64 } %"asm", 5
      %"extra_regs_6.1" = extractvalue { i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64 } %"asm", 6
      %"extra_regs_7.1" = extractvalue { i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64 } %"asm", 7
      %"extra_regs_8.1" = extractvalue { i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64 } %"asm", 8
      %"extra_regs_9.1" = extractvalue { i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64 } %"asm", 9
      %"extra_regs_10.1" = extractvalue { i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64 } %"asm", 10
      %"extra_regs_11.1" = extractvalue { i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64 } %"asm", 11
      %"loop_counter.1" = add i64 %"loop_counter", 1
      %"loop_cond.1" = icmp slt i64 %"loop_counter.1", %"N"
      br i1 %"loop_cond.1", label %"loop", label %"end"
    end:
      %"ret" = phi i64 [0, %"entry"], [%"checksum.1", %"loop"]
      ret i64 %"ret"
    }
    '''
    
    module = '''
    define i64 @"test"(i64 %"N")
    {
    entry:
      %"loop_cond" = icmp slt i64 0, %"N"
      br i1 %"loop_cond", label %"loop", label %"end"
    loop:
      %"loop_counter" = phi i64 [0, %"entry"], [%"loop_counter.1", %"loop"]
      %"dstsrc_0" = phi i64 [0, %"entry"], [%"dstsrc_0.1", %"loop"]
      %"dstsrc_1" = phi i64 [0, %"entry"], [%"dstsrc_1.1", %"loop"]
      %"dstsrc_2" = phi i64 [0, %"entry"], [%"dstsrc_2.1", %"loop"]
      %"dstsrc_3" = phi i64 [0, %"entry"], [%"dstsrc_3.1", %"loop"]
      %"dstsrc_4" = phi i64 [0, %"entry"], [%"dstsrc_4.1", %"loop"]
      %"dstsrc_5" = phi i64 [0, %"entry"], [%"dstsrc_5.1", %"loop"]
      %"dstsrc_6" = phi i64 [0, %"entry"], [%"dstsrc_6.1", %"loop"]
      %"dstsrc_7" = phi i64 [0, %"entry"], [%"dstsrc_7.1", %"loop"]
      %"dstsrc_8" = phi i64 [0, %"entry"], [%"dstsrc_8.1", %"loop"]
      %"dstsrc_9" = phi i64 [0, %"entry"], [%"dstsrc_9.1", %"loop"]
      %"dstsrc_10" = phi i64 [0, %"entry"], [%"dstsrc_10.1", %"loop"]
      %"dstsrc_11" = phi i64 [0, %"entry"], [%"dstsrc_11.1", %"loop"]
      %"dstsrc_0.1" = call i64 asm sideeffect "add $1, $0", "=r,i,r" (i64 1, i64 %"dstsrc_0")
      %"dstsrc_1.1" = call i64 asm sideeffect "add $1, $0", "=r,i,r" (i64 1, i64 %"dstsrc_1")
      %"dstsrc_2.1" = call i64 asm sideeffect "add $1, $0", "=r,i,r" (i64 1, i64 %"dstsrc_2")
      %"dstsrc_3.1" = call i64 asm sideeffect "add $1, $0", "=r,i,r" (i64 1, i64 %"dstsrc_3")
      %"dstsrc_4.1" = call i64 asm sideeffect "add $1, $0", "=r,i,r" (i64 1, i64 %"dstsrc_4")
      %"dstsrc_5.1" = call i64 asm sideeffect "add $1, $0", "=r,i,r" (i64 1, i64 %"dstsrc_5")
      %"dstsrc_6.1" = call i64 asm sideeffect "add $1, $0", "=r,i,r" (i64 1, i64 %"dstsrc_6")
      %"dstsrc_7.1" = call i64 asm sideeffect "add $1, $0", "=r,i,r" (i64 1, i64 %"dstsrc_7")
      %"dstsrc_8.1" = call i64 asm sideeffect "add $1, $0", "=r,i,r" (i64 1, i64 %"dstsrc_8")
      %"dstsrc_9.1" = call i64 asm sideeffect "add $1, $0", "=r,i,r" (i64 1, i64 %"dstsrc_9")
      %"dstsrc_10.1" = call i64 asm sideeffect "add $1, $0", "=r,i,r" (i64 1, i64 %"dstsrc_10")
      %"dstsrc_11.1" = call i64 asm sideeffect "add $1, $0", "=r,i,r" (i64 1, i64 %"dstsrc_11")
      %"loop_counter.1" = add i64 %"loop_counter", 1
      %"loop_cond.1" = icmp slt i64 %"loop_counter.1", %"N"
      br i1 %"loop_cond.1", label %"loop", label %"end"
    end:
      %"ret" = phi i64 [0, %"entry"], [%"dstsrc_0.1", %"loop"]
      ret i64 %"ret"
    }
    '''
    
    module2 = '''
    define i64 @"test"(i64 %"N")
    {
    entry:
      %"loop_cond" = icmp slt i64 0, %"N"
      br i1 %"loop_cond", label %"loop", label %"end"
    loop:
      %"loop_counter" = phi i64 [0, %"entry"], [%"loop_counter.1", %"loop"]
      %"checksum" = phi i64 [0, %"entry"], [%"checksum.1", %"loop"]
      %"checksum.1" = call i64 asm sideeffect "
          add $0, $1",
          "*r,i" (i64 %"checksum", i64 1)
      %"loop_counter.1" = add i64 %"loop_counter", 1
      %"loop_cond.1" = icmp slt i64 %"loop_counter.1", %"N"
      br i1 %"loop_cond.1", label %"loop", label %"end"
    end:
      %"ret" = phi i64 [0, %"entry"], [%"checksum.1", %"loop"]
      ret i64 %"ret"
    }
    '''
    
    module3 = str(InstructionTest())
    
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
        for i in range(100):
            start = time.perf_counter()
            res = cfunc(N)
            end = time.perf_counter()
            benchtime = end-start
            cur_freq = psutil.cpu_freq().current*1e6
            print('The result ({}) in {:.6f} cy / it, {:.6f}s'.format(res, benchtime/N*cur_freq, benchtime))
    