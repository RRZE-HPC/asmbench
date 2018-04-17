#!/usr/bin/env python3
from ctypes import CFUNCTYPE, c_int
import sys
import time

import llvmlite.ir as ll
import llvmlite.binding as llvm
import psutil

def construct_test():
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
    counter_type = ll.IntType(32)
    checksum_type = ll.IntType(32)  # ll.DoubleType()
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
            "addl $2, $0\n",
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

if __name__ == '__main__':
    llvm.initialize()
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
    llvm.initialize_native_asmparser()
    
    module = construct_test()
    
    # Alternative, use raw LLVM IR:
    module = '''
    define i32 @"test"(i32 %"N")
    {
    entry:
      %"loop_cond" = icmp slt i32 0, %"N"
      br i1 %"loop_cond", label %"loop", label %"end"
    loop:
      %"loop_counter" = phi i32 [0, %"entry"], [%"loop_counter.1", %"loop"]
      %"checksum" = phi i32 [0, %"entry"], [%"checksum.1", %"loop"]
      %"extra_regs_1" = phi i32 [0, %"entry"], [%"extra_regs_1.1", %"loop"]
      %"extra_regs_2" = phi i32 [0, %"entry"], [%"extra_regs_2.1", %"loop"]
      %"extra_regs_3" = phi i32 [0, %"entry"], [%"extra_regs_3.1", %"loop"]
      %"loop_counter.1" = add i32 %"loop_counter", 1
      %"asm" = call { i32, i32, i32, i32 } asm sideeffect "
          addl $4, $0
          addl $4, $1
          addl $4, $2
          addl $4, $3",
          "=r,=r,=r,=r,i,r,r,r,r" (i32 1, i32 %"checksum", i32 %"extra_regs_1", i32 %"extra_regs_2", i32 %"extra_regs_3")
      %"checksum.1" = extractvalue { i32, i32, i32, i32 } %"asm", 0
      %"extra_regs_1.1" = extractvalue { i32, i32, i32, i32 } %"asm", 1
      %"extra_regs_2.1" = extractvalue { i32, i32, i32, i32 } %"asm", 2
      %"extra_regs_3.1" = extractvalue { i32, i32, i32, i32 } %"asm", 3
      %"loop_cond.1" = icmp slt i32 %"loop_counter.1", %"N"
      br i1 %"loop_cond.1", label %"loop", label %"end"
    end:
      %"ret" = phi i32 [0, %"entry"], [%"checksum.1", %"loop"]
      ret i32 %"ret"
    }
    '''
    
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
        cfunc = CFUNCTYPE(c_int, c_int)(cfptr)
    
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
    