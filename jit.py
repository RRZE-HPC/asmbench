#!/usr/bin/env python3
from ctypes import CFUNCTYPE, c_int
import sys
import time

import llvmlite.ir as ll
import llvmlite.binding as llvm
import psutil

llvm.initialize()
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()
llvm.initialize_native_asmparser()

tm = llvm.Target.from_default_triple().create_target_machine()
tm.set_asm_verbosity(100)

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
module = ll.Module()

counter_type = ll.IntType(32)
func_ty = ll.FunctionType(counter_type, [counter_type])
func = ll.Function(module, func_ty, name='test')

func.args[0].name = 'N'

bb_entry = func.append_basic_block('entry')
irbuilder = ll.IRBuilder(bb_entry)
bb_loop = irbuilder.append_basic_block('loop')
bb_end = irbuilder.append_basic_block('end')
init_counter = ll.Constant(counter_type, 0)
loop_cond = irbuilder.icmp_signed('<', init_counter, func.args[0], name="loop_cond")
irbuilder.cbranch(loop_cond, bb_loop, bb_end)

with irbuilder.goto_block(bb_loop):
    loop_counter_phi = irbuilder.phi(counter_type, name="loop_counter")
    loop_counter_phi.add_incoming(init_counter, bb_entry)
    
    loop_counter = irbuilder.add(loop_counter_phi, ll.Constant(counter_type, 1), name="loop_counter")
    
    # Insert assembly here:
    # IRBuilder.asm(ftype, asm, constraint, args, side_effect, name='')
    asm_ftype = ll.FunctionType(counter_type, [counter_type])
    asm = irbuilder.asm(asm_ftype,
                        "addl $2, $0\nsubl $3, $0\nsubl $4, $0\n"
                        "addl $2, $0\nsubl $3, $0\nsubl $4, $0\n"
                        "addl $2, $0\nsubl $3, $0\nsubl $4, $0\n"
                        "addl $2, $0\nsubl $3, $0\nsubl $4, $0",
                        "=r,r,i,i,i",
                        (loop_counter, counter_type(23), counter_type(13), counter_type(10)),
                        side_effect=True, name="asm")
    loop_counter_final = asm
    loop_counter_phi.add_incoming(loop_counter_final, bb_loop)
    
    loop_cond = irbuilder.icmp_signed('<', loop_counter_final, func.args[0], name="loop_cond")
    irbuilder.cbranch(loop_cond, bb_loop, bb_end)
with irbuilder.goto_block(bb_end):
    ret_phi = irbuilder.phi(counter_type, name="ret")
    ret_phi.add_incoming(init_counter, bb_entry)
    ret_phi.add_incoming(loop_counter_final, bb_loop)
    irbuilder.ret(ret_phi)

print('=== LLVM IR')
print(module)

# Convert textual LLVM IR into in-memory representation.
llvm_module = llvm.parse_assembly(str(module))
llvm_module.verify()

# Compile the module to machine code using MCJIT
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
        start = time.clock()
        res = cfunc(N)
        end = time.clock()
        benchtime = end-start
        cur_freq = psutil.cpu_freq().current*1e3
        print('The result ({}) in {} cy / it'.format(res, benchtime/N*cur_freq))
    
    
    