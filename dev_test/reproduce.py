#!/usr/bin/env python3
import ctypes

import llvmlite.binding as llvm


llvm.initialize()
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()
llvm.initialize_native_asmparser()

code = """
define i64 @"test"(i64 %"N")
{
entry:
  %"loop_cond" = icmp slt i64 0, %"N"
  br i1 %"loop_cond", label %"loop", label %"end"

loop:
  %"loop_counter" = phi i64 [0, %"entry"], [%"loop_counter.1", %"loop"]
  %"in.0" = phi i32 [3, %"entry"], [%"out.0", %"loop"]


  %"reg.0" = call i32 asm  "add $2, $0", "=r,0,i" (i32 %"in.0", i32 1)
  %"out.0" = call i32 asm  "add $2, $0", "=r,0,i" (i32 %"reg.0", i32 1)
  %"loop_counter.1" = add i64 %"loop_counter", 1
  %"loop_cond.1" = icmp slt i64 %"loop_counter.1", %"N"
  br i1 %"loop_cond.1", label %"loop", label %"end"

end:
  %"ret" = phi i64 [0, %"entry"], [%"loop_counter", %"loop"]

  ret i64 %"ret"
}
"""

features = llvm.get_host_cpu_features().flatten()
# znver1 on naples and skylake-avx512 on skylake-sp
for cpu in ["skylake-avx512", "znver1"]:
    tm =  llvm.Target.from_default_triple().create_target_machine(
        cpu=cpu, opt=2)
    tm.set_asm_verbosity(0)

    module = llvm.parse_assembly(code)
    asm = tm.emit_assembly(module)
    print(asm)
    with llvm.create_mcjit_compiler(module, tm) as ee:
        ee.finalize_object()
        cfptr = ee.get_function_address('test')
        cfunc = ctypes.CFUNCTYPE(ctypes.c_int64, ctypes.c_int64)(cfptr)
        print('->', cfunc(100000))


