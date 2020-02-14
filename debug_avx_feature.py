#!/usr/bin/env python3
import llvmlite.binding as llvm

llvm.initialize()
# From
# >>> cp = (ctypes.c_char_p * 1)()
# >>> ffi.lib.LLVMPY_GetHostCPUFeatures(cp)
# >>> print(cp[0])
# llvm.set_option('', '-mattr=+sse2,+cx16,-tbm,-avx512ifma,-avx512dq,-fma4,+prfchw,+bmi2,+xsavec,+fsgsbase,+popcnt,+aes,+xsaves,-avx512er,-avx512vpopcntdq,-clwb,-avx512f,-clzero,-pku,+mmx,-lwp,-xop,+rdseed,-sse4a,-avx512bw,+clflushopt,+xsave,-avx512vl,-avx512cd,+avx,+rtm,+fma,+bmi,+rdrnd,-mwaitx,+sse4.1,+sse4.2,+avx2,+sse,+lzcnt,+pclmul,-prefetchwt1,+f16c,+ssse3,+sgx,+cmov,-avx512vbmi,+movbe,+xsaveopt,-sha,+adx,-avx512pf,+sse3')
# llvm.set_option('', '-march=native')
# llvm.set_option('', '-mcpu=native')
# llvm.set_option('', '-version')
# llvm.set_option('', '-help-list-hidden')
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()
llvm.initialize_native_asmparser()
# llvm.set_option('', '-help-list-hidden')

ir = '''

target triple = "x86_64-apple-darwin17.5.0"

define <4 x double> @testv(i32**, i32) {

  %out = tail call <4 x double> asm "vaddpd $1, $2, $0", "=x,x,x,~{dirflag},~{fpsr},~{flags}"(<4 x double> <double 0.123, double 0.123, double 0.123, double 0.123>, <4 x double> <double 0.123, double 0.123, double 0.123, double 0.123>)
  ret <4 x double> %out
}
'''

module = llvm.parse_assembly(ir)
module.verify()
features = llvm.get_host_cpu_features().flatten()
cpu = llvm.get_host_cpu_name()
tm = llvm.Target.from_default_triple().create_target_machine(cpu=cpu, features=features)
with llvm.create_mcjit_compiler(module, tm) as ee:
    ee.finalize_object()
    print(tm.emit_assembly(module))
