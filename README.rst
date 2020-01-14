asmbench
========

A benchmark toolkit for assembly instructions using the LLVM JIT.

Usage
=====

To benchmark latency and throughput of a 64bit integer add use the following command:

``asmbench 'add {src:i64:r}, {srcdst:i64:r}'``

To benchmark two instructions interleaved use this:

``asmbench 'add {src:i64:r}, {srcdst:i64:r}' 'sub {src:i64:r}, {srcdst:i64:r}'``

To find out more add `-h` for help and `-v` for verbose mode.

Operand Templates
=================
Operands always follow this form: ``{direction:data_type:pass_type}``.

Direction may be ``src``, ``dst`` or ``srcdst``. This will allow asmbench to serialize the code (wherever possible). ``src`` operands are  read, but not modiefied by the instruction. ``dst`` operands are modified to, but not read. ``srcdst`` operands will be read and modified by the instruction.

Data and Pass Types:

* ``i64:r`` -> 64bit general purpose register (gpr) (e.g., ``%rax``)
* ``i32:r`` -> 32bit gpr (e.g., ``%ecx``)
* ``<2 x double>:x`` -> 128bit SSE register with two double precision floating-point numbers (e.g., ``%xmm1``)
* ``<4 x float>:x`` -> 128bit SSE register with four single precision floating-point numbers (e.g., ``%xmm1``)
* ``<4 x double>:x`` -> 256bit AVX register with four single precision floating-point numbers (e.g., ``%ymm1``)
* ``<8 x float>:x`` -> 256bit AVX register with four single precision floating-point numbers (e.g., ``%ymm1``)
* ``<8 x double>:x`` -> 512bit AVX512 register with four single precision floating-point numbers (e.g., ``%zmm1``)
* ``<16 x float>:x`` -> 512bit AVX512 register with four single precision floating-point numbers (e.g., ``%zmm1``)
* ``i8:23`` -> immediate 0 (i.e., ``$23``)
