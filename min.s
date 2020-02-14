	.section	__TEXT,__text,regular,pure_instructions
	.macosx_version_min 10, 13
	.section	__TEXT,__literal8,8byte_literals
	.p2align	3               ## -- Begin function testv
LCPI0_0:
	.quad	4593527504729830064     ## 0x3fbf7ced916872b0
	.section	__TEXT,__text,regular,pure_instructions
	.globl	_testv
	.p2align	4, 0x90
_testv:                                 ## @testv
	.cfi_startproc
## BB#0:
	vbroadcastsd	LCPI0_0(%rip), %ymm0 ## ymm0 = [4593527504729830064,4593527504729830064,4593527504729830064,4593527504729830064]
	## InlineAsm Start
	vaddpd	%ymm0, %ymm0, %ymm0
	## InlineAsm End
	retq
	.cfi_endproc
                                        ## -- End function

.subsections_via_symbols
