	.section	__TEXT,__text,regular,pure_instructions
	.macosx_version_min 10, 13
	.globl	_test                   ## -- Begin function test
	.p2align	4, 0x90
_test:                                  ## @test
	.cfi_startproc
## %bb.0:                               ## %entry
	testq	%rdi, %rdi
	jle	LBB0_1
## %bb.2:                               ## %loop.preheader
	movl	$3, %ecx
	movq	$-1, %rdx
	.p2align	4, 0x90
LBB0_3:                                 ## %loop
                                        ## =>This Inner Loop Header: Depth=1
	## InlineAsm Start
	addl	$1, %ecx
	## InlineAsm End
	leaq	1(%rdx), %rax
	addq	$2, %rdx
	cmpq	%rdi, %rdx
	movq	%rax, %rdx
	## InlineAsm Start
	addl	$1, %ecx
	## InlineAsm End
	jl	LBB0_3
## %bb.4:                               ## %end
	retq
LBB0_1:
	xorl	%eax, %eax
	retq
	.cfi_endproc
                                        ## -- End function

.subsections_via_symbols
