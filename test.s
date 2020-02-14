	.section	__TEXT,__text,regular,pure_instructions
	.macosx_version_min 10, 13
	.globl	_foo                    ## -- Begin function foo
	.p2align	4, 0x90
_foo:                                   ## @foo
	.cfi_startproc
## BB#0:
	pushq	%rbp
Lcfi0:
	.cfi_def_cfa_offset 16
Lcfi1:
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
Lcfi2:
	.cfi_def_cfa_register %rbp
	xorl	%eax, %eax
	testl	%edi, %edi
	jle	LBB0_2
	.p2align	4, 0x90
LBB0_1:                                 ## =>This Inner Loop Header: Depth=1
	## InlineAsm Start
	addl	$23, %eax

	## InlineAsm End
	## InlineAsm Start
	subl	$13, %eax

	## InlineAsm End
	## InlineAsm Start
	subl	$10, %eax

	## InlineAsm End
	incl	%eax
	cmpl	%edi, %eax
	jl	LBB0_1
LBB0_2:
	popq	%rbp
	retq
	.cfi_endproc
                                        ## -- End function
	.section	__TEXT,__literal8,8byte_literals
	.p2align	3               ## -- Begin function benchmark
LCPI1_0:
	.quad	4696837146684686336     ## double 1.0E+6
	.section	__TEXT,__text,regular,pure_instructions
	.globl	_benchmark
	.p2align	4, 0x90
_benchmark:                             ## @benchmark
	.cfi_startproc
## BB#0:
	pushq	%rbp
Lcfi3:
	.cfi_def_cfa_offset 16
Lcfi4:
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
Lcfi5:
	.cfi_def_cfa_register %rbp
	pushq	%r14
	pushq	%rbx
	subq	$48, %rsp
Lcfi6:
	.cfi_offset %rbx, -32
Lcfi7:
	.cfi_offset %r14, -24
	movq	%rsi, %r14
	movss	%xmm0, -20(%rbp)        ## 4-byte Spill
	movl	%edi, %ebx
	leaq	-56(%rbp), %rdi
	xorl	%esi, %esi
	callq	_gettimeofday
	movl	%ebx, %edi
	callq	*%r14
	leaq	-40(%rbp), %rdi
	xorl	%esi, %esi
	callq	_gettimeofday
	movq	-40(%rbp), %rax
	subq	-56(%rbp), %rax
	cvtsi2sdq	%rax, %xmm1
	movl	-32(%rbp), %eax
	subl	-48(%rbp), %eax
	xorps	%xmm0, %xmm0
	cvtsi2sdl	%eax, %xmm0
	mulsd	LCPI1_0(%rip), %xmm0
	addsd	%xmm1, %xmm0
	movss	-20(%rbp), %xmm1        ## 4-byte Reload
                                        ## xmm1 = mem[0],zero,zero,zero
	cvtss2sd	%xmm1, %xmm1
	divsd	%xmm1, %xmm0
	leaq	L_.str(%rip), %rdi
	movb	$1, %al
	callq	_printf
	addq	$48, %rsp
	popq	%rbx
	popq	%r14
	popq	%rbp
	retq
	.cfi_endproc
                                        ## -- End function
	.section	__TEXT,__literal4,4byte_literals
	.p2align	2               ## -- Begin function main
LCPI2_0:
	.long	1326386456              ## float 2.4E+9
	.section	__TEXT,__text,regular,pure_instructions
	.globl	_main
	.p2align	4, 0x90
_main:                                  ## @main
	.cfi_startproc
## BB#0:
	pushq	%rbp
Lcfi8:
	.cfi_def_cfa_offset 16
Lcfi9:
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
Lcfi10:
	.cfi_def_cfa_register %rbp
	movq	8(%rsi), %rdi
	callq	_atoi
	leaq	_foo(%rip), %rsi
	movss	LCPI2_0(%rip), %xmm0    ## xmm0 = mem[0],zero,zero,zero
	movl	%eax, %edi
	callq	_benchmark
	xorl	%eax, %eax
	popq	%rbp
	retq
	.cfi_endproc
                                        ## -- End function
	.section	__TEXT,__cstring,cstring_literals
L_.str:                                 ## @.str
	.asciz	"%.3f (clock cycles)\n"

	.comm	_latency,8,3            ## @latency
	.comm	_ninst,8,3              ## @ninst

.subsections_via_symbols
