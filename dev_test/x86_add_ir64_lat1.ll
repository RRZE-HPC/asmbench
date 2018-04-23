define i64 @"test"(i64 %"N")
{
entry:
  %"loop_cond" = icmp slt i64 0, %"N"
  br i1 %"loop_cond", label %"loop", label %"end"
loop:
  %"loop_counter" = phi i64 [0, %"entry"], [%"loop_counter.1", %"loop"]
  %"checksum" = phi i64 [0, %"entry"], [%"checksum.1", %"loop"]
  %"checksum.1" = call i64 asm sideeffect "
      addq $2, $1",
      "=r,r,i" (i64 %"checksum", i64 1)
  %"loop_counter.1" = add i64 %"loop_counter", 1
  %"loop_cond.1" = icmp slt i64 %"loop_counter.1", %"N"
  br i1 %"loop_cond.1", label %"loop", label %"end"
end:
  %"ret" = phi i64 [0, %"entry"], [%"checksum.1", %"loop"]
  ret i64 %"ret"
}