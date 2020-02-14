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
