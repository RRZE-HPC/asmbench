define i64 @"test"(i64 %"N")
{
entry:
  %"loop_cond" = icmp slt i64 0, %"N"
  br i1 %"loop_cond", label %"loop", label %"end"
loop:
  %"loop_counter" = phi i64 [0, %"entry"], [%"loop_counter.1", %"loop"]
  %"dstsrc_0" = phi i64 [0, %"entry"], [%"dstsrc_0.1", %"loop"]
  %"dstsrc_1" = phi i64 [0, %"entry"], [%"dstsrc_1.1", %"loop"]
  %"dstsrc_2" = phi i64 [0, %"entry"], [%"dstsrc_2.1", %"loop"]
  %"dstsrc_3" = phi i64 [0, %"entry"], [%"dstsrc_3.1", %"loop"]
  %"dstsrc_4" = phi i64 [0, %"entry"], [%"dstsrc_4.1", %"loop"]
  %"dstsrc_5" = phi i64 [0, %"entry"], [%"dstsrc_5.1", %"loop"]
  %"dstsrc_6" = phi i64 [0, %"entry"], [%"dstsrc_6.1", %"loop"]
  %"dstsrc_7" = phi i64 [0, %"entry"], [%"dstsrc_7.1", %"loop"]
  %"dstsrc_8" = phi i64 [0, %"entry"], [%"dstsrc_8.1", %"loop"]
  %"dstsrc_9" = phi i64 [0, %"entry"], [%"dstsrc_9.1", %"loop"]
  %"dstsrc_10" = phi i64 [0, %"entry"], [%"dstsrc_10.1", %"loop"]
  %"dstsrc_11" = phi i64 [0, %"entry"], [%"dstsrc_11.1", %"loop"]
  %"dstsrc_0.1" = call i64 asm sideeffect "add $1, $0", "=r,i,r" (i64 1, i64 %"dstsrc_0")
  %"dstsrc_1.1" = call i64 asm sideeffect "add $1, $0", "=r,i,r" (i64 1, i64 %"dstsrc_1")
  %"dstsrc_2.1" = call i64 asm sideeffect "add $1, $0", "=r,i,r" (i64 1, i64 %"dstsrc_2")
  %"dstsrc_3.1" = call i64 asm sideeffect "add $1, $0", "=r,i,r" (i64 1, i64 %"dstsrc_3")
  %"dstsrc_4.1" = call i64 asm sideeffect "add $1, $0", "=r,i,r" (i64 1, i64 %"dstsrc_4")
  %"dstsrc_5.1" = call i64 asm sideeffect "add $1, $0", "=r,i,r" (i64 1, i64 %"dstsrc_5")
  %"dstsrc_6.1" = call i64 asm sideeffect "add $1, $0", "=r,i,r" (i64 1, i64 %"dstsrc_6")
  %"dstsrc_7.1" = call i64 asm sideeffect "add $1, $0", "=r,i,r" (i64 1, i64 %"dstsrc_7")
  %"dstsrc_8.1" = call i64 asm sideeffect "add $1, $0", "=r,i,r" (i64 1, i64 %"dstsrc_8")
  %"dstsrc_9.1" = call i64 asm sideeffect "add $1, $0", "=r,i,r" (i64 1, i64 %"dstsrc_9")
  %"dstsrc_10.1" = call i64 asm sideeffect "add $1, $0", "=r,i,r" (i64 1, i64 %"dstsrc_10")
  %"dstsrc_11.1" = call i64 asm sideeffect "add $1, $0", "=r,i,r" (i64 1, i64 %"dstsrc_11")
  %"loop_counter.1" = add i64 %"loop_counter", 1
  %"loop_cond.1" = icmp slt i64 %"loop_counter.1", %"N"
  br i1 %"loop_cond.1", label %"loop", label %"end"
end:
  %"ret" = phi i64 [0, %"entry"], [%"dstsrc_0.1", %"loop"]
  ret i64 %"ret"
}