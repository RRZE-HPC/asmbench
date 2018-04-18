define i64 @"test"(i64 %"N")
{
entry:
  %"loop_cond" = icmp slt i64 0, %"N"
  br i1 %"loop_cond", label %"loop", label %"end"
loop:
  %"loop_counter" = phi i64 [0, %"entry"], [%"loop_counter.1", %"loop"]
  %"checksum" = phi i64 [0, %"entry"], [%"checksum.1", %"loop"]
  %"extra_regs_1" = phi i64 [0, %"entry"], [%"extra_regs_1.1", %"loop"]
  %"extra_regs_2" = phi i64 [0, %"entry"], [%"extra_regs_2.1", %"loop"]
  %"extra_regs_3" = phi i64 [0, %"entry"], [%"extra_regs_3.1", %"loop"]
  %"extra_regs_4" = phi i64 [0, %"entry"], [%"extra_regs_4.1", %"loop"]
  %"extra_regs_5" = phi i64 [0, %"entry"], [%"extra_regs_5.1", %"loop"]
  %"extra_regs_6" = phi i64 [0, %"entry"], [%"extra_regs_6.1", %"loop"]
  %"extra_regs_7" = phi i64 [0, %"entry"], [%"extra_regs_7.1", %"loop"]
  %"extra_regs_8" = phi i64 [0, %"entry"], [%"extra_regs_8.1", %"loop"]
  %"extra_regs_9" = phi i64 [0, %"entry"], [%"extra_regs_9.1", %"loop"]
  %"extra_regs_10" = phi i64 [0, %"entry"], [%"extra_regs_10.1", %"loop"]
  %"extra_regs_11" = phi i64 [0, %"entry"], [%"extra_regs_11.1", %"loop"]
  %"asm" = call { i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64 } asm sideeffect "
      add $12, $0
      add $12, $1
      add $12, $2
      add $12, $3
      add $12, $4
      add $12, $5
      add $12, $6
      add $12, $7
      add $12, $8
      add $12, $9
      add $12, $10
      add $12, $11",
      "=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,=r,i,r,r,r,r,r,r,r,r,r,r,r,r"
      (i64 1, i64 %"checksum", i64 %"extra_regs_1", i64 %"extra_regs_2", i64 %"extra_regs_3",
       i64 %"extra_regs_4", i64 %"extra_regs_5", i64 %"extra_regs_6", i64 %"extra_regs_7",
       i64 %"extra_regs_8", i64 %"extra_regs_9", i64 %"extra_regs_10", i64 %"extra_regs_11")
  %"checksum.1" = extractvalue { i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64 } %"asm", 0
  %"extra_regs_1.1" = extractvalue { i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64 } %"asm", 1
  %"extra_regs_2.1" = extractvalue { i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64 } %"asm", 2
  %"extra_regs_3.1" = extractvalue { i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64 } %"asm", 3
  %"extra_regs_4.1" = extractvalue { i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64 } %"asm", 4
  %"extra_regs_5.1" = extractvalue { i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64 } %"asm", 5
  %"extra_regs_6.1" = extractvalue { i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64 } %"asm", 6
  %"extra_regs_7.1" = extractvalue { i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64 } %"asm", 7
  %"extra_regs_8.1" = extractvalue { i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64 } %"asm", 8
  %"extra_regs_9.1" = extractvalue { i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64 } %"asm", 9
  %"extra_regs_10.1" = extractvalue { i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64 } %"asm", 10
  %"extra_regs_11.1" = extractvalue { i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64 } %"asm", 11
  %"loop_counter.1" = add i64 %"loop_counter", 1
  %"loop_cond.1" = icmp slt i64 %"loop_counter.1", %"N"
  br i1 %"loop_cond.1", label %"loop", label %"end"
end:
  %"ret" = phi i64 [0, %"entry"], [%"checksum.1", %"loop"]
  ret i64 %"ret"
}