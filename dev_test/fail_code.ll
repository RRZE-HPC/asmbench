define i64 @"test"(i64 %"N")
{
entry:
  %"loop_cond" = icmp slt i64 0, %"N"
  br i1 %"loop_cond", label %"loop", label %"end"

loop:
  %"loop_counter" = phi i64 [0, %"entry"], [%"loop_counter.1", %"loop"]
  %in.0 = phi i64 [1, %"entry"], [%out.0, %"loop"]
  %in.1 = phi i64 [1, %"entry"], [%out.1, %"loop"]
  %in.2 = phi i64 [1, %"entry"], [%out.2, %"loop"]
  %in.3 = phi i64 [1, %"entry"], [%out.3, %"loop"]
  %in.4 = phi i64 [1, %"entry"], [%out.4, %"loop"]
  %in.5 = phi i64 [1, %"entry"], [%out.5, %"loop"]
  %in.6 = phi i64 [1, %"entry"], [%out.6, %"loop"]
  %in.7 = phi i64 [1, %"entry"], [%out.7, %"loop"]

  %"reg.0" = call i64 asm  "add $2, $0", "=r,0,i" (i64 %in.0, i64 1)
  %"reg.1" = call i64 asm  "add $2, $0", "=r,0,i" (i64 %"reg.0", i64 1)
  %"reg.2" = call i64 asm  "add $2, $0", "=r,0,i" (i64 %"reg.1", i64 1)
  %"reg.3" = call i64 asm  "add $2, $0", "=r,0,i" (i64 %"reg.2", i64 1)
  %"reg.4" = call i64 asm  "add $2, $0", "=r,0,i" (i64 %"reg.3", i64 1)
  %"reg.5" = call i64 asm  "add $2, $0", "=r,0,i" (i64 %"reg.4", i64 1)
  %"reg.6" = call i64 asm  "add $2, $0", "=r,0,i" (i64 %"reg.5", i64 1)
  %out.0 = call i64 asm  "add $2, $0", "=r,0,i" (i64 %"reg.6", i64 1)
  %"reg.7" = call i64 asm  "add $2, $0", "=r,0,i" (i64 %in.1, i64 1)
  %"reg.8" = call i64 asm  "add $2, $0", "=r,0,i" (i64 %"reg.7", i64 1)
  %"reg.9" = call i64 asm  "add $2, $0", "=r,0,i" (i64 %"reg.8", i64 1)
  %"reg.10" = call i64 asm  "add $2, $0", "=r,0,i" (i64 %"reg.9", i64 1)
  %"reg.11" = call i64 asm  "add $2, $0", "=r,0,i" (i64 %"reg.10", i64 1)
  %"reg.12" = call i64 asm  "add $2, $0", "=r,0,i" (i64 %"reg.11", i64 1)
  %"reg.13" = call i64 asm  "add $2, $0", "=r,0,i" (i64 %"reg.12", i64 1)
  %out.1 = call i64 asm  "add $2, $0", "=r,0,i" (i64 %"reg.13", i64 1)
  %"reg.14" = call i64 asm  "add $2, $0", "=r,0,i" (i64 %in.2, i64 1)
  %"reg.15" = call i64 asm  "add $2, $0", "=r,0,i" (i64 %"reg.14", i64 1)
  %"reg.16" = call i64 asm  "add $2, $0", "=r,0,i" (i64 %"reg.15", i64 1)
  %"reg.17" = call i64 asm  "add $2, $0", "=r,0,i" (i64 %"reg.16", i64 1)
  %"reg.18" = call i64 asm  "add $2, $0", "=r,0,i" (i64 %"reg.17", i64 1)
  %"reg.19" = call i64 asm  "add $2, $0", "=r,0,i" (i64 %"reg.18", i64 1)
  %"reg.20" = call i64 asm  "add $2, $0", "=r,0,i" (i64 %"reg.19", i64 1)
  %out.2 = call i64 asm  "add $2, $0", "=r,0,i" (i64 %"reg.20", i64 1)
  %"reg.21" = call i64 asm  "add $2, $0", "=r,0,i" (i64 %in.3, i64 1)
  %"reg.22" = call i64 asm  "add $2, $0", "=r,0,i" (i64 %"reg.21", i64 1)
  %"reg.23" = call i64 asm  "add $2, $0", "=r,0,i" (i64 %"reg.22", i64 1)
  %"reg.24" = call i64 asm  "add $2, $0", "=r,0,i" (i64 %"reg.23", i64 1)
  %"reg.25" = call i64 asm  "add $2, $0", "=r,0,i" (i64 %"reg.24", i64 1)
  %"reg.26" = call i64 asm  "add $2, $0", "=r,0,i" (i64 %"reg.25", i64 1)
  %"reg.27" = call i64 asm  "add $2, $0", "=r,0,i" (i64 %"reg.26", i64 1)
  %out.3 = call i64 asm  "add $2, $0", "=r,0,i" (i64 %"reg.27", i64 1)
  %"reg.28" = call i64 asm  "add $2, $0", "=r,0,i" (i64 %in.4, i64 1)
  %"reg.29" = call i64 asm  "add $2, $0", "=r,0,i" (i64 %"reg.28", i64 1)
  %"reg.30" = call i64 asm  "add $2, $0", "=r,0,i" (i64 %"reg.29", i64 1)
  %"reg.31" = call i64 asm  "add $2, $0", "=r,0,i" (i64 %"reg.30", i64 1)
  %"reg.32" = call i64 asm  "add $2, $0", "=r,0,i" (i64 %"reg.31", i64 1)
  %"reg.33" = call i64 asm  "add $2, $0", "=r,0,i" (i64 %"reg.32", i64 1)
  %out.4 = call i64 asm  "add $2, $0", "=r,0,i" (i64 %"reg.33", i64 1)
  %out.5 = call i64 asm  "add $2, $0", "=r,0,i" (i64 %in.5, i64 1)
  %out.6 = call i64 asm  "add $2, $0", "=r,0,i" (i64 %in.6, i64 1)
  %out.7 = call i64 asm  "add $2, $0", "=r,0,i" (i64 %in.7, i64 1)
  %"loop_counter.1" = add i64 %"loop_counter", 1
  %"loop_cond.1" = icmp slt i64 %"loop_counter.1", %"N"
  br i1 %"loop_cond.1", label %"loop", label %"end"

end:
  %"ret" = phi i64 [-1, %"entry"], [%"loop_counter", %"loop"]
  ret i64 %"ret"
}
