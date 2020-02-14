define <4 x double> @testv(i32**, i32) {

  %out = tail call <4 x double> asm "vaddpd $1, $2, $0", "=x,x,x,~{dirflag},~{fpsr},~{flags}"(<4 x double> <double 0.123, double 0.123, double 0.123, double 0.123>, <4 x double> <double 0.123, double 0.123, double 0.123, double 0.123>)
  ret <4 x double> %out
}

