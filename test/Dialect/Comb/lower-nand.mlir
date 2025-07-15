// RUN: circt-opt %s | FileCheck %s

hw.module @and(in %a: i1, in %b: i1, out x: i1) {
  %0 = comb.and %a, %b : i1
  hw.output %0 : i1
}

hw.module @or(in %a: i1, in %b: i1, out x: i1) {
  %0 = comb.or %a, %b : i1
  hw.output %0 : i1
}

hw.module @xor(in %a: i1, in %b: i1, out x: i1) {
  %0 = comb.or %a, %b : i1
  hw.output %0 : i1
}
