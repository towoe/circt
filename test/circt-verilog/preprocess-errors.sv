// RUN: circt-verilog %s -E --verify-diagnostics
// REQUIRES: slang

// expected-error @below {{}}
`include "unknown.sv"
