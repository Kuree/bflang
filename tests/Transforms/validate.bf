// RUN: bf-opt --validate-bf --allow-unregistered-dialect --split-input-file --verify-diagnostics %s

func.func @valid() {
  "op.op"() : () -> ()
  bf.loop.start
  "op.op"() : () -> ()
  bf.loop.end
  "op.op"() : () -> ()
  bf.loop.start
  "op.op"() : () -> ()
  bf.loop.end
  return
}

// -----

func.func @valid_nested() {
  "op.op"() : () -> ()
  bf.loop.start
  "op.op"() : () -> ()
  bf.loop.start
  "op.op"() : () -> ()
  bf.loop.end
  "op.op"() : () -> ()
  bf.loop.end
  return
}


// -----

func.func @invalid() {
  "op.op"() : () -> ()
  // expected-error@below {{missing matching ']'}}
  bf.loop.start
  "op.op"() : () -> ()
  return
}

// -----

func.func @invalid_nested() {
  "op.op"() : () -> ()
  // expected-error@below {{missing matching ']'}}
  bf.loop.start
  "op.op"() : () -> ()
  bf.loop.start
  "op.op"() : () -> ()
  bf.loop.end
  "op.op"() : () -> ()
  return
}
