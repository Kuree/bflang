bflang - Brainfuck compiler in MLIR
-----------------------------------
`bflang` is a compiler that turns brainfuck code directly into a native executable.
The executable does not rely on any runtime other than `libc`.

Currently only `x86_64` on linux is supported.

# How to build
It should build with LLVM-18+. To install required LLVM toolchains, make sure
`llvm`, `mlir`, and `lld` are installed properly in your environment. On Ubuntu 20.04+,
you can install them via

```shell
sudo apt-get install -y libllvm18 llvm-18-dev mlir-18-tools libmlir-18 libmlir-18-dev liblld-18 liblld-18-dev
```

Once you have everything installed, run the following command:

```shell
mkdir build
cd build
cmake ../ -GNinja -DCMAKE_BUILD_TYPE=Release
ninja
```

Remove `-GNinja` if you prefer `make`. The executable `bflang` will be in `bin/` folder.

If you wish to run the regression tests, make sure to have `lit` installed. You can
install `lit` via `pip install lit`. Then you can specify the `lit` in the `cmake`
command via `-DLLVM_EXTERNAL_LIT=$(which lit)`. You can invoke the tests via `ninja check-bf`.

# Usage

```shell
bflang -O1 example.bf -o example
```

Or pipe it from `STDIN`:
```shell
echo "." | bfclang -O1 -o example
```

See `bflang -h` for more details.

# Optimizations
The input code is first parsed directly into a high-level dialect called `bf` where
each legal token in brainfuck is represented as an operation. Then the compiler
promotes the loop start `[` and end `]` into a region op, called `bf.loop`, which
makes the dialect lowering much easier.

The `bf` dialect is then lowered into standard dialects such as `arith`, `cf`, and
`llvm`. The data pointer is first represented as a global variable but then gets
promoted to a stack `alloca` before turning into virtual registers via the
`mem2reg` pass.

One thing to note that all the standard dialects are eventually lowered to `llvm`
dialect, and ultimately converted to `LLVM` IR. As a result, we do not want to
perform optimizations that can be done in LLVM automatically, such as removing
duplicated store after store or load after store. Instead, we focus on optimizations
that cannot be done in LLVM so far.

One optimization done here is the unmodified load. If the compiler can prove that
a particular cell is untouched, any load from that cell can be replaced with `0`.
