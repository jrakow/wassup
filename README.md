# wassup: A WebAssembly Superoptimizer

*wassup* is a superoptimizer for WebAssemby modules.
A superoptimizer finds not only an improved version of your program like a conventional optimizer would, but the optimal version.
Currenly, it can handle straight-line code, so no control flow, that only uses integers.

## Installation

This is assuming you have Cargo installed.

Clone this repository

```text
git clone https://github.com/jrakow/wassup
```

Install the command-line tool

```text
cargo install --path cli wassup-cli
```

## Example

To superoptimize the WebAssembly module `module.wasm` and save the optimized module in `module-opt.wasm`:

```text
wassup module.wasm -o module-opt.wasm
```

## Usage

```text
wassup 0.1.0
Julius Rakow <julius@rakow.me>
A WebAssembly Superoptimizer

USAGE:
    wassup [FLAGS] [OPTIONS] <INPUT>

FLAGS:
        --stdin      Read the input from stdin
    -h, --help       Prints help information
    -V, --version    Prints version information

OPTIONS:
        --opt-size-i32 <SIZE>         Size of a 32 bit integer in optimization [default: 4]
        --opt-size-i64 <SIZE>         Size of a 64 bit integer in optimization, 0 if i64 disabled [default: 8]
    -o, --output <FILE>               Where to write the result
        --timeout <TIMEOUT>           Timeout in milliseconds for the superoptimization of a single snippet
        --transval-size-i32 <SIZE>    Size of a 32 bit integer in translation validation [default: 6]
        --transval-size-i64 <SIZE>    Size of a 64 bit integer in translation validation, 0 if i64 disabled [default:
                                      12]

ARGS:
    <INPUT>    Input file to optimize
```

## Repository Organization

`encoding` contains the encoding of WebAssembly semantics into Z3.
This crate is the library for superoptimization and `cli` contains the command-line binary.

## License

This project is licensed under the MIT license.
