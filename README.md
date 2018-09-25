# asmbench

A benchmark toolkit for assembly instructions using the LLVM JIT.

## Usage

To benchmark latency and throughput of a 64bit integer add use the following command:
```
python -m asmbench 'add {src:i64:r}, {srcdst:i64:r}'
```

To benchmark two instructions interleaved use this:
```
python -m asmbench 'add {src:i64:r}, {srcdst:i64:r}' 'sub {src:i64:r}, {srcdst:i64:r}'
```

To find out more add `-h` for help and `-v` for verbose mode.
