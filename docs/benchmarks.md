# Benchmarks

Machine:
- Apple Silicon Mac
- Release build

## Naive Matrix Multiplication

| Size | Time |
|--------|--------|
| 128x128 | ~1.98 ms |

Notes:

- Pure Rust implementation
- Row-major storage
- No SIMD
- No cache blocking
- No hardware acceleration
