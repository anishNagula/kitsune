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

# SIMD Benchmarks

Machine:
- Apple Silicon
- Release build

## 128x128 Matrix Multiplication

| Backend | Time |
|----------|----------|
| Scalar | ~1.97 ms |
| NEON | ~0.297 ms |

Speedup:

~6.6x

Notes:

- Row-major storage
- Matrix B transposed before multiplication
- NEON vectorized dot products
