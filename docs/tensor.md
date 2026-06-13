# Tensor Runtime

Current Tensor Structure

```rust
pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}
```

## Design Decisions

Tensors are currently stored as contiguous dense arrays. To keep it Cache-friendly, Simple indexing, Easy SIMD optimization


Data is stored in row-major order.

### Implemented Features
- Tensor creation
- Shape check
- Flood Zeros
- Flood Ones
- Reshape
- Transpose
- Math ops (add, sub, mul)
