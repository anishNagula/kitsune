# Kitsune

A transformer inference runtime written in Rust.

Kitsune is an educational yet production-inspired AI systems project focused on understanding and implementing the complete inference stack behind modern Large Language Models.

Instead of wrapping existing inference engines, Kitsune aims to build the critical components from first principles:

- Tensor Runtime
- SIMD Optimizations
- GGUF Model Loading
- Tokenization
- Transformer Execution
- KV Cache Management
- Quantized Inference
- Benchmarking & Profiling

## Architecture

```text
GGUF Model
    ↓
Model Loader
    ↓
Tokenizer
    ↓
Tensor Runtime
    ↓
Transformer Layers
    ↓
KV Cache
    ↓
Sampler
    ↓
Generated Tokens
```

## Workspace Structure
```
kitsune/
│
├── crates/
│   ├── tensor/
│   ├── simd/
│   ├── gguf/
│   ├── tokenizer/
│   ├── runtime/
│   ├── transformer/
│   ├── kv_cache/
│   ├── sampler/
│   └── cli/
│
├── benches/
├── docs/
├── examples/
└── models/
```
