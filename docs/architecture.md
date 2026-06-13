# Kitsune Architecture

Kitsune is designed as a layered inference runtime.

Each subsystem remains independently testable and benchmarkable.

## Layer 1: Tensor Runtime

Responsibilities:

- Tensor storage
- Shape management
- Mathematical operations
- Memory layout

Consumers:

- Transformer
- Quantization
- KV Cache

## Layer 2: Model Loading

Responsibilities:

- GGUF parsing
- Metadata extraction
- Weight loading

Consumers:

- Runtime
- Transformer

## Layer 3: Transformer Execution

Responsibilities:

- RMSNorm
- Attention
- Feed Forward Networks
- Residual Connections

Consumers:

- Runtime

## Layer 4: Generation

Responsibilities:

- Sampling
- Token streaming
- KV cache management

Consumers:

- CLI
