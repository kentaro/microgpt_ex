# MicroGPT

[![Elixir](https://img.shields.io/badge/elixir-%3E%3D1.19-purple.svg)](https://elixir-lang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A minimal, dependency-free GPT implementation in pure Elixir with automatic differentiation, inspired by [Andrej Karpathy's microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95).

**200 lines of typed, functional Elixir code** that trains a GPT model from scratch.

## Features

- **Complete GPT-2 Architecture**: Multi-head attention, MLP, RMS normalization
- **Automatic Differentiation**: Custom autograd engine with backpropagation
- **Full Type Specifications**: Every function is typed with `@spec`
- **Functional Programming**: Immutable data structures, pure functions
- **Zero Dependencies**: Only Elixir standard library
- **Auto Dataset Download**: Downloads training data automatically on first run

## Quick Start

```bash
mix run
```

That's it. Training data is downloaded automatically, just like the Python version.

### Expected Output

```
Downloading dataset...
num docs: 32033
vocab size: 27
num params: 4192
step    1 / 1000 | loss 5.4328
step    2 / 1000 | loss 5.0459
...
step 1000 / 1000 | loss 2.3416

--- inference (new, hallucinated names) ---
sample  1: aemena
sample  2: kara
sample  3: aria
sample  4: amelem
...
```

### Public API

```elixir
# Run full training + inference (also called automatically by `mix run`)
MicroGPT.run()

# Generate samples from a trained model
MicroGPT.inference(uchars, bos, vocab_size, state_dict)
MicroGPT.inference(uchars, bos, vocab_size, state_dict, 10)  # custom sample count
```

## Architecture

### GPT Model

- **Embedding Dimension**: 16
- **Attention Heads**: 4
- **Layers**: 1
- **Context Length**: 16 tokens
- **Vocabulary**: Character-level (27 tokens including BOS)

### Training

- **Optimizer**: Adam (beta1=0.85, beta2=0.99, eps=1e-8)
- **Learning Rate**: 0.01 with linear decay
- **Loss**: Cross-entropy per token
- **Dataset**: Names from [makemore](https://github.com/karpathy/makemore)

### Value Module (Autograd)

Each `Value` node tracks forward computation and local gradients for backpropagation:

```elixir
%Value{data: float(), grad: float(), children: [Value.t()], local_grads: [float()], id: reference()}
```

Operations: `add`, `mul`, `pow`, `log`, `exp`, `relu`, `div`, `sub`, `neg`

**Key insight**: Each `Value` has a unique `reference()` ID. Gradients are accumulated in a map during backward pass, maintaining immutability throughout.

## Comparison with Python

| Feature | Python (microgpt.py) | Elixir (MicroGPT) |
|---------|---------------------|-------------------|
| Lines of Code | 199 | 200 |
| Type Annotations | None | **Full @spec** |
| Dependencies | Python stdlib | Elixir stdlib |
| Mutation | Yes (obj.grad += ...) | **No (immutable)** |
| Style | Imperative | **Functional** |
| Run | `python microgpt.py` | `mix run` |
| Final Loss | 2.65 | **2.34** |
| Min Loss | 1.58 | **1.52** |

## File Structure

```
lib/
├── micro_gpt.ex              # Main implementation (200 lines)
└── microgpt_ex/
    └── application.ex        # Auto-runs MicroGPT.run() on `mix run`
```

## Limitations

- **Slow**: Pure Elixir without GPU/BLAS optimization (~40 min for 1000 steps)
- **Small**: Only 4192 parameters (educational scale)
- **Single Layer**: Simplified architecture for clarity
- **Character-level**: Not subword tokenization

For production use, consider [Nx](https://github.com/elixir-nx/nx) + [EXLA](https://github.com/elixir-nx/nx/tree/main/exla).

## License

MIT License - see [LICENSE](LICENSE) file

## Credits

- Inspired by [Andrej Karpathy's microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)
- Training data from [makemore](https://github.com/karpathy/makemore)

## Author

[@kentaro](https://github.com/kentaro)
