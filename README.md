# MicroGPT

[![Elixir](https://img.shields.io/badge/elixir-%3E%3D1.19-purple.svg)](https://elixir-lang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A minimal, dependency-free GPT implementation in pure Elixir with automatic differentiation, inspired by [Andrej Karpathy's microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95).

**180 lines of typed, functional Elixir code** that trains a GPT model from scratch.

## Features

- 🎯 **Complete GPT-2 Architecture**: Multi-head attention, MLP, RMS normalization
- 🔄 **Automatic Differentiation**: Custom autograd engine with backpropagation
- 📐 **Full Type Specifications**: Every function is typed with `@spec`
- 🎨 **Functional Programming**: Immutable data structures, pure functions
- 📦 **Zero Dependencies**: Only Elixir standard library
- 🚀 **180 Lines**: Compact, readable implementation

## Installation

```elixir
def deps do
  [
    {:micro_gpt, github: "kentaro/microgpt_ex"}
  ]
end
```

## Quick Start

### Download Training Data

```bash
curl -L https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt -o input.txt
```

### Run Training

```elixir
# Full training (1000 steps, ~78 minutes)
MicroGPT.main()
```

### Expected Output

```
num docs: 32033
vocab size: 27
num params: 4192
step    1 / 1000 | loss 4.7292
step    2 / 1000 | loss 6.6752
step    3 / 1000 | loss 6.9993
...
step 1000 / 1000 | loss 2.5234

--- inference (new, hallucinated names) ---
sample  1: adelina
sample  2: kael
sample  3: emira
...
```

## Architecture

### Value Module (Autograd)

```elixir
defmodule MicroGPT.Value do
  @type t :: %__MODULE__{
    data: float(),
    grad: float(),
    children: list(t),
    local_grads: list(float()),
    id: reference()
  }

  @spec add(t, t | number()) :: t
  @spec mul(t, t | number()) :: t
  @spec backward(t, list(t)) :: list(t)
end
```

Supports operations: `add`, `mul`, `pow`, `log`, `exp`, `relu`, `div`, `sub`, `neg`

### GPT Model

- **Embedding Dimension**: 16
- **Attention Heads**: 4
- **Layers**: 1
- **Context Length**: 16 tokens
- **Vocabulary**: Character-level (27 tokens including BOS)

### Training

- **Optimizer**: Adam (β₁=0.85, β₂=0.99, ε=1e-8)
- **Learning Rate**: 0.01 with linear decay
- **Loss**: Cross-entropy per token
- **Dataset**: Names from [makemore](https://github.com/karpathy/makemore)

## Implementation Highlights

### Functional Backpropagation

```elixir
def backward(loss, params) do
  {_, topo} = build_topo(loss, MapSet.new(), [])
  grads = Enum.reduce(Enum.reverse(topo), %{loss.id => 1.0}, fn node, acc ->
    node_grad = Map.get(acc, node.id, 0.0)
    Enum.zip(node.children, node.local_grads)
    |> Enum.reduce(acc, fn {child, lg}, a ->
      Map.update(a, child.id, lg * node_grad, &(&1 + lg * node_grad))
    end)
  end)
  Enum.map(params, fn p -> %{p | grad: Map.get(grads, p.id, 0.0)} end)
end
```

**Key insight**: Each `Value` has a unique `reference()` ID. Gradients are accumulated in a map, then applied to parameters—maintaining immutability throughout.

### Multi-Head Attention

```elixir
for h <- 0..(n_head - 1) do
  hs = h * head_dim
  q_h = Enum.slice(q, hs, head_dim)
  k_h = Enum.map(keys, &Enum.slice(&1, hs, head_dim))
  v_h = Enum.map(values, &Enum.slice(&1, hs, head_dim))

  attn_logits = for t <- 0..(length(k_h) - 1) do
    dot_product(q_h, Enum.at(k_h, t)) / sqrt(head_dim)
  end

  attn_weights = softmax(attn_logits)
  weighted_sum(attn_weights, v_h)
end |> List.flatten()
```

## Performance

| Steps | Time |
|-------|------|
| 1 | ~4.7s |
| 10 | ~47s |
| 100 | ~7.8min |
| **1000** | **~78min** |

**Note**: This is a pure Elixir implementation without BLAS optimization. For production use, consider [Nx](https://github.com/elixir-nx/nx) + [EXLA](https://github.com/elixir-nx/nx/tree/main/exla).

## Why This Exists

1. **Educational**: Demonstrates that a complete GPT can be implemented in ~200 lines
2. **Functional**: Shows how to do deep learning in a purely functional style
3. **Typed**: Proves that type-safe ML code can be as concise as dynamic code
4. **Minimal**: No dependencies—understand every line

## Comparison with Python

| Feature | Python (microgpt.py) | Elixir (MicroGPT) |
|---------|---------------------|-------------------|
| Lines of Code | 199 | **180** |
| Type Annotations | None | **Full** |
| Dependencies | Python stdlib | Elixir stdlib |
| Mutation | Yes (obj.grad += ...) | **No (immutable)** |
| Style | Imperative | **Functional** |

## File Structure

```
lib/
├── micro_gpt.ex          # Main implementation (180 lines)
└── micro_gpt_test.ex     # Quick test (2 steps)

IMPLEMENTATION.md          # Technical deep dive
README.md                  # This file
input.txt                  # Training data (download separately)
```

## Testing

```bash
# Quick test (2 steps, ~10 seconds)
mix run -e "MicroGPTTest.test()"

# Full training (1000 steps, ~78 minutes)
mix run -e "MicroGPT.main()"
```

## Technical Details

### Autograd Engine

Each operation creates a new `Value` node with:
- `data`: Forward pass result
- `children`: Input values
- `local_grads`: Partial derivatives (∂output/∂input)
- `id`: Unique reference for tracking

Backward pass:
1. Build topological ordering of computation graph
2. Traverse in reverse, accumulating gradients via chain rule
3. Store gradients in `Map(id => grad)`
4. Return updated parameter structs

### Topological Sort

```elixir
defp build_topo(%Value{} = v, visited, topo) do
  if MapSet.member?(visited, v.id) do
    {visited, topo}
  else
    {visited, topo} = v.children
      |> Enum.reduce({MapSet.put(visited, v.id), topo},
                     fn c, {vis, top} -> build_topo(c, vis, top) end)
    {visited, topo ++ [v]}  # Append (not prepend!) for correct order
  end
end
```

**Critical**: Use `topo ++ [v]` (append) not `[v | topo]` (prepend) to get correct topological order: `[inputs, ..., loss]` not `[loss, ..., inputs]`.

## Limitations

- **Slow**: Pure Elixir without GPU/BLAS optimization
- **Small**: Only 4192 parameters (educational scale)
- **Single Layer**: Simplified architecture for clarity
- **Character-level**: Not subword tokenization

## Future Work

- [ ] Port to Nx/EXLA for GPU acceleration
- [ ] Add more layers and increase model size
- [ ] Implement BPE tokenization
- [ ] Multi-GPU training with distributed Elixir

## License

MIT License - see [LICENSE](LICENSE) file

## Credits

- Inspired by [Andrej Karpathy's microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)
- Training data from [makemore](https://github.com/karpathy/makemore)

## Contributing

Contributions welcome! Please:
1. Keep the implementation under 200 lines
2. Maintain full type specifications
3. Preserve functional programming style
4. Add tests for new features

## Author

[@kentaro](https://github.com/kentaro)

---

**"The most atomic way to train and inference a GPT in pure, dependency-free Elixir."** 🚀
