defmodule MicroGPT do
  defmodule Value do
    @type t :: %__MODULE__{data: float(), grad: float(), children: list(t), local_grads: list(float()), id: reference()}
    defstruct data: 0.0, grad: 0.0, children: [], local_grads: [], id: nil
    @spec new(number(), list(t), list(float())) :: t
    def new(data, children \\ [], local_grads \\ []), do: %__MODULE__{data: data * 1.0, children: children, local_grads: local_grads, id: make_ref()}
    @spec add(t, t | number()) :: t
    def add(%__MODULE__{data: d1} = v1, %__MODULE__{data: d2} = v2), do: new(d1 + d2, [v1, v2], [1.0, 1.0])
    def add(v1, n) when is_number(n), do: add(v1, new(n))
    @spec mul(t, t | number()) :: t
    def mul(%__MODULE__{data: d1} = v1, %__MODULE__{data: d2} = v2), do: new(d1 * d2, [v1, v2], [d2, d1])
    def mul(v1, n) when is_number(n), do: mul(v1, new(n))
    @spec pow(t, number()) :: t
    def pow(%__MODULE__{data: d} = v, n), do: new(:math.pow(d, n), [v], [n * :math.pow(d, n - 1)])
    @spec log(t) :: t
    def log(%__MODULE__{data: d} = v), do: new(:math.log(d), [v], [1.0 / d])
    @spec exp(t) :: t
    def exp(%__MODULE__{data: d} = v), do: new(:math.exp(d), [v], [:math.exp(d)])
    @spec relu(t) :: t
    def relu(%__MODULE__{data: d} = v), do: new(max(0.0, d), [v], [if(d > 0, do: 1.0, else: 0.0)])
    @spec neg(t) :: t
    def neg(v), do: mul(v, -1)
    @spec sub(t, t | number()) :: t
    def sub(v1, v2), do: add(v1, neg(if is_number(v2), do: new(v2), else: v2))
    @spec div(t, t | number()) :: t
    def div(v1, v2), do: mul(v1, pow(if(is_number(v2), do: new(v2), else: v2), -1))
    @spec backward(t, list(t)) :: list(t)
    def backward(loss, params) do
      {_, topo} = build_topo(loss, MapSet.new(), [])
      grads = Enum.reduce(Enum.reverse(topo), %{loss.id => 1.0}, fn node, acc ->
        node_grad = Map.get(acc, node.id, 0.0)
        Enum.zip(node.children, node.local_grads) |> Enum.reduce(acc, fn {child, lg}, a ->
          Map.update(a, child.id, lg * node_grad, &(&1 + lg * node_grad))
        end)
      end)
      Enum.map(params, fn p -> %{p | grad: Map.get(grads, p.id, 0.0)} end)
    end
    defp build_topo(%__MODULE__{} = v, visited, topo) do
      if MapSet.member?(visited, v.id), do: {visited, topo}, else: v.children |> Enum.reduce({MapSet.put(visited, v.id), topo}, fn c, {vis, top} -> build_topo(c, vis, top) end) |> then(fn {vis, top} -> {vis, top ++ [v]} end)
    end
  end

  @type matrix :: list(list(Value.t()))
  @type state_dict :: %{String.t() => matrix}
  @type params :: list(Value.t())

  @spec main() :: :ok
  def main do
    :rand.seed(:exsss, {42, 42, 42})
    docs = load_dataset()
    {uchars, bos, vocab_size} = setup_tokenizer(docs)
    IO.puts("num docs: #{length(docs)}\nvocab size: #{vocab_size}")
    {state_dict, params} = init_params(vocab_size)
    IO.puts("num params: #{length(params)}")
    {state_dict, _params} = train(docs, uchars, bos, vocab_size, state_dict, params, 1000)
    inference(uchars, bos, vocab_size, state_dict, 20)
  end

  @spec load_dataset() :: list(String.t())
  defp load_dataset do
    unless File.exists?("input.txt"), do: raise("Please download: curl -L https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt -o input.txt")
    File.read!("input.txt") |> String.split("\n", trim: true) |> Enum.shuffle()
  end

  @spec setup_tokenizer(list(String.t())) :: {list(String.t()), integer(), integer()}
  defp setup_tokenizer(docs) do
    uchars = docs |> Enum.join() |> String.graphemes() |> Enum.uniq() |> Enum.sort()
    {uchars, length(uchars), length(uchars) + 1}
  end

  @spec init_params(integer()) :: {state_dict, params}
  defp init_params(vocab_size) do
    {n_embd, n_layer, block_size} = {16, 1, 16}
    matrix = fn nout, nin, std -> for _ <- 1..nout, do: for(_ <- 1..nin, do: Value.new(:rand.normal(0.0, std))) end
    state_dict = %{"wte" => matrix.(vocab_size, n_embd, 0.08), "wpe" => matrix.(block_size, n_embd, 0.08), "lm_head" => matrix.(vocab_size, n_embd, 0.08)}
    state_dict = Enum.reduce(0..(n_layer - 1), state_dict, fn i, sd ->
      Map.merge(sd, %{"layer#{i}.attn_wq" => matrix.(n_embd, n_embd, 0.08), "layer#{i}.attn_wk" => matrix.(n_embd, n_embd, 0.08), "layer#{i}.attn_wv" => matrix.(n_embd, n_embd, 0.08), "layer#{i}.attn_wo" => matrix.(n_embd, n_embd, 0.08), "layer#{i}.mlp_fc1" => matrix.(4 * n_embd, n_embd, 0.08), "layer#{i}.mlp_fc2" => matrix.(n_embd, 4 * n_embd, 0.08)})
    end)
    params = for mat <- Map.values(state_dict), row <- mat, p <- row, do: p
    {state_dict, params}
  end

  @spec linear(list(Value.t()), matrix) :: list(Value.t())
  defp linear(x, w), do: for(wo <- w, do: Enum.zip(wo, x) |> Enum.reduce(Value.new(0), fn {wi, xi}, acc -> Value.add(acc, Value.mul(wi, xi)) end))

  @spec softmax(list(Value.t())) :: list(Value.t())
  defp softmax(logits) do
    max_val = Enum.map(logits, & &1.data) |> Enum.max()
    exps = Enum.map(logits, &Value.exp(Value.sub(&1, max_val)))
    total = Enum.reduce(exps, Value.new(0), &Value.add/2)
    Enum.map(exps, &Value.div(&1, total))
  end

  @spec rmsnorm(list(Value.t())) :: list(Value.t())
  defp rmsnorm(x) do
    ms = Enum.reduce(x, Value.new(0), fn xi, acc -> Value.add(acc, Value.mul(xi, xi)) end) |> Value.div(length(x))
    scale = Value.pow(Value.add(ms, 1.0e-5), -0.5)
    Enum.map(x, &Value.mul(&1, scale))
  end

  @spec gpt(integer(), integer(), list(list(list(Value.t()))), list(list(list(Value.t()))), state_dict) :: {list(Value.t()), list(list(list(Value.t()))), list(list(list(Value.t())))}
  defp gpt(token_id, pos_id, keys, values, state_dict) do
    {_n_embd, n_head, n_layer, head_dim} = {16, 4, 1, 4}
    tok_emb = Enum.at(state_dict["wte"], token_id)
    pos_emb = Enum.at(state_dict["wpe"], pos_id)
    x = Enum.zip(tok_emb, pos_emb) |> Enum.map(fn {t, p} -> Value.add(t, p) end) |> rmsnorm()
    {x, {updated_keys, updated_values}} = Enum.reduce(0..(n_layer - 1), {x, {keys, values}}, fn li, {x, {ks, vs}} ->
      x_res = x
      x = rmsnorm(x)
      {q, k, v} = {linear(x, state_dict["layer#{li}.attn_wq"]), linear(x, state_dict["layer#{li}.attn_wk"]), linear(x, state_dict["layer#{li}.attn_wv"])}
      x_attn = for h <- 0..(n_head - 1) do
        hs = h * head_dim
        {q_h, k_h, v_h} = {Enum.slice(q, hs, head_dim), Enum.map(Enum.at(ks, li) ++ [k], &Enum.slice(&1, hs, head_dim)), Enum.map(Enum.at(vs, li) ++ [v], &Enum.slice(&1, hs, head_dim))}
        attn_logits = for t <- 0..(length(k_h) - 1), do: Enum.zip(q_h, Enum.at(k_h, t)) |> Enum.reduce(Value.new(0), fn {qi, ki}, acc -> Value.add(acc, Value.mul(qi, ki)) end) |> Value.div(:math.sqrt(head_dim))
        attn_weights = softmax(attn_logits)
        for j <- 0..(head_dim - 1), do: Enum.with_index(v_h) |> Enum.reduce(Value.new(0), fn {vt, t}, acc -> Value.add(acc, Value.mul(Enum.at(attn_weights, t), Enum.at(vt, j))) end)
      end |> List.flatten()
      x = linear(x_attn, state_dict["layer#{li}.attn_wo"]) |> Enum.zip(x_res) |> Enum.map(fn {a, b} -> Value.add(a, b) end)
      x_res = x
      x = rmsnorm(x) |> linear(state_dict["layer#{li}.mlp_fc1"]) |> Enum.map(&Value.relu/1) |> linear(state_dict["layer#{li}.mlp_fc2"]) |> Enum.zip(x_res) |> Enum.map(fn {a, b} -> Value.add(a, b) end)
      {x, {List.update_at(ks, li, &(&1 ++ [k])), List.update_at(vs, li, &(&1 ++ [v]))}}
    end)
    {linear(x, state_dict["lm_head"]), updated_keys, updated_values}
  end

  @spec train(list(String.t()), list(String.t()), integer(), integer(), state_dict, params, integer()) :: {state_dict, params}
  defp train(docs, uchars, bos, vocab_size, state_dict, params, num_steps) do
    {block_size, n_layer, lr, beta1, beta2, eps} = {16, 1, 0.01, 0.85, 0.99, 1.0e-8}
    {m, v} = {List.duplicate(0.0, length(params)), List.duplicate(0.0, length(params))}
    Enum.reduce(0..(num_steps - 1), {state_dict, params, m, v}, fn step, {sd, ps, m, v} ->
      doc = Enum.at(docs, rem(step, length(docs)))
      tokens = [bos] ++ (String.graphemes(doc) |> Enum.map(&Enum.find_index(uchars, fn c -> c == &1 end))) ++ [bos]
      n = min(block_size, length(tokens) - 1)
      {keys, values} = {List.duplicate([], n_layer), List.duplicate([], n_layer)}
      {losses, _, _} = Enum.reduce(0..(n - 1), {[], keys, values}, fn pos_id, {ls, ks, vs} ->
        {token_id, target_id} = {Enum.at(tokens, pos_id), Enum.at(tokens, pos_id + 1)}
        {logits, ks, vs} = gpt(token_id, pos_id, ks, vs, sd)
        probs = softmax(logits)
        loss_t = Value.neg(Value.log(Enum.at(probs, target_id)))
        {ls ++ [loss_t], ks, vs}
      end)
      loss = Enum.reduce(losses, Value.new(0), &Value.add/2) |> Value.div(n)
      ps = Value.backward(loss, ps)
      lr_t = lr * (1 - step / num_steps)
      updates = Enum.zip([m, v, ps]) |> Enum.map(fn {mi, vi, p} ->
        {m_new, v_new} = {beta1 * mi + (1 - beta1) * p.grad, beta2 * vi + (1 - beta2) * :math.pow(p.grad, 2)}
        {m_hat, v_hat} = {m_new / (1 - :math.pow(beta1, step + 1)), v_new / (1 - :math.pow(beta2, step + 1))}
        {m_new, v_new, %{p | data: p.data - lr_t * m_hat / (:math.pow(v_hat, 0.5) + eps), grad: 0.0}}
      end)
      {m, v, ps} = {Enum.map(updates, &elem(&1, 0)), Enum.map(updates, &elem(&1, 1)), Enum.map(updates, &elem(&1, 2))}
      sd = rebuild_state_dict(ps, vocab_size, n_layer)
      IO.puts("step #{String.pad_leading("#{step + 1}", 4)} / #{num_steps} | loss #{Float.round(loss.data, 4)}")
      {sd, ps, m, v}
    end) |> then(fn {sd, ps, _, _} -> {sd, ps} end)
  end

  defp rebuild_state_dict(params, vocab_size, n_layer) do
    {n_embd, block_size} = {16, 16}
    shapes = [
      {"wte", {vocab_size, n_embd}},
      {"wpe", {block_size, n_embd}},
      {"lm_head", {vocab_size, n_embd}}
    ] ++ (for i <- 0..(n_layer-1), key <- ["attn_wq", "attn_wk", "attn_wv", "attn_wo"], do: {"layer#{i}.#{key}", {n_embd, n_embd}}) ++ (for i <- 0..(n_layer-1), do: [{"layer#{i}.mlp_fc1", {4*n_embd, n_embd}}, {"layer#{i}.mlp_fc2", {n_embd, 4*n_embd}}]) |> List.flatten()
    {state_dict, _} = Enum.reduce(shapes, {%{}, params}, fn {name, {nout, nin}}, {sd, ps} ->
      {matrix_params, rest} = Enum.split(ps, nout * nin)
      matrix = Enum.chunk_every(matrix_params, nin)
      {Map.put(sd, name, matrix), rest}
    end)
    state_dict
  end

  @spec inference(list(String.t()), integer(), integer(), state_dict, integer()) :: :ok
  defp inference(uchars, bos, vocab_size, state_dict, num_samples) do
    IO.puts("\n--- inference (new, hallucinated names) ---")
    {block_size, n_layer, temperature} = {16, 1, 0.5}
    for sample_idx <- 0..(num_samples - 1) do
      {keys, values} = {List.duplicate([], n_layer), List.duplicate([], n_layer)}
      {sample, _, _, _} = Enum.reduce_while(0..(block_size - 1), {[], bos, keys, values}, fn pos_id, {samp, tok_id, ks, vs} ->
        {logits, ks, vs} = gpt(tok_id, pos_id, ks, vs, state_dict)
        probs = softmax(Enum.map(logits, &Value.div(&1, temperature)))
        weights = Enum.map(probs, & &1.data)
        token_id = weighted_choice(0..(vocab_size - 1) |> Enum.to_list(), weights)
        if token_id == bos, do: {:halt, {samp, token_id, ks, vs}}, else: {:cont, {samp ++ [Enum.at(uchars, token_id)], token_id, ks, vs}}
      end)
      IO.puts("sample #{String.pad_leading("#{sample_idx + 1}", 2)}: #{Enum.join(sample)}")
    end
  end

  @spec weighted_choice(list(integer()), list(float())) :: integer()
  defp weighted_choice(items, weights) do
    total = Enum.sum(weights)
    r = :rand.uniform() * total
    {item, _} = Enum.zip(items, weights) |> Enum.reduce_while({nil, 0.0}, fn {item, w}, {_, acc} -> if acc + w >= r, do: {:halt, {item, acc + w}}, else: {:cont, {item, acc + w}} end)
    item
  end
end
