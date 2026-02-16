defmodule MicroGPTTest do
  @moduledoc "Quick test version with reduced parameters"

  def test do
    Code.eval_file("lib/micro_gpt.ex")

    # Override to run just 2 steps
    IO.puts("\n=== MicroGPT Quick Test (2 training steps) ===\n")

    :rand.seed(:exsss, {42, 42, 42})

    # Load a small subset of data
    docs = File.read!("input.txt")
           |> String.split("\n", trim: true)
           |> Enum.take(50)
           |> Enum.shuffle()

    # Setup tokenizer
    uchars = docs
             |> Enum.join()
             |> String.graphemes()
             |> Enum.uniq()
             |> Enum.sort()
    bos = length(uchars)
    vocab_size = bos + 1

    IO.puts("num docs: #{length(docs)}")
    IO.puts("vocab size: #{vocab_size}")

    # Initialize with same params as original
    n_embd = 16
    n_layer = 1
    block_size = 16

    matrix = fn nout, nin, std ->
      for _ <- 1..nout, do: for(_ <- 1..nin, do: MicroGPT.Value.new(:rand.normal(0.0, std)))
    end

    state_dict = %{
      "wte" => matrix.(vocab_size, n_embd, 0.08),
      "wpe" => matrix.(block_size, n_embd, 0.08),
      "lm_head" => matrix.(vocab_size, n_embd, 0.08)
    }

    state_dict = Enum.reduce(0..(n_layer - 1), state_dict, fn i, sd ->
      Map.merge(sd, %{
        "layer#{i}.attn_wq" => matrix.(n_embd, n_embd, 0.08),
        "layer#{i}.attn_wk" => matrix.(n_embd, n_embd, 0.08),
        "layer#{i}.attn_wv" => matrix.(n_embd, n_embd, 0.08),
        "layer#{i}.attn_wo" => matrix.(n_embd, n_embd, 0.08),
        "layer#{i}.mlp_fc1" => matrix.(4 * n_embd, n_embd, 0.08),
        "layer#{i}.mlp_fc2" => matrix.(n_embd, 4 * n_embd, 0.08)
      })
    end)

    params = for mat <- Map.values(state_dict), row <- mat, p <- row, do: p
    IO.puts("num params: #{length(params)}")

    # Train for just 2 steps
    IO.puts("\nTraining 2 steps...")
    train_minimal(docs, uchars, bos, state_dict, params, 2)

    IO.puts("\n✅ Test completed successfully!")
  end

  defp train_minimal(docs, uchars, bos, state_dict, params, num_steps) do
    {block_size, n_layer, lr, beta1, beta2, eps} = {16, 1, 0.01, 0.85, 0.99, 1.0e-8}
    {m, v} = {List.duplicate(0.0, length(params)), List.duplicate(0.0, length(params))}

    Enum.reduce(0..(num_steps - 1), {state_dict, params, m, v}, fn step, {sd, ps, m, v} ->
      doc = Enum.at(docs, rem(step, length(docs)))
      tokens = [bos] ++ (String.graphemes(doc) |> Enum.map(&Enum.find_index(uchars, fn c -> c == &1 end))) ++ [bos]
      n = min(block_size, length(tokens) - 1)

      {keys, values} = {List.duplicate([], n_layer), List.duplicate([], n_layer)}

      # Forward pass
      {losses, _, _} = Enum.reduce(0..(n - 1), {[], keys, values}, fn pos_id, {ls, ks, vs} ->
        {token_id, target_id} = {Enum.at(tokens, pos_id), Enum.at(tokens, pos_id + 1)}
        logits = gpt_forward(token_id, pos_id, ks, vs, sd)
        probs = softmax(logits)
        loss_t = MicroGPT.Value.neg(MicroGPT.Value.log(Enum.at(probs, target_id)))
        {ls ++ [loss_t], ks, vs}
      end)

      loss = Enum.reduce(losses, MicroGPT.Value.new(0), &MicroGPT.Value.add/2) |> MicroGPT.Value.div(n)

      # Backward pass
      ps = MicroGPT.Value.backward(loss, ps)

      # Adam update
      lr_t = lr * (1 - step / num_steps)
      updates = Enum.zip([m, v, ps]) |> Enum.map(fn {mi, vi, p} ->
        {m_new, v_new} = {beta1 * mi + (1 - beta1) * p.grad, beta2 * vi + (1 - beta2) * :math.pow(p.grad, 2)}
        {m_hat, v_hat} = {m_new / (1 - :math.pow(beta1, step + 1)), v_new / (1 - :math.pow(beta2, step + 1))}
        {m_new, v_new, %{p | data: p.data - lr_t * m_hat / (:math.pow(v_hat, 0.5) + eps), grad: 0.0}}
      end)

      {m, v, ps} = {Enum.map(updates, &elem(&1, 0)), Enum.map(updates, &elem(&1, 1)), Enum.map(updates, &elem(&1, 2))}

      IO.puts("step #{String.pad_leading("#{step + 1}", 4)} / #{num_steps} | loss #{Float.round(loss.data, 4)}")

      {sd, ps, m, v}
    end)
  end

  defp gpt_forward(token_id, pos_id, keys, values, state_dict) do
    {n_embd, n_head, n_layer, head_dim} = {16, 4, 1, 4}
    tok_emb = Enum.at(state_dict["wte"], token_id)
    pos_emb = Enum.at(state_dict["wpe"], pos_id)
    x = Enum.zip(tok_emb, pos_emb) |> Enum.map(fn {t, p} -> MicroGPT.Value.add(t, p) end) |> rmsnorm()

    {x, _} = Enum.reduce(0..(n_layer - 1), {x, {keys, values}}, fn li, {x, {ks, vs}} ->
      x_res = x
      x = rmsnorm(x)
      {q, k, v} = {linear(x, state_dict["layer#{li}.attn_wq"]), linear(x, state_dict["layer#{li}.attn_wk"]), linear(x, state_dict["layer#{li}.attn_wv"])}

      x_attn = for h <- 0..(n_head - 1) do
        hs = h * head_dim
        {q_h, k_h, v_h} = {Enum.slice(q, hs, head_dim), Enum.map(Enum.at(ks, li) ++ [k], &Enum.slice(&1, hs, head_dim)), Enum.map(Enum.at(vs, li) ++ [v], &Enum.slice(&1, hs, head_dim))}
        attn_logits = for t <- 0..(length(k_h) - 1), do: Enum.zip(q_h, Enum.at(k_h, t)) |> Enum.reduce(MicroGPT.Value.new(0), fn {qi, ki}, acc -> MicroGPT.Value.add(acc, MicroGPT.Value.mul(qi, ki)) end) |> MicroGPT.Value.div(:math.sqrt(head_dim))
        attn_weights = softmax(attn_logits)
        for j <- 0..(head_dim - 1), do: Enum.with_index(v_h) |> Enum.reduce(MicroGPT.Value.new(0), fn {vt, t}, acc -> MicroGPT.Value.add(acc, MicroGPT.Value.mul(Enum.at(attn_weights, t), Enum.at(vt, j))) end)
      end |> List.flatten()

      x = linear(x_attn, state_dict["layer#{li}.attn_wo"]) |> Enum.zip(x_res) |> Enum.map(fn {a, b} -> MicroGPT.Value.add(a, b) end)
      x_res = x
      x = rmsnorm(x) |> linear(state_dict["layer#{li}.mlp_fc1"]) |> Enum.map(&MicroGPT.Value.relu/1) |> linear(state_dict["layer#{li}.mlp_fc2"]) |> Enum.zip(x_res) |> Enum.map(fn {a, b} -> MicroGPT.Value.add(a, b) end)
      {x, {List.update_at(ks, li, &(&1 ++ [k])), List.update_at(vs, li, &(&1 ++ [v]))}}
    end)

    linear(x, state_dict["lm_head"])
  end

  defp linear(x, w), do: for(wo <- w, do: Enum.zip(wo, x) |> Enum.reduce(MicroGPT.Value.new(0), fn {wi, xi}, acc -> MicroGPT.Value.add(acc, MicroGPT.Value.mul(wi, xi)) end))

  defp softmax(logits) do
    max_val = Enum.map(logits, & &1.data) |> Enum.max()
    exps = Enum.map(logits, &MicroGPT.Value.exp(MicroGPT.Value.sub(&1, max_val)))
    total = Enum.reduce(exps, MicroGPT.Value.new(0), &MicroGPT.Value.add/2)
    Enum.map(exps, &MicroGPT.Value.div(&1, total))
  end

  defp rmsnorm(x) do
    ms = Enum.reduce(x, MicroGPT.Value.new(0), fn xi, acc -> MicroGPT.Value.add(acc, MicroGPT.Value.mul(xi, xi)) end) |> MicroGPT.Value.div(length(x))
    scale = MicroGPT.Value.pow(MicroGPT.Value.add(ms, 1.0e-5), -0.5)
    Enum.map(x, &MicroGPT.Value.mul(&1, scale))
  end
end
