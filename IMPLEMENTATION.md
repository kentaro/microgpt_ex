# MicroGPT Elixir Implementation

## 概要

Andrej Karpathyの`microgpt.py`（199行）を、Elixirで187行で完全に再実装しました。

## ファイル構成

- **`lib/micro_gpt.ex`** (187行): 完全な実装
  - モジュール名: `MicroGPT`
  - 外部依存なし（Elixir標準ライブラリのみ）

## 実装の特徴

### 1. 型システムの活用

すべての主要関数に`@spec`型注釈を付与：

```elixir
@spec add(t, t | number()) :: t
@spec backward(t, list(t)) :: list(t)
@spec linear(list(Value.t()), matrix) :: list(Value.t())
```

### 2. 関数型プログラミング

- **不変データ構造**: すべてのValueは不変
- **純粋関数**: 副作用なし（乱数とIO以外）
- **値の受け渡し**: `backward/2`は更新されたparamsを返す

### 3. 自動微分（Autograd）

```elixir
defmodule Value do
  @type t :: %__MODULE__{
    data: float(),
    grad: float(),
    children: list(t),
    local_grads: list(float()),
    id: reference()  # 一意識別子
  }
end
```

- 各Valueに一意の`id` (reference)を付与
- 計算グラフを構築し、トポロジカルソートで逆伝播
- 勾配は`Map(id => grad)`で管理し、最後にparamsに適用

### 4. GPTアーキテクチャ

- **Token + Position Embeddings**
- **Multi-Head Attention**: 4ヘッド、head_dim=4
- **MLP with ReLU**: 4x拡張
- **RMS Normalization**: LayerNormの簡略版
- **Residual Connections**

### 5. Adamオプティマイザ

- 第1モーメント (m) と第2モーメント (v) のバッファ
- バイアス補正
- 線形学習率減衰

## 実行方法

```bash
# コンパイル
mix compile

# トレーニング実行（1000ステップ）
mix run -e "MicroGPT.main()"
```

## 期待される出力

```
num docs: 32032
vocab size: 27
num params: 2080
step    1 / 1000 | loss 3.xxxx
step    2 / 1000 | loss 3.xxxx
...
step 1000 / 1000 | loss 2.xxxx

--- inference (new, hallucinated names) ---
sample  1: adelina
sample  2: kael
...
```

## パフォーマンスについて

純粋Elixir実装のため、最適化されたBLASライブラリを使用するPython版と比べて**非常に遅い**です。
これは教育目的の実装であり、以下を重視しています：

1. **正確性**: すべてのアルゴリズムを忠実に再現
2. **可読性**: 関数型スタイルでクリーンな実装
3. **型安全性**: 完全な型注釈

実用的なGPTトレーニングには、NxやExlaなどの数値計算ライブラリの使用を推奨します。

## 技術的なハイライト

### 問題: トポロジカルソートの順序

初期実装では`[v | topo]`（前置）を使用していたため、順序が逆になっていました。
解決策: `topo ++ [v]`（後置）に変更し、正しい順序を実現。

### 問題: 勾配の蓄積

Elixirは不変なので、Pythonのように`p.grad += ...`ができません。
解決策: `Map(id => grad)`で勾配を管理し、最後に新しいValue構造体を生成。

```elixir
def backward(loss, params) do
  grads = # ... 勾配を計算してマップに蓄積
  Enum.map(params, fn p -> %{p | grad: Map.get(grads, p.id, 0.0)} end)
end
```

## 行数の内訳

- Value定義 & 演算子: 30行
- モデル関数 (linear, softmax, rmsnorm, gpt): 45行
- トレーニングループ: 30行
- 推論: 20行
- データセット & 初期化: 25行
- ヘルパー関数: 15行
- 型定義 & ドキュメント: 22行

**合計: 187行** ✅（200行以内）

## まとめ

✅ microgpt.pyの全機能を実装
✅ 型システムを完全活用
✅ 関数型プログラミングの原則に従う
✅ 200行以内（187行）
✅ 外部依存なし
✅ 正しい勾配計算を検証済み
