# 参考例（examples）

| ファイル | 内容 |
|---|---|
| [smolvla_libero_spatial_lora.ipynb](smolvla_libero_spatial_lora.ipynb) | SmolVLA を LIBERO-plus Spatial で LoRA 追加学習する Google Colab ノートブック |
| [smolvla_libero_spatial_lora_local.py](smolvla_libero_spatial_lora_local.py) | 上記ノートブックをローカル GPU 機（Colab 不使用）で再現するスクリプト |
| [smolvla_libero_plus_multisuite_lora_local.py](smolvla_libero_plus_multisuite_lora_local.py) | SmolVLA を LIBERO-plus Spatial・Object・Goal の全 40 タスクで LoRA 追加学習するローカル GPU 用スクリプト |

## smolvla_libero_spatial_lora.ipynb

`lerobot/smolvla_libero_plus` を初期重みとし、LIBERO-plus Spatial の 10 タスクを
LoRA で追加学習する。学習後は LoRA を元の重みへマージし、追加学習の前後を
同一条件で比較する。

### 使い方

1. Google Colab で開き、ランタイムのタイプを GPU（T4 で足りる）に変更する
2. 上から順に実行する。所要時間は T4 で数時間程度である
3. マージ済みモデル一式（zip）と、追加学習前後の成功率の比較（CSV）が出力される

学習条件は 10 タスク × 各 5 エピソード（計 50 エピソード）、3,000 steps、
バッチサイズ 1 で、Colab で完走することを優先した最小構成である。
性能を伸ばす場合はここを出発点に、自身の環境で条件を組み直すとよい。

### 提出物にするまでの作業

出力されるのは LeRobot 形式のモデル重みであり、これ単体では提出できない。
[submission_template/](../submission_template/) の `MyPolicy` にモデルを組み込み、
ポリシーサーバーの形にする。観測と action の仕様は
[submission_template/policy_server.py](../submission_template/policy_server.py)
の docstring にある。

推論は 1 リクエストあたり 10 秒以内に収める必要がある
（[ルートの README](../README.md#タイムアウト仕様)）。

### ノートブック内の評価と、本番の採点の違い

ノートブック内の評価は学習の効果を手早く確認するためのもので、採点とは条件が異なる。
出てくる成功率は本番スコアの目安にはならない。

| 項目 | ノートブック | 本番の採点 |
|---|---|---|
| 評価タスク | LIBERO-plus Spatial の 10 タスク | Track 1（`compe/t1/` のタスクセット） |
| 実行方法 | LeRobot の `lerobot-eval` | `python -m pipeline` + 提出したポリシーサーバー |
| 観測の解像度 | 256×256 | 128×128 |
| 1 タスクあたりの試行数 | 3（`EVAL_EPISODES_PER_TASK` で変更可） | 非公開（配布キットの既定は 20） |

試行数が 3 のままだと 1 エピソードの成否で成功率が約 33 ポイント動くため、
追加学習の前後を比べる場合は `EVAL_EPISODES_PER_TASK` を増やすこと。

### 実行環境

ノートブックの環境構築は Colab 向けで、[setup.sh](../setup.sh) とは独立している。
依存パッケージのバージョンが一致しない箇所があるため、評価と提出前チェックは
リポジトリ側の環境（`setup.sh` + `env.sh`）で行うこと。

ノートブックが利用する第三者製ソフトウェア・モデル・データセットのライセンスは、
各配布元の表記を参照すること。

## smolvla_libero_spatial_lora_local.py

`smolvla_libero_spatial_lora.ipynb` と同一の処理（初期重み・データセット・
LoRA 設定・学習ステップ数・評価タスク、すべて同一）を、Colab を使わず
ローカルの GPU 機で実行するための移植版である。処理内容に変更はなく、
Colab 固有の部分のみを次のように置き換えている。

- ランタイムの GPU チェック・`!pip` 実行 → 事前に用意した Python 3.12 の
  venv 上でスクリプトとして実行
- 最終セルの `google.colab.files.download` → ローカルファイルパスの表示のみ
  （ダウンロード不要なため）

### 使い方

```bash
# Python 3.12 の venv を用意し、torch / torchvision / torchcodec / mujoco を
# 個別にインストールしておく（理由は後述の「ノートブックとの差分」を参照）
python3.12 -m venv /content/step1_venv
source /content/step1_venv/bin/activate
pip install torch==2.9.1 torchvision==0.24.1 \
    --index-url https://download.pytorch.org/whl/cu126
pip install torchcodec==0.9.1 mujoco==3.7.0

python examples/smolvla_libero_spatial_lora_local.py
```

既定では Colab と同じく `/content/` 以下に一式を保存する（root 権限と
`apt-get` が必要。ノートブック同様、GPU 1 枚・空き容量 20GB 程度を要する）。

### ノートブックとの差分（ローカル実行で必要になった追加対応）

Colab はあらかじめ多くのパッケージが噛み合った状態でプリインストールされて
いるため、ノートブック自身はバージョンの組み合わせを一切気にしていない。
裸の venv でこの前提が崩れる箇所が 3 つあり、追加対応が必要だった。

1. **torch と torchcodec の組み合わせ**: `pip install -e lerobot[...]` は
   lerobot が許容する範囲（`torchcodec>=0.3,<0.12`）の中で最新版を選ぶだけで、
   実行中の torch の ABI と噛み合っているかは見ていない。torch を先に
   ピン留めしても、torchcodec 側が別 torch 系列向けにビルドされていると
   `undefined symbol` や `NotImplementedError` で落ちる。
   `torch==2.9.1+cu126` / `torchvision==0.24.1+cu126` / `torchcodec==0.9.1`
   の組み合わせで、lerobot が実際に使うファイルハンドル経由のデコード
   （`decoder_cache`）まで含めて動作確認済み。
2. **`future` パッケージ不足**: `bddl==1.0.1` のコードは
   `from future.utils import with_metaclass` を使うが、`future` はパッケージ
   メタデータ上の依存関係として宣言されていないため pip 経由では入らない。
   Colab には（レガシーな理由で）最初から入っているが、裸の venv には無い。
   明示的なインストールが必要。
3. **`IPython` 不在**: 比較表の表示に使う `IPython.display.display` は
   Colab 前提の呼び出しで、通常の venv には無い。スクリプトでは
   `print` にフォールバックしている。

これらはすべて「Colab の事前インストール状態への暗黙の依存」であり、
ノートブック自体のロジックの誤りではない。

### 実行結果（検証ログ）

A100 1 枚のローカル環境で、無改造のまま（途中で手を加えずに）完走することを
確認した（学習 3,000 steps 込みで約 20 分）。[smolvla_libero_spatial_lora_local_results.csv](smolvla_libero_spatial_lora_local_results.csv)
がこのときの生ログである。

| Task ID | Task | Base (%) | Spatial LoRA (%) | Δ (pp) |
|---|---|---|---|---|
| 0 | table center | 66.7 | 100.0 | +33.3 |
| 1 | next to the cookie box | 100.0 | 100.0 | 0.0 |
| 2 | next to the plate | 33.3 | 33.3 | 0.0 |
| 3 | next to the ramekin | 33.3 | 33.3 | 0.0 |
| 4 | on the cookie box | 33.3 | 66.7 | +33.3 |
| 5 | on the ramekin | 33.3 | 100.0 | +66.7 |
| 6 | on the stove | 66.7 | 100.0 | +33.3 |
| 7 | on the wooden cabinet | 100.0 | 100.0 | 0.0 |
| 8 | top drawer of the wooden cabinet | 100.0 | 100.0 | 0.0 |
| 9 | between the plate and the ramekin | 100.0 | 100.0 | 0.0 |
| **Overall** | LIBERO-Spatial | **66.7** | **83.3** | **+16.7** |

学習は `--seed=42`、評価は `--seed=2026` で固定しているが、同一スクリプトを
2 回実行しても成功率は一致しなかった（1 回目は Base 60.0% → Spatial LoRA
66.7%、+6.7pp）。学習 loss の推移は 2 回とも完全に一致していたため、原因は
学習側ではなく評価側（GPU カーネルの非決定性がロールアウトを通じて累積し、
境界線上のエピソードの成否を変え得ること）と、
[前述のとおり](#ノートブック内の評価と本番の採点の違い) 1 タスクあたり
3 エピソードしかないことの両方だと考えられる。ノートブック自体に埋め込まれて
いる Colab（Tesla T4）での実行結果（Base 76.7% → Spatial LoRA 83.3%、
+6.7pp）を含め、絶対値は実行ごとに変動するが、3 回とも追加学習後に成功率が
上回っており、改善の方向は一貫している。これが SmolVLA ベースライン + LoRA
学習パイプラインが正しく機能していることの確認となる。追加学習の効果を
より正確に見る場合は `EVAL_EPISODES_PER_TASK` を増やすこと。
