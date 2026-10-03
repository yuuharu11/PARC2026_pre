# PARC 2026 予選 Track 1：視覚摂動に頑健なロボット操作方策

[PARC 2026](https://github.com/matsuolab/PARC2026_pre) 予選 Track 1 への個人での取り組みをまとめたリポジトリです。
Track 1 では、背景テクスチャや照明を変化させた LIBERO-plus のシミュレーション環境で、
ロボットアームによる pick-and-place タスクの性能を評価します。

Physical Intelligence が公開している **pi0.5-LIBERO** を LoRA で追加学習し、
複数のチェックポイントの重みを加重平均する **weighted model soup** を用いて、全タスク共通の単一方策を作成・提出しました。

運営が配布した評価ハーネスを使用し、データ変換、学習、推論アダプタ、
実験・分析用のスクリプトを実装しました。

## 結果

Track 1 の公開 4 タスクについて、それぞれ 16 エピソード（各最大 600 ステップ）のローカル評価を行いました。

| タスク | 摂動 | 成功率 |
|---|---|---:|
| 棚の上段にある黒いボウルを皿に置く | 背景 | 93.8% |
| トマトソースをバスケットに入れる | 背景（最も強い） | 56.3% |
| 牛乳をバスケットに入れる | 照明 | 100% |
| ボウルをコンロに置く | 照明 | 93.8% |
| **全体** | | **85.9%** |

衝突率（操作対象以外の物体を動かしたエピソードの割合）は 3.1%、平均 jerk（加速度の変化率）は 4.59 でした。
評価では、衝突が発生したエピソードは失敗と判定され、軌道の滑らかさもスコアに反映されます。

### モデルの変遷

| 構成 | 成功率 |
|---|---:|
| SmolVLA（追加学習なし） | 0% |
| SmolVLA + LoRA + ルールベース制御 | 50% |
| pi0.5-LIBERO（追加学習なし） | 81.3% |
| **pi0.5-LIBERO + LoRA + model soup（提出モデル）** | **85.9%** |

提出モデル以外は各タスク 8 エピソード、提出モデルは各タスク 16 エピソードで評価しています。

## アプローチ

1. **モデル選定**
   SmolVLA、TurboVLA、VLANeXt、GR00T などを比較しました。1 つの公開チェックポイントで全タスクに対応でき、
   多様なデータでの事前学習による汎化性能が期待できることから、pi0.5-LIBERO を選びました。
   PyTorch 版と JAX 版では JAX 版の成功率が高かったため、JAX 版（openpi）を使っています。
   詳細は [docs/model_selection.md](docs/model_selection.md) にあります。
2. **データ変換**
   Hugging Face 上の `lerobot/libero_plus` は openpi の学習コードでは直接読み込めないため、
   2 段階の変換パイプラインを実装しました。その過程で、チャンク境界でのインデックスのずれや、
   Parquet のメタデータが欠落する不具合も修正しています。
3. **LoRA 学習**
   ベースモデルの重みを固定し、LoRA のパラメータ（全体の約 1.5%、約 5,000 万）だけを学習しました。
   成功率が最も低かったトマトソースのタスクには 350 エピソード、他の 3 タスクには各 60 エピソードを用い、
   合計 530 エピソードで学習しました。
4. **Model soup**
   同じ設定で学習を繰り返しても成功率に最大約 11 ポイントの差が出たため、学習ステップ 700・750・775 の
   チェックポイントの重みを 2:1:1 で加重平均し、単一のチェックポイントに依存することによる性能のばらつきを抑えました。
5. **推論**
   1 回の推論で 10 ステップ分の行動を生成し、先頭の 5 ステップを実行してから次の推論を行います。
   サーバー起動時に JIT コンパイルを行い、各リクエストの処理時間を 10 秒以内に収めています。

### 改善につながらなかった試み

- **データ拡張**：評価環境の暗い背景を模した色調変換を加えましたが、トマトソースのタスクの成功率は
  43.8% から 6.2〜31.2% に下がりました。
- **学習対象を 40 タスクに拡大**：LIBERO-plus 全体での成功率は上がったものの、公開 4 タスクの学習比率が下がり、
  運営による評価スコアは 0.391 から 0.260〜0.313 に下がりました。データ拡張を強めるほど jerk も増えました。
- **jerk を抑える補助損失**：成功率も衝突率も改善しませんでした。
- **複数の推論結果を時間方向に平均する**：jerk は 4.59 から 3.47 に下がりましたが、
  成功率は 85.9% から 81.2% に下がり、衝突率は 3.1% から 14.1% に増えました。

## リポジトリ構成

このプロジェクトで実装・作成したファイルを、以下のディレクトリにまとめています。

| パス | 内容 |
|---|---|
| [pi05_lora/](pi05_lora/) | データ変換、LoRA 学習、model soup、各種実験と分析のスクリプト |
| [submission_template/](submission_template/) | 提出用のポリシーサーバーと pi0.5 推論アダプタ |
| [smolvla_baseline/](smolvla_baseline/) | 最初に試した SmolVLA の学習スクリプト |
| [tests/](tests/) | 推論アダプタと学習設定のテスト |
| [docs/](docs/) | モデル選定の調査メモ、SmolVLA ベースラインの記録 |

以下は運営の配布物をほぼそのまま使用しています。

| パス | 内容 |
|---|---|
| [pipeline/](pipeline/)、[compe/t1/](compe/t1/) | Track 1 の評価パイプラインとタスク定義 |
| [evaluate.py](evaluate.py)、[validate_submission.py](validate_submission.py) | 提出 zip の評価と事前チェック |
| [setup.sh](setup.sh)、[Dockerfile](Dockerfile) | 採点環境と同じ環境の構築 |

評価ハーネスの仕様（提出形式、成功判定、タイムアウト）は、運営配布の README を保存した
[docs/competition_harness.md](docs/competition_harness.md) を参照してください。

## 使い方

評価環境と推論環境は別々に構築します。`setup.sh` は評価ハーネス用の環境を作成しますが、
既定の pi0.5 バックエンドに必要な openpi はインストールしません。

```bash
# 評価環境を構築（採点環境と同じ構成）
bash setup.sh
source env.sh
```

次に、[openpi の公式手順](https://github.com/Physical-Intelligence/openpi#installation)に従って推論環境を構築します。
Linux、対応する NVIDIA GPU、および `uv` が必要です。以下はリポジトリ直下で実行してください。

```bash
export OPENPI_ROOT="${OPENPI_ROOT:-/tmp/openpi}"
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git "$OPENPI_ROOT"
(
  cd "$OPENPI_ROOT"
  GIT_LFS_SKIP_SMUDGE=1 uv sync
  GIT_LFS_SKIP_SMUDGE=1 uv pip install --python "$OPENPI_ROOT/.venv/bin/python" -e .
)
uv pip install --python "$OPENPI_ROOT/.venv/bin/python" fastapi uvicorn msgpack

# 推論に必要なモジュールを読み込めることを確認
"$OPENPI_ROOT/.venv/bin/python" -c \
  'from openpi.policies import policy_config; from openpi_client import image_tools; import fastapi, uvicorn, msgpack'

# ポリシーサーバーを起動（既定は pi0.5 / JAX）
PI05_CHECKPOINT=/path/to/soup_checkpoint \
  "$OPENPI_ROOT/.venv/bin/python" submission_template/policy_server.py --port 8000
```

既に openpi を取得済みの場合は、`OPENPI_ROOT` にそのディレクトリの絶対パスを指定し、clone を省略してください。
`PI05_CHECKPOINT` には、モデルの重みと `assets/` を含むチェックポイントのディレクトリを指定します。

別のターミナルで評価環境を有効にして、Track 1 を評価します。

```bash
source env.sh
python -m pipeline --server-url http://localhost:8000 --track track1 \
  --n-episodes 16 --max-steps 600
```

学習、データ変換、model soup の手順は [pi05_lora/README.md](pi05_lora/README.md) を参照してください。
学習データとモデルの重みはサイズが大きいため、このリポジトリには含めていません。

## 謝辞・ライセンス

- 評価ハーネスは、松尾研究室が配布した [PARC2026_pre](https://github.com/matsuolab/PARC2026_pre) に基づいています。
- ベースモデルは [openpi](https://github.com/Physical-Intelligence/openpi)（Apache-2.0）の pi0.5-LIBERO です。
  PaliGemma／Gemma 由来の重みには Gemma Terms of Use も適用されます。
- 学習データには [lerobot/libero_plus](https://huggingface.co/datasets/lerobot/libero_plus) を使用しました。
- 第三者ソフトウェアのライセンスは [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) を参照してください。
