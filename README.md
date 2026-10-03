# PARC 2026 予選 Track 1：視覚摂動に頑健なロボット操作方策

[PARC 2026](https://github.com/matsuolab/PARC2026_pre) 予選 Track 1 に個人で取り組んだリポジトリです。
Track 1 は、背景テクスチャや照明に摂動を加えた LIBERO-plus のシミュレーション環境で、
ロボットアームに pick-and-place タスクを実行させる課題です。

Physical Intelligence が公開している **pi0.5-LIBERO** を LoRA で追加学習し、
複数のチェックポイントを重み付き平均した **weighted model soup** を、全タスク共通の単一方策として提出しました。

評価ハーネスは運営の配布物をそのまま使い、データ変換、学習、推論サーバー、
実験・分析用のスクリプトを自分で実装しています。

## 結果

公開されている Track 1 の 4 タスクで、各 16 エピソード（最大 600 ステップ）をローカル評価した結果です。

| タスク | 摂動 | 成功率 |
|---|---|---:|
| 棚の上段にある黒いボウルを皿に置く | 背景 | 93.8% |
| トマトソースをバスケットに入れる | 背景（最も強い） | 56.3% |
| 牛乳をバスケットに入れる | 照明 | 100% |
| ボウルをコンロに置く | 照明 | 93.8% |
| **全体** | | **85.9%** |

衝突率（操作対象以外の物体を動かした割合）は 3.1%、平均 jerk は 4.59 でした。
衝突したエピソードは失敗扱いになり、軌道の滑らかさもスコアに影響します。

### モデルの変遷

| 構成 | 成功率 |
|---|---:|
| SmolVLA（追加学習なし） | 0% |
| SmolVLA + LoRA に、ルールベースの制御を組み合わせたもの | 50% |
| pi0.5-LIBERO（追加学習なし） | 81.3% |
| **pi0.5-LIBERO + LoRA + model soup（提出モデル）** | **85.9%** |

上の 3 行は各 8 エピソード、提出モデルは各 16 エピソードでの評価です。

## アプローチ

1. **モデル選定**
   SmolVLA、TurboVLA、VLANeXt、GR00T などを比較しました。公開チェックポイントが 1 つで全タスクを扱えること、
   多様なデータで学習されていて汎化が期待できることから、pi0.5-LIBERO を選びました。
   PyTorch 版と JAX 版では JAX 版の成功率が高かったため、JAX 版（openpi）を使っています。
   詳細は [docs/model_selection.md](docs/model_selection.md) にあります。
2. **データ変換**
   Hugging Face 上の `lerobot/libero_plus` は openpi の学習コードでそのまま読めない形式のため、
   2 段階の変換パイプラインを実装しました。その過程で、chunk 境界でのインデックスのずれや、
   Parquet のメタデータが欠落する不具合も修正しています。
3. **LoRA 学習**
   ベースモデルの重みは固定し、LoRA の行列（全体の約 1.5%、約 5,000 万パラメータ）だけを学習しました。
   成功率が最も低かったトマトソースのタスクはデータを増やし、350 エピソードにしています（他のタスクは各 60、計 530）。
4. **Model soup**
   同じ設定で学習し直しても成功率が最大 11 ポイントほどぶれたため、step 700・750・775 の
   チェックポイントを 2:1:1 で平均し、1 つのチェックポイントの当たり外れに左右されにくくしました。
5. **推論**
   1 回の推論で 10 ステップ分の行動を生成し、先頭の 5 ステップを実行してから次の推論を行います。
   JIT コンパイルはサーバー起動時に済ませておき、1 リクエスト 10 秒の制限に収めています。

### 効果がなかった工夫

- **データ拡張**：評価環境の暗い背景を模した色調変換を加えましたが、トマトソースのタスクの成功率は
  43.8% から 6.2〜31.2% に下がりました。
- **学習タスクを 40 に増やす**：LIBERO-plus 全体での成功率は上がったものの、公開 4 タスクへの学習が薄まり、
  運営による評価スコアは 0.391 から 0.260〜0.313 に下がりました。データ拡張を強めるほど jerk も増えました。
- **jerk を抑える補助損失**：成功率も衝突率も改善しませんでした。
- **複数の推論結果を時間方向に平均する**：jerk は 4.59 から 3.47 に下がりましたが、
  成功率は 85.9% から 81.2% に下がり、衝突率は 3.1% から 14.1% に増えました。

## リポジトリ構成

自分で実装したもの：

| パス | 内容 |
|---|---|
| [pi05_lora/](pi05_lora/) | データ変換、LoRA 学習、model soup、各種実験と分析のスクリプト |
| [submission_template/](submission_template/) | 提出用のポリシーサーバーと pi0.5 推論アダプタ |
| [smolvla_baseline/](smolvla_baseline/) | 最初に試した SmolVLA の学習スクリプト |
| [tests/](tests/) | 推論アダプタと学習設定のテスト |
| [docs/](docs/) | モデル選定の調査メモ、SmolVLA ベースラインの記録 |

運営の配布物（ほぼ変更なし）：

| パス | 内容 |
|---|---|
| [pipeline/](pipeline/)、[compe/t1/](compe/t1/) | Track 1 の評価パイプラインとタスク定義 |
| [evaluate.py](evaluate.py)、[validate_submission.py](validate_submission.py) | 提出 zip の評価と事前チェック |
| [setup.sh](setup.sh)、[Dockerfile](Dockerfile) | 採点環境と同じ環境の構築 |

評価ハーネスの仕様（提出形式、成功判定、タイムアウト）は
[docs/competition_harness.md](docs/competition_harness.md) にまとめています。

## 使い方

```bash
# 環境構築（採点環境と同じ構成）
bash setup.sh
source env.sh

# ポリシーサーバーを起動（既定は pi0.5 / JAX）
PI05_CHECKPOINT=/path/to/soup_checkpoint \
  python submission_template/policy_server.py --port 8000

# 別のターミナルで Track 1 を評価
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
