# PARC 2026 予選 Track 1 — pi0.5-LIBERO LoRA による視覚摂動に頑健なマニピュレーション方策

[PARC 2026](https://github.com/matsuolab/PARC2026_pre) 予選 Track 1（LIBERO-plus の視覚摂動付き
マニピュレーションタスク）に取り組んだ個人プロジェクトです。
Physical Intelligence の **pi0.5-LIBERO** を LoRA で追加学習し、複数チェックポイントの
**weighted model soup** を単一方策として提出しました。

運営配布の評価ハーネス（[matsuolab/PARC2026_pre](https://github.com/matsuolab/PARC2026_pre)）を土台に、
データパイプライン・学習・推論サーバー・実験ツールを実装しています。

## 結果

公開 Track 1 の 4 タスク × 16 episode（ローカル評価、最大 600 steps）:

| タスク | 成功率 |
|---|---:|
| drawer bowl → plate | 93.8% |
| tomato sauce → basket（背景摂動が最も強い） | 56.3% |
| milk → basket（照明摂動） | 100% |
| bowl → stove（照明摂動） | 93.8% |
| **全体** | **85.9%** |

衝突率 3.1%（本プロジェクト中で最良）、平均 cartesian jerk 4.59。

検討過程での推移:

| 構成 | Track 1 成功率 |
|---|---:|
| SmolVLA（事前学習のみ） | 0/32 |
| SmolVLA LoRA + スクリプト制御のハイブリッド | 16/32（50%） |
| pi0.5-LIBERO（公式チェックポイント、追加学習なし） | 26/32（81.3%） |
| **pi0.5-LIBERO LoRA weighted soup（提出モデル）** | **85.9%**（16 ep/task） |

詳細は技術レポート [report/PARC2026_track1_report.tex](report/PARC2026_track1_report.tex) を参照してください。

## アプローチ

1. **モデル選定** — SmolVLA・TurboVLA・VLANeXt・GR00T などを比較し、全 suite を単一重みで扱え、
   異種データ co-training による汎化が期待できる pi0.5-LIBERO（JAX 版 openpi）を採用。
   PyTorch 版より JAX 版の方が Track 1 で高成功率だった。→ [docs/model_selection.md](docs/model_selection.md)
2. **データ変換** — `lerobot/libero_plus`（v3.0 スキーマ）を openpi が読める形式へ変換する
   2 段階パイプラインを実装。chunk 境界の index 処理や Arrow schema metadata 欠落のバグも修正。
3. **LoRA 学習** — VLM と action expert の LoRA 行列のみ学習（約 50M / 3.4B, 1.47%）。
   最も難しい tomato-sauce を重点的にサンプリング（他 60 ep に対し 350 ep、計 530 ep）。
4. **Model soup** — 同一レシピでも ±11pt のばらつきがあったため、step 700/750/775 の
   LoRA パラメータを 2:1:1 で平均し、単一チェックポイント依存を低減。
5. **推論** — 10 ステップの action chunk のうち先頭 5 ステップを実行する receding-horizon 方式。
   JIT を起動時に済ませ、各リクエストを 10 秒制限内に収める。

### うまくいかなかったこと

- **合成 augmentation**（暗色化・色調変換などテクスチャ摂動の模倣）: tomato-sauce 単体で
  43.8% → 6.2–31.2% に低下
- **40 タスクへの学習対象拡大**: LIBERO-plus validation は改善したが、公開 4 タスクの信号が希釈され
  本番スコアは低下（0.391 → 0.260–0.313）。augmentation を強めるほど jerk も増加
- **jerk 補助損失**: 成功率・衝突率は改善せず
- **時間方向のアンサンブル**: jerk は 4.59 → 3.47 に下がったが、成功率 85.9% → 81.2%、
  衝突率 3.1% → 14.1% に悪化

## リポジトリ構成

本プロジェクトで実装・変更した主な部分:

| パス | 内容 |
|---|---|
| [pi05_lora/](pi05_lora/) | データ変換パイプライン、LoRA 学習、augmentation／jerk 損失の実験、model soup ツール、診断スクリプト |
| [submission_template/pi05_policy.py](submission_template/pi05_policy.py) | openpi/JAX の pi0.5 推論アダプタ（提出モデル） |
| [submission_template/pi05_lerobot_policy.py](submission_template/pi05_lerobot_policy.py) | 比較用の LeRobot PyTorch 版 pi0.5 アダプタ |
| [submission_template/policy_server.py](submission_template/policy_server.py) | バックエンド切替（`POLICY_BACKEND`）付きのポリシーサーバー |
| [examples/](examples/) | SmolVLA LoRA 学習のローカル実行版（初期ベースライン） |
| [tests/](tests/) | アダプタ・学習設定の単体テスト |
| [report/](report/) | 技術レポート（LaTeX） |
| [docs/](docs/) | モデル選定調査、SmolVLA ベースラインの記録 |

運営配布の評価ハーネス（ほぼそのまま使用）:

| パス | 内容 |
|---|---|
| [pipeline/](pipeline/), [compe/t1/](compe/t1/) | Track 1 評価パイプラインとタスク定義 |
| [evaluate.py](evaluate.py), [validate_submission.py](validate_submission.py) | 提出 zip の評価・検証 |
| [setup.sh](setup.sh), [Dockerfile](Dockerfile) | 採点環境と同一の環境構築 |

配布ハーネスの使い方（提出形式、成功判定、タイムアウト仕様など）は
[docs/competition_harness.md](docs/competition_harness.md) にまとめています。

## 使い方

```bash
# 環境構築（採点環境と同一）
bash setup.sh
source env.sh

# pi0.5 ポリシーサーバーを起動（既定バックエンドは pi05 / JAX）
PI05_CHECKPOINT=/path/to/soup_checkpoint \
  python submission_template/policy_server.py --port 8000

# Track 1 評価
python -m pipeline --server-url http://localhost:8000 --track track1 \
  --n-episodes 16 --max-steps 600
```

学習・データ変換・model soup の手順は [pi05_lora/README.md](pi05_lora/README.md) を参照してください。
学習データとチェックポイントはサイズの都合でリポジトリには含めていません。

## 謝辞・ライセンス

- 評価ハーネスは松尾研究室による [PARC2026_pre](https://github.com/matsuolab/PARC2026_pre) の配布物に基づきます。
- ベースモデル: [openpi](https://github.com/Physical-Intelligence/openpi)（Apache-2.0）の pi0.5-LIBERO。
  PaliGemma/Gemma 由来の重みには Gemma Terms of Use が適用されます。
- 学習データ: [lerobot/libero_plus](https://huggingface.co/datasets/lerobot/libero_plus)
- 第三者ソフトウェアのライセンスは [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) を参照してください。
