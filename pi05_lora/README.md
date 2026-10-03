# pi0.5-LIBERO の LoRA 学習

pi0.5-LIBERO の追加学習に使用したデータ変換、学習、model soup、実験・分析用のスクリプトをまとめています。

## ファイル構成

| 用途 | ファイル |
|---|---|
| データ変換 | `stage1_decode_libero_plus.py`, `stage2_build_openpi_dataset.py`, `merge_shard_datasets.py`, `extract_task_dataset.py`, `oversample_task.py`, `build_uniform_task_dataset.py`, `prepare_smoke_dataset.py` |
| 学習・チェックポイントの検証 | `train_pi05_lora.py`, `verify_checkpoint.py`, `paths.py` |
| Model soup（提出モデル） | `soup_checkpoints.py` |
| 実験（提出モデルでは不使用） | `texture_perturbation.py` + `augmented_data_config.py` (`--augment`), `standard_augmentation.py` + `standard_augmented_data_config.py` (`--standard-augment`), `jerk_loss.py` (`--jerk-loss-weight`) |
| 分析 | `diagnose_tomato_failure.py`, `scan_tomato_brightness.py` |

各スクリプトは同じディレクトリ内のモジュールを読み込むため、この配置を維持してください。
実行時は、`python pi05_lora/train_pi05_lora.py` のようにファイルのパスを指定します。

## データとチェックポイントの保存先

学習データとチェックポイントはリポジトリの外に保存します。
保存先は [`paths.py`](paths.py) で定義しており、環境変数で変更できます。

| 環境変数 | 既定値 | 内容 |
|---|---|---|
| `PARC_DATA_ROOT` | `/work/PARC2026_data` | LIBERO-plus の元データと openpi 形式のデータセット（配下の `lerobot/` に保存） |
| `PARC_TRAINING_ROOT` | `/work/PARC2026_training` | 学習チェックポイント（配下の `checkpoints/` に保存） |
| `OPENPI_ROOT` | `/tmp/openpi` | openpi のソースと専用の仮想環境 `.venv` |
| `OPENPI_DATA_HOME` | `/tmp/openpi-data` | ベースモデルの pi0.5-LIBERO チェックポイントを含む openpi のアセットキャッシュ |

各スクリプトのパス指定用引数（`--...`）でも、使用するファイルやディレクトリを指定できます。

## 学習と推論の動作確認

[ルート README](../README.md#使い方)に従って、openpi の環境を構築してください。
以下のコマンドはリポジトリ直下で実行します。`OPENPI_ROOT` が未設定または空の場合は、`/tmp/openpi` を使用します。
別の場所にある openpi を使う場合は、その絶対パスを `OPENPI_ROOT` に設定してください。

openpi の仮想環境と A100 を使用して、20 ステップの学習で動作を確認します。

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
"${OPENPI_ROOT:-/tmp/openpi}/.venv/bin/python" pi05_lora/train_pi05_lora.py \
  --steps 20 --batch-size 8 --save-interval 10 --overwrite
```

40 タスク全体のダウンロードが完了していない場合は、先頭から連続して取得できているデータから、
ファイルのコピーを伴わない小規模なデータセットを作成できます。このデータセットは学習の動作確認にのみ使用してください。

```bash
"${OPENPI_ROOT:-/tmp/openpi}/.venv/bin/python" pi05_lora/prepare_smoke_dataset.py
"${OPENPI_ROOT:-/tmp/openpi}/.venv/bin/python" pi05_lora/train_pi05_lora.py \
  --dataset-repo-id physical-intelligence/libero-smoke \
  --steps 20 --batch-size 8 --save-interval 10 --overwrite
```

作成された LoRA チェックポイントを読み込み、推論のコンパイルまで確認します。

```bash
"${OPENPI_ROOT:-/tmp/openpi}/.venv/bin/python" pi05_lora/verify_checkpoint.py
```

提出用の推論アダプタは、チェックポイントのディレクトリか、その親ディレクトリに
`training_manifest.json` がある場合、LoRA チェックポイントとして自動判定します。
`PI05_VARIANT=lora` を指定して明示的に選択することもできます。

### 学習設定

公式の pi0.5-LIBERO チェックポイントを初期値として、Gemma 2B の LoRA ランクを 16、
action expert の LoRA ランクを 32 に設定しています。
LoRA 以外のパラメータはすべて固定し、全約 34 億パラメータのうち、約 5,000 万（1.47%）を学習します。
EMA（重みの指数移動平均）は無効にしています。

## LIBERO-plus のデータ変換

`lerobot/libero_plus`（Hugging Face Hub、v3.0 スキーマ、20 fps）は、
openpi が使用する旧バージョンの LeRobot では直接読み込めません。
以下の 2 段階で、`image` と `wrist_image` をトップレベルのキーに持つ openpi 対応のデータセットに変換します。

1. `stage1_decode_libero_plus.py`（評価環境の仮想環境で実行）：対象エピソードをデコードし、エピソードごとの `.npz` ファイルに保存します。
2. `stage2_build_openpi_dataset.py`（openpi の仮想環境で実行）：`.npz` ファイルから、openpi 形式の LeRobotDataset（20 fps）を構築します。`--append` を指定すると、既存のデータセットに追加できます。

変換後のデータセットを加工するスクリプトも用意しています。

- `merge_shard_datasets.py`：個別に構築した分割データセットを、ファイル単位で 1 つに統合します。処理に時間がかかる LeRobotDataset のフレーム単位の API を介さずに統合できます。
- `extract_task_dataset.py`：指定したタスクの全エピソードを抽出し、単一タスクのデータセットを作成します。複数タスクの LoRA 学習による干渉を切り分けるために使用しました。
- `oversample_task.py`：指定したタスクのエピソードを、同じデータセット内でさらに N−1 回複製します。合成データを使わず、実データだけで学習バッチ内の対象タスクの比率を高めます。

Parquet データをコピーする上記 3 つのスクリプトでは、pandas の `read_parquet` / `to_parquet` ではなく、
`pyarrow` を直接使用しています。pandas 経由では、画像列の自動デコードに必要な
Parquet スキーマの `huggingface` メタデータが失われるためです。
このメタデータがないと、Hugging Face の `datasets` は画像をテンソルではなく `bytes` と `path` を持つ辞書として読み込み、
学習時に `hf_transform_to_torch` 内で `Could not infer dtype of dict` エラーが発生します。

## トマトソースのタスクに対するデータ拡張と分析

Track 1 のトマトソースのタスクは、試したすべての設定で、他の 3 タスクより成功率が低い状態でした。
`diagnose_tomato_failure.py` と `scan_tomato_brightness.py` で調べたところ、
評価環境の背景摂動では、PBR テクスチャの GLOSS チャネルをテーブルの色として使用しており、
シーンが極端に暗くなっていました（平均輝度は約 30/255）。
学習データで最も暗いエピソードでも平均輝度は約 45/255 で、評価環境に近い例は見つかりませんでした。

`texture_perturbation.py` の `TexturePerturbation` データクラスでは、この暗色化に加え、
LIBERO-plus のテクスチャアセットで確認した 2 種類の変化を模しています。
NRM チャネルによる青・紫の色調変化と、REFL / AO / DISP チャネルによる彩度低下です。
これらは `repack_transforms` を通じて学習時にのみ適用し、推論時には適用しません。

`train_pi05_lora.py` に `--augment` を指定すると、`augmented_data_config.py` を通じてこのデータ拡張が有効になります。
トマトソースのタスクに限って適用確率や暗色化の比率を高めるための引数
`--boosted-prob` と `--boosted-dark-weight` も用意しています。

### 実験結果

合成データによる拡張は、試したすべての強度で、実データのみの学習より成功率が低くなりました。
これは 4 タスクをまとめた学習と、トマトソースのみの学習の両方で確認しています。
トマトソースのみの評価では、データ拡張なしの成功率が 43.8%、中程度の拡張では 31.2%、
強い拡張では 6.2% でした（いずれも 32 エピソード）。

実験からは、4 タスクで LoRA の表現能力を共有することの影響が、より大きいと考えられました。
`oversample_task.py` でトマトソースの実エピソード数を 3 倍にすると、
Track 1 全体の成功率は 75.0% から 76.6%、トマトソースの成功率は 25.0% から 31.2% に改善しました
（各タスク 16 エピソード、合成データによる拡張なし）。
この設定は、提出候補 `pi05_submission_0811.zip` に使用しました。
