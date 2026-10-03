# ポリシーサーバー

提出物として採点環境で起動する HTTP サーバーです。`/health`、`/reset`、`/act` の 3 つのエンドポイントを持ち、
観測を受け取って 7 次元の行動を返します。サーバー部分は運営のテンプレートのままで、
`MyPolicy` クラスの中身だけを実装しています。

## バックエンド

`POLICY_BACKEND` で推論に使うモデルを切り替えます。

| 値 | 実装 | 用途 |
|---|---|---|
| `pi05`（既定） | [pi05_policy.py](pi05_policy.py) | openpi / JAX 版 pi0.5-LIBERO。**提出モデル** |
| `pi05_lerobot` | [pi05_lerobot_policy.py](pi05_lerobot_policy.py) | LeRobot / PyTorch 版 pi0.5。JAX 版との比較用 |
| 上記以外 | [policy_server.py](policy_server.py) 内 | SmolVLA（初期ベースライン） |

### pi0.5（JAX 版、提出モデル）

```bash
PI05_CHECKPOINT=/path/to/soup_checkpoint \
  python submission_template/policy_server.py --port 8000
```

| 環境変数 | 既定値 | 内容 |
|---|---|---|
| `PI05_CHECKPOINT` | `submission_template/pi05_weights` | チェックポイントのディレクトリ |
| `PI05_VARIANT` | 自動判定 | `base` または `lora`。`training_manifest.json` があれば LoRA と判定する |
| `PI05_ACTION_CHUNK` | `5` | 生成した 10 ステップのうち、再推論までに実行するステップ数 |
| `PI05_TEMPORAL_ENSEMBLE` | `0` | 時間方向のアンサンブル（実験用。提出モデルでは無効） |
| `PI05_WARMUP` | `1` | 起動時に JIT コンパイルを済ませる |

学習時と同じ前処理（画像の 180 度回転と padding 付きリサイズ、手先姿勢の axis-angle 変換）を推論時にも適用し、
出力した行動は [-1, 1] にクリップします。

### pi0.5（LeRobot / PyTorch 版、比較用）

```bash
POLICY_BACKEND=pi05_lerobot \
LEROBOT_PI05_CHECKPOINT=/path/to/pi05_libero_finetuned_v044 \
LEROBOT_PI05_DEVICE=cuda \
  python submission_template/policy_server.py --port 8000
```

- 画像の 180 度回転を標準で行います。回転しない場合、評価で 0/8 だったためです。比較するときは `LEROBOT_PI05_FLIP_IMAGES=0` を指定します。
- 1 回に実行するステップ数は `LEROBOT_PI05_ACTION_CHUNK` で変えられます（既定 5）。
- PyTorch 版に必要な Transformers の差分は [transformers_replace/](transformers_replace/) に同梱しており、
  初期化時に `transformers==4.53.2` に適用します。

### SmolVLA（初期ベースライン）

```bash
POLICY_BACKEND=smolvla \
SMOLVLA_CHECKPOINT=/path/to/smolvla_merged_model \
  python submission_template/policy_server.py --port 8000
```

採点環境（Python 3.10）で `lerobot[smolvla]==0.4.4` を pip で入れると、不要な `evdev` のビルドに失敗します。
そのため、LeRobot 0.4.4 のソースを `lerobot/` として提出物に同梱していました。

```bash
cp -a "$(python -c 'import pathlib, lerobot; print(pathlib.Path(lerobot.__file__).parent)')" \
  submission_template/lerobot
find submission_template/lerobot -type d -name __pycache__ -prune -exec rm -rf {} +
```

> **状態ベクトルの回転表現について**
> SmolVLA のチェックポイントは、手先の回転を Euler 角ではなく axis-angle で学習しています。
> 当初 Euler 角で実装していたため、追加学習なしでも衝突率が約 80% と異常に高くなっていました。
> 両者は次元も値の範囲も近く、正規化統計量を見ただけでは区別できません。
> LeRobot の `LiberoProcessorStep` の実装を確認して判明しました。

## 評価と提出前チェック

```bash
# リポジトリ直下で実行する
python -m pipeline --server-url http://localhost:8000 --track track1 --n-episodes 2 --max-steps 600

python validate_submission.py submission_template/   # 展開済みディレクトリを検査
python validate_submission.py submission.zip         # zip を検査（サーバーの起動確認まで行う）
```

モデルの重み（`pi05_weights/`、`pi05_lerobot_weights/`、`model_weights/`）と同梱用の `lerobot/` は
サイズが大きいため Git の管理対象外です。

採点環境の制約（1 リクエスト 10 秒以内、起動 120 秒以内、外部通信なしなど）は
[docs/competition_harness.md](../docs/competition_harness.md) を参照してください。
