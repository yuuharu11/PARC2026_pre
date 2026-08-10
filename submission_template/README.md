# 提出物テンプレート

`policy_server.py` には `lerobot/smolvla_libero_plus` ベースラインが実装済みである。
LoRAをマージしたモデル一式を `model_weights/` に置くと自動的にそちらを使う。
ローカル比較ではコピーせず、次のようにチェックポイントを指定できる。

```bash
SMOLVLA_CHECKPOINT=/content/smolvla_libero_plus_multisuite_lora_merged \
  python policy_server.py --port 8000
```

デバイスは自動選択される。必要なら `SMOLVLA_DEVICE=cuda` または `cpu` で固定できる。
採点環境のネットワークに依存しないよう、最終提出には重みを同梱すること。

## LeRobot版pi0.5-LIBERO

LeRobot 0.4.4のPyTorch版pi0.5を比較する場合は、LIBERO学習済みcheckpointを指定して
専用backendを選ぶ。

```bash
POLICY_BACKEND=pi05_lerobot \
LEROBOT_PI05_CHECKPOINT=/work/PARC2026_models/pi05_libero_finetuned_v044 \
LEROBOT_PI05_DEVICE=cuda \
python policy_server.py --port 8000
```

このbackendは公式LeRobot LIBERO評価と同じ画像180度回転、axis-angle state変換、
保存済みnormalizerを使用する。既定では生成chunkから5 actionずつ実行する。変更する場合は
`LEROBOT_PI05_ACTION_CHUNK`を指定する。最終提出ではcheckpointを
`pi05_lerobot_weights/`へ配置し、tokenizerも同ディレクトリの`tokenizer/`へ同梱する。
PyTorch版pi0.5が必要とするopenpiのTransformers互換差分は`transformers_replace/`に同梱し、
backend初期化時に`transformers==4.53.2`へ適用する。

Track 1の基準評価は、サーバーを起動した後にリポジトリ直下から次を実行する。

```bash
python -m pipeline --server-url http://localhost:8000 --track track1 \
  --n-episodes 8 --max-steps 300 --output-dir results/pi05_lerobot_track1_8ep
```

採点環境は Python 3.10 であり、`lerobot[smolvla]==0.4.4` を通常インストールすると
不要な hardware 依存の `pynput -> evdev` が入り、`Python.h` 不在でビルドに失敗する。
そのため、動作確認済み環境の LeRobot 0.4.4 パッケージを提出物へソース同梱する。

```bash
cp -a "$(python -c 'import pathlib, lerobot; print(pathlib.Path(lerobot.__file__).parent)')" \
  submission_template/lerobot
find submission_template/lerobot -type d -name __pycache__ -prune -exec rm -rf {} +
```

`requirements.txt` はSmolVLA推論に必要な依存のみを列挙しており、`evdev`を導入しない。

## ディレクトリ構成

```
submission.zip
├── policy_server.py     # ← MyPolicy クラスを編集する（必須）
├── requirements.txt     # ← 追加依存があれば記載（必須）
├── lerobot/             # ← 推論用LeRobot 0.4.4ソース（K8s互換）
└── model_weights/       # ← チェックポイント等を配置（任意）
```

## 手順

1. `policy_server.py` の `MyPolicy` クラスにモデルのロードと推論を実装する
2. モデル重みを `model_weights/` に配置する
3. 追加ライブラリがあれば `requirements.txt` に追記する
4. zip にまとめて提出する:
   ```bash
   zip -r submission.zip policy_server.py requirements.txt lerobot/ model_weights/
   ```

## ローカルテスト

```bash
# サーバー起動
pip install -r requirements.txt
python policy_server.py

# 別ターミナルで評価（Track 1 の example タスクで疎通確認）
python -m pipeline --server-url http://localhost:8000 --track track1 --n-episodes 2 --max-steps 10
```

`--track track1` を指定すると Track 1 の example タスクが実行される。
配布環境のセットアップおよび評価コマンドの詳細な使用方法は、リポジトリ直下の
[README.md](../README.md) を参照すること。

## 提出前セルフチェック（推奨）

提出から採点までは 1 回あたり 15〜20 分を要する。フル採点を待たずにローカルで問題を
検出できるよう、リポジトリ直下に提出物チェックスクリプト（[validate_submission.py](../validate_submission.py)）
を用意している。**提出前に必ず実行すること。**

```bash
# いずれもリポジトリ直下で実行する
# zip を丸ごと検査（静的チェック + サーバーを起動して I/O スモークテストまで）
python validate_submission.py submission.zip

# 展開済みディレクトリでも可（このテンプレートを指定する）
python validate_submission.py submission_template/

# サーバーを起動せず静的チェックのみ（高速）
python validate_submission.py submission.zip --static

# 依存を入れてから動的チェック（クリーンな環境での再現確認）
python validate_submission.py submission.zip --install
```

検査内容: zip 健全性 / Zip Slip・zip bomb / サイズ上限（20GB）/ 必須ファイル /
`policy_server.py` の構文・エンドポイント / `requirements.txt` の構文・外部ソース
禁止（`git+`・`--index-url` 等は不可）/ サーバー起動 → `/health`→`/reset`→`/act`
において action が float32 shape (7,) であること、NaN/Inf を含まないこと、レイテンシまでを確認する。

`PASS` かつ ERROR 0 件であれば提出可能である（WARN は推奨事項である）。採点環境でも
unzip 直後に同一の検査を実施する。

## 注意事項

- `policy_server.py` のサーバー部分（FastAPI エンドポイント、シリアライゼーション）は変更しないこと
- `requirements.txt` に `git+https://…` や `--index-url` 等の外部ソース指定は使用できない（採点環境は外部通信を遮断する）
- `get_action()` は 10 秒以内に応答すること。1 リクエストでも 10 秒を超過すると
  そのトラック全体が 0 点となる（詳細はリポジトリ直下の [README.md](../README.md) の
  「タイムアウト仕様」を参照すること）
- `reset()` はエピソードごとに呼び出される。内部状態（action chunking のキャッシュ等）をクリアすること

> **SmolVLA 実装上の注意:** pretrained checkpoint の config.json は state shape を (6,) と
> 記載しているが、実際に保存されている正規化統計量（policy_preprocessor の
> normalizer safetensors）は (8,) である（config.json の記載は誤り、または stale）。
> `eef_pos(3) + [回転](3) + gripper_qpos(2)` という構成自体は統計量のレンジから
> 妥当だが、回転成分を当初 Euler 角（`scipy.spatial.transform.Rotation.as_euler`）と
> 誤って実装していた。lerobot 本体の `lerobot/processor/env_processor.py` の
> `LiberoProcessorStep`（このチェックポイントの学習・`lerobot-eval` 評価が実際に
> 使っている変換）を確認したところ、四元数を **axis-angle**（回転軸×回転角）に
> 変換していることが判明した。Euler角と axis-angle はどちらも3次元で値域も
> 似ているため、統計量レンジのチェックだけでは区別できず、この不一致は
> 衝突率の異常な高さ（pretrained単体で約80%）という形で現れていた。
> `policy_server.py` は現在 axis-angle 変換に修正済み（衝突率は大幅に改善したが、
> Track1 の摂動タスクでの成功率向上にはさらなる学習が必要）。
