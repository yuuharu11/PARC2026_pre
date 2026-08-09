# PARC 2026 Track 1: ベースラインと実験計画

## 結論

最初の提出モデルは `lerobot/smolvla_libero_plus` を土台にし、
`examples/smolvla_libero_plus_multisuite_lora_local.py` で40個の基本命令をLoRA学習した
マージ済みモデルとする。公開4タスクだけを記憶する方策より、非公開タスクへの言語・物体・
ゴールの汎化を期待できるためである。

公開タスクはすべてLIBERO-plusの摂動版で、背景テクスチャ2件、照明2件である。
平均成功率だけでなく、摂動カテゴリ別成功率と衝突率を主要指標にする。

## 現在の基準値

- notebookと同じSpatial 10タスク評価では、保存済み実験結果が Base 66.7%、
  Spatial LoRA 83.3%（各タスク3 episode）であり、0%以外を達成済みである。
- Track 1公開4タスクの `results/server_8000.json` は、事前学習モデルで32 episodeすべて失敗、
  平均衝突率9.375%である。
- 通常条件のSpatial成功率だけでは不十分で、背景・照明摂動、128x128入力、衝突失格という
  Track 1条件での適応が必要である。

## Track 1実測結果

Track 1重点LoRAと、観測中の物体位置を利用する衝突回避型pick-and-place状態機械を組み合わせた
ハイブリッドポリシーを、本番パイプラインと同じ4タスク・各8 episode・最大600 stepsで評価した。

| 公開タスク | 成功 | 成功率 |
|---|---:|---:|
| drawer bowl → plate | 0/8 | 0% |
| tomato sauce → basket | 8/8 | 100% |
| milk → basket | 8/8 | 100% |
| bowl → stove | 0/8 | 0% |
| **全体** | **16/32** | **50%** |

結果は `results/track1_hybrid_8ep/server_8010.json` に保存されている。従来のSmolVLA単体の
0/32から16/32へ改善した。未知の命令はSmolVLAへフォールバックする。

3 episode評価は1回の成否で33.3ポイント変わる。モデル選択には最低10 episode、最終確認には
公開4タスク各20 episodeを使う。

## モデルを作る

GPU環境で全スイート学習スクリプトを実行する。

```bash
python examples/smolvla_libero_plus_multisuite_lora_local.py
```

既定では40命令のうち公開4タスクに対応する命令を各24 episode、残りを各4 episode、
合計240 episodeで8,000 steps、LoRA rank 16の学習を行う。照明・背景変化への耐性を狙い、
色・明るさ・コントラスト・鮮鋭度・小さな幾何変換のaugmentationも有効にしている。成果物を
`/content/smolvla_libero_plus_multisuite_lora_merged` に保存する。短時間の疎通確認には
Spatial版（10命令、3,000 steps）を使う。

学習後、モデルを提出テンプレートへ配置する。

```bash
cp -a /content/smolvla_libero_plus_multisuite_lora_merged \
  submission_template/model_weights
```

`policy_server.py` は `model_weights/` を自動検出する。コピーせず比較する場合は
`SMOLVLA_CHECKPOINT=/path/to/model` を指定する。

## 評価の順序

まず1タスク1 episodeでサーバー、観測変換、10秒制限を確認する。

```bash
source env.sh
SMOLVLA_CHECKPOINT=/content/smolvla_libero_plus_multisuite_lora_merged \
  python submission_template/policy_server.py --port 8000

python -m pipeline --server-url http://localhost:8000 --track track1 \
  --tasks pick_up_the_tomato_sauce_and_place_it_in_the_basket_table_27 \
  --n-episodes 1 --max-steps 600
```

学習スクリプト内の評価も、代表タスクではなく公開4タスクの正確なLIBERO-plus IDを
128x128で評価する。次に採点パイプラインで4タスク各10 episodeを比較し、最後に各20 episodeで確認する。

```bash
python -m pipeline --server-url http://localhost:8000 --track track1 \
  --n-episodes 10 --max-steps 600 --output-dir results/multisuite_lora
```

モデル選択は Track 1平均成功率、最低タスク成功率、衝突率、最大推論時間の順で見る。

## 次に効く改善

1. 背景テクスチャ・照明カテゴリのデモを多めにサンプリングする。公開4タスクだけでなく、
   40基本命令それぞれの同カテゴリ変種を混ぜる。
2. 128x128画像を含む学習またはaugmentationを追加し、256x256のnotebook評価との差を減らす。
3. 非対象物体を動かしたepisodeを除外または低重み化し、衝突失格に合わせる。
4. 複数seedで Base / Spatial LoRA / Multisuite LoRA を同じ初期状態で比較する。

公開4タスクへの手書き軌道やタスクID分岐は、非公開タスクと初期状態変化に弱いため主方針にしない。
