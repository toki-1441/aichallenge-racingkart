# planner_bev — BEV 条件付き多軌道 planner（P1）

`design_docs/planner_training_inputs_and_losses.md` および `planner_model_flow_matching_architecture.md` に沿った **本格学習用コード**です。

## 構成

| パス | 役割 |
|------|------|
| `config/train.yaml` | Hydra 設定（BEV 形状、`K`、損失係数、`hydra.job.chdir: false`） |
| `lib/schema.py` | `4×256×144` 等の契約定数 |
| `lib/model.py` | CNN encoder + `K` 本軌道ヘッド → `(B,K,T,2)` |
| `lib/loss.py` | `L_pose`（モード割当 Huber）+ 任意 `L_div` / `L_curv` |
| `lib/data.py` | `.npz` データセット（`bev`, `traj_gt`, `mode_id`） |
| `lib/rule_score.py` | 推論時 **ルールスコア**でヘッド選択（numpy） |
| `prepare_data.py` | `synthetic` サブコマンドで合成データ生成 |
| `extract_data_from_bag.py` | rosbag 抽出の **スキャフォールド**（未実装） |
| `train.py` | 学習ループ + **詳細 TensorBoard**（スカラー／画像／軌道図／ヒストグラム） |
| `lib/tb_utils.py` | TensorBoard 用 BEV 可視化・軌道 matplotlib・重み／勾配ヒスト |
| `infer_demo.py` | 1 サンプル forward + ルール選択の確認 |
| `run_pipeline.bash` | 合成データ生成 → 学習 |

## セットアップ

```bash
cd aichallenge/ml_workspace/planner_bev
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## データ

- **開発用**: `prepare_data.py synthetic` が `datasets/synthetic/{train,val}` に `.npz` を書き出す（`datasets/` は通常 `.gitignore`）。
- **本番用**: `extract_data_from_bag.py` に rosbag→`.npz` を実装するか、外部パイプラインで同スキーマの `.npz` を生成する。

各 `.npz` のキー:

- `bev`: `float32` `(4, 256, 144)`
- `traj_gt`: `float32` `(T, 2)` — ego `x` 前方、`y` 左
- `mode_id`: `int64` スカラー — `0 … K-1`（P1 ハード割当教師）

任意: `aux` `float32` `(D,)` — 使う場合は `config/train.yaml` の `model.aux_dim` を `D` に合わせる。

## 学習

```bash
cd aichallenge/ml_workspace/planner_bev
python3 prepare_data.py synthetic \
  --out-train datasets/synthetic/train \
  --out-val datasets/synthetic/val \
  --num-train 1024 --num-val 128
python3 train.py
```

上書き例:

```bash
python3 train.py train.epochs=50 train.batch_size=16 train.loss.div_lambda=0.02
```

チェックポイント: `checkpoints/best_model.pth`（`hydra.job.chdir: false` のため `planner_bev/` 直下）。

## TensorBoard（詳細可視化）

学習開始時に **`logs/<タイムスタンプ>/`** が作られます。別ターミナルで:

```bash
cd aichallenge/ml_workspace/planner_bev
tensorboard --logdir logs --port 6006 --bind_all
```

ブラウザで `http://localhost:6006` を開くと、次のタグが利用できます。

| プレフィックス | 内容 |
|----------------|------|
| `SCALARS` **train/batch/** | 各ステップの `loss_total` / `loss_pose` / `loss_div` / `loss_curv` / `lr` / `grad_norm_l2` |
| **train/epoch/**, **val/epoch/** | エポック平均損失・`lr`・`val_train_loss_ratio` |
| **val/hist/** | 検証全バッチの「教師ヘッド上の平均点 L2」分布ヒストグラム + mean/std スカラー |
| **weights/**, **grads/** | 畳み込み／ヘッド等の重み・勾配ヒストグラム（頻度は `train.tensorboard.*`） |
| **IMAGES** **train/viz/bev/**, **val/viz/bev/** | 4 チャネル（lane / traj / obstacles / ego）をダウンサンプルした RGB 擬似画像 |
| **FIGURES** **train/viz/trajectory_xy**, **val/viz/trajectory_xy** | GT と K 本予測の平面軌跡（教師ヘッド強調） |
| **TEXT** **config/** | フル YAML と JSON 化したハイパラツリー |

頻度・対象レイヤは `config/train.yaml` の `train.tensorboard` で調整します（例: `batch_scalars_interval: 10` でログ量削減、`weight_histogram_substrings: []` で全パラメータの重みヒスト、`log_graph: true` で計算グラフ追加※環境によって失敗する場合あり）。

## 推論デモ

```bash
python3 infer_demo.py --ckpt checkpoints/best_model.pth --npz datasets/synthetic/val/000000.npz
```

（`--ckpt` は `best_model.pth` のパス）

## ルールスコア

`lib/rule_score.py` の `score_trajectory` / `select_best_trajectory` は、BEV の **障害チャネル・レーンチャネル**と軌道幾何の簡易コストで **低いほど良い**スカラーを返す。重みは呼び出し側で調整。

## 参照ドキュメント

- `design_docs/planner_training_inputs_and_losses.md`
- `design_docs/planner_model_flow_matching_architecture.md`
