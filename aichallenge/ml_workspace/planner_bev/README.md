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
| `config/planner_record_topics.txt` | 学習用 **ros2 bag record** の既定トピック一覧 |
| `config/planner_record_topics_extended.txt` | debug 画像・V2X 等を含む拡張リスト |
| `config/planner_record_qos_overrides.yaml` | `ros2 bag record` 用 QoS（BEST_EFFORT 軌道等）。**clock-only bag** 対策 |
| `scripts/record_planner_training_bag.bash` | mcap + zstd + QoS。**撮影前**に `ros2 topic list` と `ros2 topic info`（Publisher 数）で検証（`--check-topics` は検証のみ）。マニフェストはサイドカー→成功後に bag 内へコピー |
| `scripts/write_planner_record_manifest.py` | マニフェスト JSON 生成（上記から呼び出し） |
| `extract_data_from_bag.py` | rosbag2 → `train`/`val` の `.npz` 抽出（`rosbags` 必須） |
| `train.py` | 学習ループ + **詳細 TensorBoard**（スカラー／画像／軌道図／ヒストグラム） |
| `lib/tb_utils.py` | TensorBoard 用 BEV 可視化・軌道 matplotlib・重み／勾配ヒスト |
| `infer_demo.py` | 1 サンプル forward + ルール選択の確認 |
| `viz_val_infer_bev.py` | **val の一部**で推論し、BEV 上に GT・全ヘッド・ルール選択を重ねた **PNG** を出力 |
| `scripts/audit_planner_npz.py` | **フェーズ1: データ監査** — train/val `.npz` の `mode_id`・終点・BEV チャネル統計、任意で bag 同期リプレイ（`design_docs/planner_npz_data_audit.md`） |
| `run_pipeline.bash` | 合成データ生成 → 学習 |

## セットアップ

```bash
cd aichallenge/ml_workspace/planner_bev
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-torch-cu128.txt
pip install -r requirements.txt
```

`torch` は `requirements-torch-cu128.txt` で [PyTorch の cu128 インデックス](https://download.pytorch.org/whl/cu128)から入ります（コード側の `import torch` はそのままです）。

## データ

- **開発用**: `prepare_data.py synthetic` が `datasets/synthetic/{train,val}` に `.npz` を書き出す（`datasets/` は通常 `.gitignore`）。
- **本番用**: `scripts/record_planner_training_bag.bash` で rosbag2 を取得し、`extract_data_from_bag.py` で `.npz` に変換（トピックは `design_docs/planner_rosbag_recording.md`）。録画中は BEV・軌道・odom が出ていることを確認してください。
- **データ監査（フェーズ1）**: `python3 scripts/audit_planner_npz.py --train-dir … --val-dir …` と `design_docs/planner_npz_data_audit.md` のチェックリストで、同期・`mode_id` 偏り・train/val 分布ズレを確認する。

各 `.npz` のキー:

- `bev`: `float32` `(4, 256, 144)`
- `traj_gt`: `float32` `(T, 2)` — ego `x` 前方、`y` 左
- `mode_id`: `int64` スカラー — `0 … K-1`（P1 ハード割当教師）

任意: `aux` `float32` `(D,)` — 使う場合は `config/train.yaml` の `model.aux_dim` を `D` に合わせる。

## 学習の実行方法

作業ディレクトリは常に `aichallenge/ml_workspace/planner_bev` です。

### 1. 環境

```bash
cd aichallenge/ml_workspace/planner_bev
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-torch-cu128.txt
pip install -r requirements.txt
```

（`torch` は **CUDA 12.8 付き**の公式 wheel 用インデックス `cu128` から入れます。CPU のみの場合は [PyTorch の CPU 手順](https://pytorch.org/get-started/locally/)に合わせて `torch` だけ差し替えてください。）

### 2-A. 合成データで学習（開発・動作確認）

`config/train.yaml` の `data.train_dir` / `data.val_dir`（既定は `datasets/synthetic/...`）に合わせてサンプルを生成してから学習します。

```bash
cd aichallenge/ml_workspace/planner_bev
source .venv/bin/activate
python3 prepare_data.py synthetic \
  --out-train datasets/synthetic/train \
  --out-val datasets/synthetic/val \
  --num-train 1024 --num-val 128 \
  --horizon 40 --num-heads 4 --seed 0
python3 train.py
```

ワンライナー相当: `./run_pipeline.bash`（内部は上記に近い件数で `train.py` まで実行）。

### 2-B. rosbag 由来の `.npz` で学習

bag ディレクトリ（`metadata.yaml` がある階層）を指定して抽出し、Hydra でデータパスだけ差し替えます。`--k-modes` は `config/train.yaml` の `model.num_heads` と揃えてください。教師軌道の既定トピックは **`/mpc/prediction`**（`MarkerArray`）です。旧 bag のみの場合は `--traj-topic /planning/scenario_planning/trajectory` を付けてください。`mode_id` の付け方は `--mode-label`（`design_docs/planner_mode_id_labeling.md`）で変更できます。抽出後は `scripts/audit_planner_npz.py` と `design_docs/planner_npz_data_audit.md` で監査してください。

```bash
cd aichallenge/ml_workspace/planner_bev
source .venv/bin/activate
python3 extract_data_from_bag.py \
  --bag datasets/rosbag2_planner/planner_<タイムスタンプ> \
  --out-train datasets/from_bag/train \
  --out-val datasets/from_bag/val \
  --horizon 40 --stride 2 --sync-slop-ms 120 --max-arclen-m 40 --k-modes 4 --mode-label kmeans --overwrite
python3 scripts/audit_planner_npz.py \
  --train-dir datasets/from_bag/train \
  --val-dir datasets/from_bag/val \
  --bag datasets/rosbag2_planner/planner_<タイムスタンプ> \
  --sync-slop-ms 120 --stride 2 --horizon 40 --max-arclen-m 40
python3 train.py \
  data.train_dir=datasets/from_bag/train \
  data.val_dir=datasets/from_bag/val
```

### 3. 学習の上書き（Hydra）

`config/train.yaml` を直接編集せず、コマンドラインで上書きできます。

```bash
python3 train.py train.epochs=50 train.batch_size=16 train.lr=3e-4 model.num_heads=4
```

### 4. 成果物

- **チェックポイント**: `checkpoints/best_model.pth`（`hydra.job.chdir: false` のため `planner_bev/` 直下）
- **TensorBoard**: `logs/<日時>/` — 別シェルで `tensorboard --logdir logs --port 6006 --bind_all`

### 5. 学習後の動作確認（任意）

```bash
python3 infer_demo.py --ckpt checkpoints/best_model.pth --npz datasets/synthetic/val/000000.npz
python3 infer_demo.py --ckpt checkpoints/best_model.pth --npz datasets/synthetic/val/000000.npz --selection teacher
```

## TensorBoard（詳細可視化・任意）

学習中に **`logs/<タイムスタンプ>/`** が更新されます。別ターミナルで:

```bash
cd aichallenge/ml_workspace/planner_bev
source .venv/bin/activate   # 未実行なら
tensorboard --logdir logs --port 6006 --bind_all
```

ブラウザで `http://localhost:6006` を開くと、次のタグが利用できます。

| プレフィックス | 内容 |
|----------------|------|
| `SCALARS` **train/batch/** | 各ステップの `loss_total` / `loss_pose` / `loss_aux_all`（λ>0 時）/ `loss_div` / `loss_curv` / `lr` / `grad_norm_l2` |
| **train/epoch/**, **val/epoch/** | エポック平均損失・`lr`・`val_train_loss_ratio` |
| **val/hist/** | 検証全バッチの「教師ヘッド上の平均点 L2」分布ヒストグラム + mean/std スカラー |
| **weights/**, **grads/** | 畳み込み／ヘッド等の重み・勾配ヒストグラム（頻度は `train.tensorboard.*`） |
| **IMAGES** **train/viz/bev/**, **val/viz/bev/** | 4 チャネル（lane / traj / obstacles / ego）をダウンサンプルした RGB 擬似画像 |
| **FIGURES** **train/viz/trajectory_xy**, **val/viz/trajectory_xy** | GT と K 本予測の平面軌跡（教師ヘッド強調） |
| **TEXT** **config/** | フル YAML と JSON 化したハイパラツリー |

頻度・対象レイヤは `config/train.yaml` の `train.tensorboard` で調整します（例: `batch_scalars_interval: 10` でログ量削減、`weight_histogram_substrings: []` で全パラメータの重みヒスト、`log_graph: true` で計算グラフ追加※環境によって失敗する場合あり）。

## rosbag2 録画（学習データ取得）

無限周回など長時間でも、**必要トピックだけ**を録画するスクリプトを用意しています。

1. シミュ＋Autoware（`bev_scene_stack` 等）を起動し、`source <aichallenge>/workspace/install/setup.bash` で `ros2` を有効化する。
2. **推奨**: `./scripts/record_planner_training_bag.bash --check-topics` で、`ros2 topic list` に全トピックがあることと、`ros2 topic info` の **Publisher count**（`/parameter_events` は除外、`/aichallenge/objects` は 0 でも WARN のみ）を確認する。
3. `./scripts/record_planner_training_bag.bash -o ...` で録画（**開始直前にも同じ検証**が走る。`SKIP_RECORD_PREFLIGHT=1` で省略可・非推奨）。終了は **Ctrl+C** または `-d 秒`。

`/clock` しか bag に入らない場合は `design_docs/planner_rosbag_recording.md` §3.1 を参照。

```bash
cd aichallenge/ml_workspace/planner_bev
export AICHALLENGE_WORKSPACE=/path/to/aichallenge/workspace   # 任意: 自動 source 用
chmod +x scripts/record_planner_training_bag.bash scripts/write_planner_record_manifest.py
./scripts/record_planner_training_bag.bash --check-topics   # 撮影前にトピック＋Publisher 確認のみ
./scripts/record_planner_training_bag.bash -o ./datasets/rosbag2_planner
# 拡張トピック（debug 画像等）: -t config/planner_record_topics_extended.txt
```

出力: `-o` 直下に `planner_<タイムスタンプ>/`（mcap。`ros2` が新規作成）、その親に **`planner_<タイムスタンプ>.dataset_manifest.json`**（録画開始時点のトピック一覧）。録画が一度でもディレクトリを作成できた場合は、同内容を **`planner_<タイムスタンプ>/dataset_manifest.json`** にもコピーします。

## 推論デモ

```bash
python3 infer_demo.py --ckpt checkpoints/best_model.pth --npz datasets/synthetic/val/000000.npz
python3 infer_demo.py --ckpt checkpoints/best_model.pth --npz datasets/synthetic/val/000000.npz --selection teacher
```

（`--ckpt` は学習済み重み、`--npz` は任意のサンプル。`--selection teacher` で `mode_id` ヘッドを選択表示し、**rule 選択との切り分け**に使います。）

## val 推論の BEV 可視化（画像出力）

学習後に **検証ディレクトリから最大 N 件**を取り、推論軌道を **BEV グリッド座標**（`lib/rule_score.BEVGridSpec` と同じ式）に投影して PNG を書き出します。左パネルが BEV オーバーレイ、右が ego 平面の軌道図です。

```bash
cd aichallenge/ml_workspace/planner_bev
source .venv/bin/activate
python3 viz_val_infer_bev.py \
  --ckpt checkpoints/best_model.pth \
  --config config/train.yaml \
  --val-dir datasets/synthetic/val \
  --out-dir viz_val_infer \
  --num-samples 16 \
  --device cpu
# 教師ヘッドを「選択」として描画（rule との比較用）:
# python3 viz_val_infer_bev.py ... --selection teacher
```

- **緑**: 教師 `traj_gt`  
- **シアン**: 教師モード `mode_id` に対応するヘッドの予測  
- **薄色**: その他ヘッド  
- **黄**: `rule_score` で選ばれたヘッド  

`--val-dir` を省略すると `config/train.yaml` の `data.val_dir` を使います。

## ルールスコア

`lib/rule_score.py` の `score_trajectory` / `select_best_trajectory` は、BEV の **障害チャネル・レーンチャネル**と軌道幾何の簡易コストで **低いほど良い**スカラーを返す。重みは呼び出し側で調整。

## 参照ドキュメント

- `design_docs/planner_training_inputs_and_losses.md`
- `design_docs/planner_model_flow_matching_architecture.md`
- `design_docs/planner_rosbag_recording.md`（bag に含めるトピック一覧・理由）
