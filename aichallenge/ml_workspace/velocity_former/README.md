# VelocityFormer Workspace

このworkspaceでは、[VelocityFormer](https://zenn.dev/bushio_tech/articles/7ff1e37b109402)用のデータ変換・学習・deployコードを提供しています。

- 参考記事: [自動運転AIチャレンジ：BERTを用いた車両制御モデルの構築](https://zenn.dev/bushio_tech/articles/7ff1e37b109402)
- オリジナル実装: [bushio/velocity-former](https://github.com/bushio/velocity-former)

VelocityFormer は、走行経路（trajectory）を入力として、車両制御コマンド（速度 or ステアリング）を回帰推論する BERT-tiny ベースのモデルです。
trajectory 上の連続する 2 点の方向ベクトルがなす角度（degree）を「擬似トークン ID」として BERT へ入力する点が特徴です。

VelocityFormer の実行用コードは、[velocity_former_controller](../../workspace/src/aichallenge_submit/velocity_former_controller) を参照してください。

## Quickstart (Training)

走行ログ収集 → ONNX 出力までの最短手順です。コンテナへの入り口は **rocker** (`./docker_run.sh dev`) を使う前提です。

### 前提

- リポジトリトップの `./setup.bash` 完了 (rocker と dev イメージ build 済み)
- AWSIM + Autoware が起動し、走行用コントローラ (例: `simple_pure_pursuit`) で `/planning/scenario_planning/trajectory` と `/control/command/control_cmd` が出ている状態
- 以降の手順は **すべて rocker shell の中** で実行 (= 別ターミナルで `./docker_run.sh dev` を起動して入る)

### 手順

```bash
# ホスト側で rocker shell を起動
./docker_run.sh dev
# 以降は rocker コンテナ内
```

1. **rosbag を記録** (rocker コンテナ内)
   ```bash
   bash /aichallenge/ml_workspace/record_data.bash
   ```
   → `aichallenge/ml_workspace/rawdata/<timestamp>/` に bag が保存される (mcap + zstd 圧縮)。

2. **データ抽出** (bag → npy)
   ```bash
   cd /aichallenge/ml_workspace/velocity_former
   python3 extract_data_from_bag.py --bags-dir ../rawdata --outdir ./datasets/
   ```
   bag ディレクトリごとに `trajectories.npy` / `velocities.npy` / `steers.npy` が出力されます。`./datasets/` 配下を `train/` と `val/` に手動で振り分けてください。

3. **学習** (速度・ステアリングを別 ONNX として個別に学習)
   ```bash
   python3 train.py data.train_dir=./datasets/train data.val_dir=./datasets/val model.label_type=velocity
   python3 train.py data.train_dir=./datasets/train data.val_dir=./datasets/val model.label_type=steering
   ```
   → `./checkpoints/` に best/last `.pth` が出力されます。

4. **ONNX 出力**
   ```bash
   python3 export_onnx.py --ckpt ./checkpoints/<best_velocity>.pth --output ./checkpoints/velocity_former_velocity.onnx
   python3 export_onnx.py --ckpt ./checkpoints/<best_steering>.pth --output ./checkpoints/velocity_former_steering.onnx
   ```

→ 実車 / シミュ上で走らせる手順は
[`velocity_former_controller`](../../workspace/src/aichallenge_submit/velocity_former_controller) の Quickstart (Deploy) を参照してください。

## 学習用データの作成

以下 2 つの Topic を含む rosbag を記録した後、`extract_data_from_bag.py` を実行します。

- [`autoware_auto_planning_msgs/msg/Trajectory`](https://github.com/tier4/autoware_auto_msgs/blob/tier4/main/autoware_auto_planning_msgs/msg/Trajectory.idl) : 走行経路 (入力)
- [`autoware_auto_control_msgs/msg/AckermannControlCommand`](https://github.com/tier4/autoware_auto_msgs/blob/tier4/main/autoware_auto_control_msgs/msg/AckermannControlCommand.idl) : 学習の target (教師) となる、速度・ステアリング情報を含むトピック

```bash
python3 extract_data_from_bag.py --bags-dir /path/to/record/ --outdir ./datasets/
```

出力されるシーケンス毎のディレクトリ構成:

```
datasets/
└── 20240101-120000/
    ├── trajectories.npy   # shape: (N, point_num, 2)  trajectory の (x, y) ポイント列
    ├── steers.npy         # shape: (N,)              ステアリング(rad)
    └── velocities.npy     # shape: (N,)              目標速度(m/s)
```

## 学習

`label_type=velocity` で速度推論モデル、`label_type=steering` でステアリング推論モデルを学習します。
両者は事前学習モデル（bert-tiny）と入出力サイズが共通で、ラベルのみ切り替えます。

```bash
# 速度推論
python3 train.py \
    data.train_dir=./datasets/train \
    data.val_dir=./datasets/val \
    model.label_type=velocity

# ステアリング推論
python3 train.py \
    data.train_dir=./datasets/train \
    data.val_dir=./datasets/val \
    model.label_type=steering
```

## ONNXエクスポート

採点環境で実行できるよう、PyTorchではなくONNX Runtimeで推論を行います。

```bash
python3 export_onnx.py --ckpt ./checkpoints/best_model.pth --output ./checkpoints/velocity_former.onnx
```

## モデルアーキテクチャ

- 事前学習モデル: [prajjwal1/bert-tiny](https://huggingface.co/prajjwal1/bert-tiny) (L=2, H=128, A=2)
- 入力: trajectory ポイント列を (x,y) → 角度(degree, 0–359 の整数 ID) 列に変換
- 出力: 速度 [m/s] または ステアリング角 [rad]
- Loss: SmoothL1Loss（tiny_lidar_net と同様）
