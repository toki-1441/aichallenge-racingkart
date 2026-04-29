# velocity_former_controller

走行経路（`autoware_auto_planning_msgs/Trajectory`）を入力に、BERT-tiny ベースの VelocityFormer で
速度・ステアリングを推論し、`AckermannControlCommand` をパブリッシュする ROS 2 ノードです。

> **学習〜デプロイの step-by-step 手順は [develop_velocity_former.md](https://automotiveaichallenge.github.io/aichallenge-documentation-racingkart/ml_sample/develop_velocity_former.html) を参照。** 本 README は技術仕様 (アーキテクチャ・パラメータ・I/O) のみ。

学習スクリプト・データ抽出は
[`aichallenge/ml_workspace/velocity_former`](../../../../ml_workspace/velocity_former) を参照してください。

## Quickstart (Deploy)

学習済み ONNX を使って車両を走らせるまでの最短手順です。コンテナへの入り口は **rocker** (`./docker_run.sh dev`) を使う前提です。

### 前提

- リポジトリトップの `./setup.bash` 完了 (rocker と dev イメージ build 済み)
- 学習済み ONNX (`velocity_former_velocity.onnx` / 任意で `velocity_former_steering.onnx`) が手元にある
  (作り方は [`ml_workspace/velocity_former`](../../../../ml_workspace/velocity_former) の Quickstart (Training) 参照)

### 手順

1. **ONNX を配置** (ホスト側で)
   ```bash
   cp /path/to/velocity_former_velocity.onnx \
      aichallenge/workspace/src/aichallenge_submit/velocity_former_controller/ckpt/
   # ステアリング推論も使う場合は velocity_former_steering.onnx も同じ場所へ
   ```

2. **rocker shell に入る** (ホスト側で)
   ```bash
   ./docker_run.sh dev
   ```
   `aichallenge/` がコンテナの `/aichallenge` にマウントされます。

3. **ビルド & launch** (rocker コンテナ内)
   ```bash
   cd /aichallenge/workspace
   colcon build --symlink-install --packages-select velocity_former_controller \
       --cmake-args -DCMAKE_BUILD_TYPE=Release
   source install/setup.bash
   ros2 launch velocity_former_controller velocity_former.launch.xml
   ```
   `velocity_former.launch.xml` の `velocity_onnx_path` のデフォルトは `$(find-pkg-share velocity_former_controller)/ckpt/velocity_former_velocity.onnx` を指しています。別パスを使う場合は launch 引数で上書きしてください:
   ```bash
   ros2 launch velocity_former_controller velocity_former.launch.xml \
       velocity_onnx_path:=/abs/path/to/velocity_former_velocity.onnx \
       steering_onnx_path:=/abs/path/to/velocity_former_steering.onnx
   ```
   ステアリング推論を有効化する場合は `config/velocity_former_node.param.yaml` の `control_mode.mode` を `both` または `steering_only` に変更してください。

4. **動作確認** (別 rocker shell で)
   ```bash
   # ホスト側で別ターミナルから
   ./docker_run.sh dev
   # rocker コンテナ内
   source /aichallenge/workspace/install/setup.bash
   ros2 topic echo /control/command/control_cmd --field longitudinal.speed
   ```
   trajectory に応じて値が変化していれば OK。

→ モデルを学習する側の手順は
[`ml_workspace/velocity_former`](../../../../ml_workspace/velocity_former) の Quickstart (Training) を参照してください。

## 依存

- ROS 2 Humble
- Python: `numpy`, `onnxruntime`

## 入出力トピック

| 種別 | トピック | 型 |
|------|----------|-----|
| Sub  | `/planning/scenario_planning/trajectory` | `autoware_auto_planning_msgs/Trajectory` |
| Pub  | `/control/command/control_cmd`           | `autoware_auto_control_msgs/AckermannControlCommand` |

## 制御モード（`control_mode.mode`）

- `velocity_only` : 速度のみ推論（`steering` は `fallback_steering`）
- `steering_only` : ステアリングのみ推論（`speed` は `fallback_velocity`）
- `both`          : 速度・ステアリングの両方を 2 つの ONNX で個別に推論

学習側もタスク（速度/ステアリング）ごとに別 ONNX を出力する設計です。

## launch

```bash
ros2 launch velocity_former_controller velocity_former.launch.xml \
    velocity_onnx_path:=/path/to/velocity_former_velocity.onnx
```

ステアリング推論を使う場合:

```bash
ros2 launch velocity_former_controller velocity_former.launch.xml \
    velocity_onnx_path:=/path/to/velocity_former_velocity.onnx \
    steering_onnx_path:=/path/to/velocity_former_steering.onnx
```

`config/velocity_former_node.param.yaml` で `control_mode.mode` を `both` などに変更してください。
