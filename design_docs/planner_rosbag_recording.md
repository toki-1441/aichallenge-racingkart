# Planner 学習用 rosbag2: 保存すべきトピックと運用

無限周回など長時間シミュレーションでは、**学習に必要な信号だけ**を選んで bag に落とすと容量・後処理が楽です。本稿は `design_docs/planner_training_inputs_and_losses.md` の **P1 契約**と、現行スタック（`reference.launch.xml` + `bev_scene_stack`）のトピック名に揃えています。

---

## 1. 学習パイプラインが最終的に欲しいもの（論理データ）

| データ | 形状・意味 | 主なソース |
|--------|------------|------------|
| **BEV テンソル** | `(C,H,W)`、`C=4`（lane / trajectory / obstacles / ego） | `bev_scene_stack` が合成した **`/bev_scene_stack/tensor`**（`Float32MultiArray`） |
| **教師軌道** | 将来 `(T,2)` にリサンプル（ego 系） | **既定: `/mpc/prediction`**（`visualization_msgs/MarkerArray`、map 系・MPC 予測）。**従来:** `/planning/scenario_planning/trajectory`（`Trajectory`、CSV 1Hz 等で静的になりやすい）。`extract_data_from_bag.py --traj-topic` で選択。 |
| **補助ベクトル（任意）** | 速度・ヨー等 | **`/localization/kinematic_state`**（`Odometry`） |
| **時間軸** | `use_sim_time` 時の再生整合 | **`/clock`** |
| **座標変換** | map ↔ base 等 | **`/tf`**, **`/tf_static`** |

**P1 の最小 bag セット**（`extract_data_from_bag` がまず期待する入力）は **BEV + odom + 教師軌道**です。教師軌道は既定で **`/mpc/prediction`**（`--traj-topic` で変更可）。  
`mode_id` は bag には含めず、**オフライン**で k-means 等により `.npz` に付与する想定です（`planner_training_inputs_and_losses.md` §4.2）。

---

## 2. トピック別: なぜ必要か

### 必須（最小）

| トピック | メッセージ型（代表） | 保存理由 |
|----------|----------------------|----------|
| `/bev_scene_stack/tensor` | `std_msgs/Float32MultiArray` | 学習入力 **BEV** の唯一の「既にラスタ化された」ソース。レイアウトは `bev_scene_stack` の channel-major（設計書 §2.1 と同一）。 |
| `/mpc/prediction` | `visualization_msgs/msg/MarkerArray` | **推奨・教師経路（MPC 制御ループから高レート）**。`multi_purpose_mpc_ros` が map 上に SPHERE マーカー列で公開。オフライン抽出の **既定**（`extract_data_from_bag.py`）。 |
| `/planning/scenario_planning/trajectory` | `autoware_auto_planning_msgs/msg/Trajectory` | **従来の教師**（`simple_trajectory_generator` 等）。BEV の trajectory チャネルは現状こちらを購読していることが多い。静的／低 Hz のときは `/mpc/prediction` を録画し抽出側で `--traj-topic` を合わせる。 |
| `/localization/kinematic_state` | `nav_msgs/msg/Odometry` | **補助入力**、BEV と教師の時刻整合・ego 速度、将来の map→base 投影。 |
| `/clock` | `rosgraph_msgs/msg/Clock` | **`use_sim_time=true`** の環境では必須。無いと bag 再生や同期抽出が破綻しやすい。 |
| `/tf` | `tf2_msgs/msg/TFMessage` | 教師軌道（通常 map）を **ego 平面座標**に揃えるため。 |
| `/tf_static` | `tf2_msgs/msg/TFMessage` | 同上。 |

### 強く推奨（デバッグ・再現性）

| トピック | 保存理由 |
|----------|----------|
| `/aichallenge/objects` | BEV の障害チャネルの原料。テンソルだけでは「なぜこう描いたか」追いにくいときに再現・差分検証に使う。 |
| `/parameter_events` | 同一 bag でもパラメータ変更が混ざったときの切り分け。 |

### 任意（容量大・学習直結はしない）

| トピック | 注意 |
|----------|------|
| `/bev_scene_stack/debug_image` | 可視化用 RGB。**帯域・容量が大きい**。品質確認用にだけ録画推奨。 |
| `/v2x/gnss/pose_with_covariance` | 他車 V2X が有効なとき BEV の相手車に効く。**1 台のみ**なら空振りでも可。 |
| `/map/vector_map_marker` | BEV のレーン描画の参照。オフラインで BEV を再生成する場合のみ意味が大きい。 |

---

## 3. 保存機構（リポジトリ内）

| パス | 役割 |
|------|------|
| `aichallenge/ml_workspace/planner_bev/config/planner_record_topics.txt` | **既定の最小＋推奨**トピック一覧（1 行 1 トピック、`#` でコメント）。 |
| `aichallenge/ml_workspace/planner_bev/config/planner_record_topics_extended.txt` | debug 画像・V2X など **拡張**録画用。 |
| `aichallenge/ml_workspace/planner_bev/config/planner_record_qos_overrides.yaml` | `ros2 bag record --qos-profile-overrides-path` 用。**BEST_EFFORT** の軌道・objects、**TRANSIENT_LOCAL** の `/tf_static` が既定の bag 購読と合わず **/clock しか入らない**症状を防ぐ。 |
| `aichallenge/ml_workspace/planner_bev/scripts/record_planner_training_bag.bash` | `ros2 bag record` を起動。**録画前**に `ros2 topic list` と `ros2 topic info`（Publisher 数）でトピックを検証。`--check-topics` で検証のみ。事前に **サイドカー** `planner_<TS>.dataset_manifest.json` を書き、録画後に bag ディレクトリへコピー。 |
| `aichallenge/ml_workspace/planner_bev/scripts/write_planner_record_manifest.py` | マニフェスト JSON 生成（スクリプトから呼び出し）。 |

### 使い方（概要）

1. シミュ／実機で Autoware 系スタックを起動（`bev_scene_stack` が出ていること）。
2. `source <workspace>/install/setup.bash`（`ros2` が使えること）。
3. **推奨**: `./scripts/record_planner_training_bag.bash --check-topics` でトピック一覧と Publisher 数だけ確認（録画はしない）。
4. `./scripts/record_planner_training_bag.bash -o ...` で録画（**録画開始直前にも同じ検証**が走る。省略は `SKIP_RECORD_PREFLIGHT=1`、非推奨）。

詳細オプションは `scripts/record_planner_training_bag.bash -h` または `planner_bev/README.md` の録画節を参照。

---

## 3.1 bag に `/clock` しか入らないとき

典型原因は次の二つです。

1. **録画開始時点でスタックが未起動**  
   `bev_scene_stack`・`simple_trajectory_generator`・EKF などが立っていないと、`ros2 topic list` にも出ず bag にも載りません。  
   **対策**: `reference.launch` 等を起動し、`ros2 topic hz /bev_scene_stack/tensor` がレートを出すまで待ってから録画する。録画スクリプトは既定で `ros2 topic list` と **`ros2 topic info` の Publisher 数**で検証します（`SKIP_RECORD_PREFLIGHT=1` で無効化可）。

2. **QoS の不一致（特に BEST_EFFORT）**  
   `simple_trajectory_generator` の **`/planning/scenario_planning/trajectory`** は **BEST_EFFORT** で出ています（`bev_scene_stack_node.py` コメント参照）。`ros2 bag record` の購読側が RELIABLE 寄りだと **マッチせずメッセージ 0 本**になります。`/aichallenge/objects` も同様に BEST_EFFORT です。**`/mpc/prediction`** は MPC 側 publisher 依存（既定 overrides は RELIABLE / keep_last）。  
   **対策**: `planner_record_qos_overrides.yaml` を `--qos-profile-overrides-path` で渡す。本リポジトリの録画スクリプトはこれを**既定で付与**します（`--no-qos-overrides` でオフ）。

---

## 4. 既存 `vehicle/record_rosbag.bash` との関係

`vehicle/record_rosbag.bash` は **評価・解析用の広いトピック集合**と **60 秒分割**を想定した例です。Planner 専用には本稿の **絞り込みリスト**の方が向きます。必要なら両方併用しても構いません。

---

## 5. 次工程（オフライン）

bag から `(bev, traj_gt, mode_id)` の `.npz` を書く処理は `planner_bev/extract_data_from_bag.py` です。**既定の教師トピックは `/mpc/prediction`**（`--traj-topic` / `--traj-source`）。旧 bag のみの場合は `--traj-topic /planning/scenario_planning/trajectory`、第三トピックが無い場合は `--traj-source odom_extrap`。録画時は本リストと QoS 運用に沿えば、その抽出器と **契約が一致**します。
