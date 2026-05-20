# planner_bev — フェーズ1: データ監査（チェックリスト）

`extract_data_from_bag.py` が書き出す `.npz`（`bev`, `traj_gt`, `mode_id`）と、抽出元 rosbag2 の **時刻整合**を確認するための手順です。量より **整合と分布の妥当性**を優先します。

## 自動集計スクリプト

`planner_bev/` をカレントにして実行します。

```bash
cd aichallenge/ml_workspace/planner_bev
python3 scripts/audit_planner_npz.py \
  --train-dir datasets/from_bag/train \
  --val-dir datasets/from_bag/val
```

機械可読出力:

```bash
python3 scripts/audit_planner_npz.py \
  --train-dir datasets/from_bag/train \
  --val-dir datasets/from_bag/val \
  --json > audit_npz.json
```

### 任意: バッグ同期のリプレイ

抽出時と **同じ** `sync_slop_ms` / `stride` / `horizon` / `max_arclen_m` を指定し、BEV 各フレームに対する **odom / trajectory の遅れ（ms）**の分位数を出します（`extract_samples` と同一ゲート）。

```bash
python3 scripts/audit_planner_npz.py \
  --train-dir datasets/from_bag/train \
  --val-dir datasets/from_bag/val \
  --bag datasets/rosbag2_planner/planner_<timestamp> \
  --sync-slop-ms 80 --stride 2 --horizon 40 --max-arclen-m 40
```

解釈の目安:

- `n_synced` が極端に少ない → `sync_slop_ms` を緩めるか、bag 内の `/localization/kinematic_state` と **教師軌道**（既定 `/mpc/prediction`、従来 `/planning/scenario_planning/trajectory`）のタイムスタンプを確認。
- `dt_bev_traj_ms` の **p99** が `sync_slop_ms` に張り付いている → ぎりぎり通過しており、実質的なズレが大きい可能性。`planner_rosbag_recording.md` の録画・QoS も併せて確認。

---

## チェックリスト（手動・目視）

### A. 時刻同期（rosbag / 抽出パラメータ）

- [ ] 上記 `--bag` 付き監査で `skipped_*` の内訳が許容範囲か（特に `skipped_traj_slop`, `skipped_odom_slop`）。
- [ ] `dt_bev_odom_ms` / `dt_bev_traj_ms` の中央値が小さいか（数十 ms 以下が望ましいことが多い）。
- [ ] 抽出に使った **`--sync-slop-ms` / `--stride`** を `train.yaml` や実験メモに記録し、監査と **同じ値**で `--bag` を再実行できるようにする。

### B. `mode_id`（k-means 終点クラスタ）の偏り

- [ ] `mode_id_counts` が極端に偏っていないか（1 クラスタに >90% 等）。偏りが強いと未使用ヘッドが増え、学習と推論の契約が崩れやすい。
- [ ] **直進が支配的**なコースでは、終点 k-means の境界が **幾何的に恣意的**になり、`mode_id` が似た軌道でバラける — フェーズ2（教師再定義）で扱う前提としてメモする。

### C. 終点分布・軌道の妥当性（npz のみ）

- [ ] `endpoint_xy` の mean / quantiles がコースと矛盾しないか（例: ほぼ直進データなのに `y` の分位が大きい等）。
- [ ] `straight_heuristic_fraction` が主観と合うか（高速周回のみなのに極端に低い等 → 座標系・ホライズン・リサンプルの確認）。
- [ ] `mono_fwd_step_ratio_mean`（監査スクリプト出力）が異常に低くないか。低いと `traj_gt` の x 単調性が崩れているサンプルが多い可能性。

### D. train / val 分割のズレ（時間末尾 `val_ratio`）

`write_npz_split` は **スタンプ昇順で並べた末尾 `val_ratio` を val** にします。

- [ ] `train_val_compare.mode_id_fraction_delta_val_minus_train` が大きくないか。大きいと **後半区間だけモード分布が変わっている**（天候・コース区間・ペース変化）可能性。
- [ ] `mean_endpoint_l2_train_vs_val_m` が大きいと、**空間的な分布シフト**の疑い（終盤だけ別レイアウト等）。
- [ ] 意図的に **ランダム split** や **セッション単位 split** に変えるかはフェーズ2以降の設計判断。

### E. BEV チャネル統計

- [ ] `bev_channel_mean_over_dataset` / `std` が train と val で大きく食い違わないか（極端な差はカメラ・正規化・話題欠落のサインになり得る）。

---

## このブロックの「完了」条件（提案）

1. 上記スクリプトを **現行 bag → npz パイプライン**に対して実行し、出力（または `audit_npz.json`）をレビュー済み。
2. チェックリスト A〜E で **NG 項目が「既知の制約」として文書化**されている（直ちに直せないものはフェーズ2以降にチケット化）。

完了後、**フェーズ2（`mode_id` / 教師の再定義）**に進みます。

---

## フェーズ2〜4（実装済みツールの索引）

| フェーズ | 内容 | リポジトリ内の入口 |
|----------|------|-------------------|
| 2 教師 | `mode_id` の付け方 | `design_docs/planner_mode_id_labeling.md`、`extract_data_from_bag.py --mode-label` |
| 3 ルール | 推論ヘッド選択の切り分け | `viz_val_infer_bev.py` / `infer_demo.py` の `--selection teacher\|rule` |
| 4 学習 | 非教師ヘッドへの弱教師 | `config/train.yaml` の `train.loss.aux_all_heads_lambda`（`lib/loss.py`） |

`rule_score` の重み調整（`w_obs` 等）はコード定数のままのため、次のイテレーションで CLI 化する場合は `lib/rule_score.py` を拡張します。
