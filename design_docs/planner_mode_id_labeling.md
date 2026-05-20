# planner_bev — `mode_id`（教師モード）の付け方

`extract_data_from_bag.py` の **`--mode-label`** で、`mode_id` の定義を選べます（`model.num_heads` / `--k-modes` と一致させる）。

| 値 | 意味 | 向いているケース |
|----|------|------------------|
| `kmeans`（既定） | 全サンプルの **軌道終点 (x,y)** を k-means で K クラスタし、そのクラスタ index | 終点が明確に K 団子に分かれる多峰データ |
| `angular` | 軌道の **弦ベクトル** `(end - start)` の方位角を **K 等分ビン**に割り当て | 終点は似通うが「向き」で分けたいとき（直進だらけで k-means が不安定になりやすい対策） |
| `singleton` | 常に `mode_id=0`。**`--k-modes 1`** 必須（学習も `model.num_heads: 1`） | まず単峰で教師品質・同期だけ切り分けたいとき |

## 使い方

```bash
cd aichallenge/ml_workspace/planner_bev
# 例: 方位ビン（K=4 のまま）
python3 extract_data_from_bag.py \
  --bag datasets/rosbag2_planner/planner_<timestamp> \
  --out-train datasets/from_bag/train \
  --out-val datasets/from_bag/val \
  --k-modes 4 --mode-label angular --overwrite

# 例: 単峰（モデルも K=1 に変更してから学習）
python3 extract_data_from_bag.py ... --k-modes 1 --mode-label singleton --overwrite
```

## 監査との関係

フェーズ1の `scripts/audit_planner_npz.py` で `mode_id_fractions` と終点分布を確認したうえで、**k-means の偏り・恣意的境界**が問題なら `angular` やデータ増、`singleton` で切り分けを検討します。

## 推論デバッグ（ルール vs 教師）

- `viz_val_infer_bev.py --selection teacher` … `mode_id` ヘッドを「選択軌道」として描画（**チートではなく**、教師ヘッド単体の当たりを見る用）。
- `infer_demo.py --selection teacher` … 同様。

本番運用のヘッド選択は引き続き `lib.rule_score`（`--selection rule`）がデフォルトです。
