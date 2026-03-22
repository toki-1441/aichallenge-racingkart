# Route Deviation Safety Monitor

Lanelet2 (LL2) マップで定義された走行可能領域から車両の自己位置が逸脱した場合に、経路逸脱フラグ (`Bool`) を publish して安全停止を促すノード。

## 概要

レーシングカート走行中に自己位置推定のずれや制御逸脱が発生し、車両が LL2 レーン外に出た場合を検知する安全監視システムである。

OSM 形式のマップファイル (`route_area.osm`) から lanelet のポリゴンを構築し、自己位置が**いずれの lanelet にも含まれない**場合に即座に逸脱フラグを publish する。

### 動作フロー

```text
/localization/kinematic_state (Odometry)
        |
        v
  0.5Hz 監視タイマー
        |
        v
  lanelet ポリゴン内判定 (ray-casting)
        |
   ┌────┴────┐
   |         |
 レーン内   レーン外
   |         |
 false     true
   |         |
   └────┬────┘
        v
  /vehicle/emergency/is_route_deviation (Bool)
```

## ROS 2 インターフェース

### Subscriptions

| トピック | 型 | 説明 |
|---|---|---|
| `/localization/kinematic_state` | `nav_msgs/msg/Odometry` | 自己位置 |

### Publications

| トピック | 型 | 説明 |
|---|---|---|
| `/vehicle/emergency/is_route_deviation` | `std_msgs/msg/Bool` | `true`: 経路逸脱中 (安全停止要求) |

## 動作仕様

- 監視周期: 0.5 Hz (2 秒間隔)
- 1 回でもレーン外と判定されたら即座に `true` を publish
- レーン内に復帰したら `false` を publish

## マップファイル

`map/route_area.osm` — OSM (XML) 形式の lanelet 定義ファイル。

- `<node>`: `local_x` / `local_y` タグで座標を定義
- `<way>`: ノード列で左右境界線を定義
- `<relation type="lanelet">`: left / right の way を組み合わせてレーンポリゴンを構成

ポリゴンは左境界の点列 + 右境界の逆順で閉じた多角形として構築され、ray-casting 法で点包含判定を行う。
