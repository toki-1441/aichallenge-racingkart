# cute_style_rviz_plugin

RViz2 panel plugin that applies a “cute” pink/purple Qt stylesheet to RViz.

Why C++: RViz pluginlib panels are C++/Qt.  
How it stays easy: the **theme is plain `.qss`** and can be **generated/published from Python**.

## What it does

- Adds an RViz panel: `cute_style_rviz_plugin/CuteStylePanel`
- Can apply:
  - Built-in cute theme (fallback)
  - A `.qss` file you select
  - A live-updated `.qss` received on `/cute_style_rviz_plugin/theme_qss` (`std_msgs/String`)

## How to use in RViz

1. Build the workspace.
2. Start RViz.
3. `Panels` → `Add New Panel…` → select `cute_style_rviz_plugin/CuteStylePanel`.
4. Click `Apply Cute Default` or choose `themes/cute_pink_purple.qss`.

If you want an RViz config that already includes this panel and a pink/purple background:

- `aichallenge_system_launch/config/autoware_vehicle_cute.rviz`

## Docker (this repo)

Inside the running container:

- Build: `cd /aichallenge/workspace && colcon build --packages-select cute_style_rviz_plugin`
- Run: `rviz2 -d /aichallenge/workspace/src/aichallenge_system/aichallenge_system_launch/config/autoware_vehicle_cute.rviz`

## Python customization workflow

Generate a QSS from a simple YAML:

- YAML example: `themes/cute_theme.yaml`
- Generate:
  - `ros2 run cute_style_rviz_plugin cute_theme_generate.py --yaml <path>/cute_theme.yaml --out /tmp/cute.qss`

Publish a QSS live to RViz:

- `ros2 run cute_style_rviz_plugin cute_theme_publish.py --qss-file /tmp/cute.qss`

RViz (with the panel open and subscription enabled) will apply it immediately.

Live-reload from YAML (edit the YAML and RViz updates automatically):

- `ros2 run cute_style_rviz_plugin cute_theme_watch.py --yaml <path>/cute_theme.yaml`

## カスタマイズ（日本語）

このプラグインは RViz2 に **Qt Stylesheet（`.qss`）** を適用するだけなので、見た目はかなり自由に調整できます（色、フォント、角丸、余白、ボタンのホバー表現など）。
テーマはテキストファイルのため、C++ を変更して再ビルドしなくても試行錯誤できます。

- `.qss` を直接編集し、パネルの `Apply From File` で適用
- `themes/` に自作テーマ（`.qss`）を追加して管理
- `themes/cute_theme.yaml` を編集し `cute_theme_generate.py` で `.qss` を生成
- `/cute_style_rviz_plugin/theme_qss`（`std_msgs/String`）に QSS を publish してライブ更新（`cute_theme_publish.py` でも可）
- `cute_theme_watch.py` で YAML を監視してライブ更新（色を変えて保存→即反映）
- パネルの `Subscribe theme topic` が有効な間は、受信した QSS が即時反映されます
