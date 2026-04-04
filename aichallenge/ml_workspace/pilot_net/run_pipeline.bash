#!/bin/bash
set -e
cd /aichallenge/ml_workspace/pilot_net

BAG_DIR="${1:?Usage: run_pipeline.bash <bag_dir> [image_height] [image_width] [color_space] [output_dim]}"
IMAGE_HEIGHT="${2:-66}"
IMAGE_WIDTH="${3:-200}"
COLOR_SPACE="${4:-yuv}"
OUTPUT_DIM="${5:-2}"

echo "=== Config: ${IMAGE_WIDTH}x${IMAGE_HEIGHT}, ${COLOR_SPACE}, output_dim=${OUTPUT_DIM} ==="

echo "=== 1. Extract data ==="
rm -rf dataset
python3 extract_data_from_bag.py \
    --seq-dirs "$BAG_DIR" \
    --outdir ./dataset/all \
    --image-topic /sensing/camera/image_raw \
    --control-topic /control/command/control_cmd \
    --debug

echo "=== 2. Prepare data ==="
python3 prepare_data.py

echo "=== 3. Train ==="
rm -rf checkpoints logs
trap 'echo "Training interrupted, continuing pipeline..."' INT
python3 train.py \
    model.image_height="$IMAGE_HEIGHT" \
    model.image_width="$IMAGE_WIDTH" \
    model.color_space="$COLOR_SPACE" \
    model.output_dim="$OUTPUT_DIM" \
    train.num_workers=0 \
    +train.loss_type=mse || true
trap - INT

echo "=== 4. Convert weights ==="
python3 convert_weight.py \
    --ckpt ./checkpoints/best_model.pth \
    --output /aichallenge/workspace/src/aichallenge_submit/pilot_net_controller/ckpt/pilotnet_weights.npy \
    --image-height "$IMAGE_HEIGHT" --image-width "$IMAGE_WIDTH" --output-dim "$OUTPUT_DIM"

echo "=== Pipeline complete ==="
