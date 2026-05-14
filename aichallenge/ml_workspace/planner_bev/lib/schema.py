"""Contract shapes aligned with design_docs/planner_training_inputs_and_losses.md."""

# bev_scene_stack default grid (see bev_scene_stack.param.yaml)
C_BEV = 4
H_BEV = 256
W_BEV = 144

CHANNEL_NAMES = ("lane", "trajectory", "obstacles", "ego")
