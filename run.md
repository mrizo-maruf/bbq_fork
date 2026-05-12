python benchmark_tracking.py --config_path examples/configs/isaac/franka_cab_dex_more.yaml --iou_threshold 0.25 --matcher hungarian --limit 100        # smoke test first; drop for a full run

python benchmark_tracking.py --config_path examples/configs/isaac/franka_cab_dex_more.yaml --dataset_root /path/to/IsaacSimData --iou_threshold 0.25 --matcher hungarian

python visualize_tracking.py --config_path examples/configs/isaac/franka_cab_dex_more.yaml --dataset_root /path/to/IsaacSimData --fps 10 --alpha 0.5