#!/usr/bin/env python3

"""
SFA3D Dual-View Inference Visualization Script
Author: laurent-19
Date: 2025

This script:
- Runs SFA3D inference on both front and back BEV maps
- Measures and displays inference times and FPS for both views
- Visualizes detection results in a combined display window
- Processes frames sequentially from the demo dataset
"""

import sys
import os
import warnings
import time

warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import torch
import numpy as np

src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("sfa"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from data_process.demo_dataset import Demo_KittiDataset
from models.model_utils import create_model
from utils.evaluation_utils import draw_predictions
import config.kitti_config as cnf
from utils.demo_utils import parse_demo_configs, do_detect

if __name__ == '__main__':
    configs = parse_demo_configs()

    model = create_model(configs)
    assert os.path.isfile(configs.pretrained_path), f"No file at {configs.pretrained_path}"
    model.load_state_dict(torch.load(configs.pretrained_path, map_location='cpu'))
    print(f"Loaded weights from {configs.pretrained_path}")

    configs.device = torch.device('cpu' if configs.no_cuda else f'cuda:{configs.gpu_idx}')
    model = model.to(device=configs.device)
    model.eval()

    demo_dataset = Demo_KittiDataset(configs)

    with torch.no_grad():
        for sample_idx in range(len(demo_dataset)):
            # Load the BEV maps and the RGB image
            metadatas, front_bevmap, back_bevmap, img_rgb = demo_dataset.load_bevmap_front_vs_back(sample_idx)

            # Measure front inference time
            t_start_front = time.time()
            front_detections, front_bevmap_out, fps_front = do_detect(
                configs, model, front_bevmap, is_front=True)
            t_end_front = time.time()
            elapsed_front = t_end_front - t_start_front

            # Measure back inference time
            t_start_back = time.time()
            back_detections, back_bevmap_out, fps_back = do_detect(
                configs, model, back_bevmap, is_front=False)
            t_end_back = time.time()
            elapsed_back = t_end_back - t_start_back

            # Convert output maps for visualization
            front_bevmap_vis = (front_bevmap_out.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            front_bevmap_vis = cv2.resize(front_bevmap_vis, (cnf.BEV_WIDTH, cnf.BEV_HEIGHT))
            front_bevmap_vis = draw_predictions(front_bevmap_vis, front_detections, configs.num_classes)
            front_bevmap_vis = cv2.rotate(front_bevmap_vis, cv2.ROTATE_90_COUNTERCLOCKWISE)

            back_bevmap_vis = (back_bevmap_out.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            back_bevmap_vis = cv2.resize(back_bevmap_vis, (cnf.BEV_WIDTH, cnf.BEV_HEIGHT))
            back_bevmap_vis = draw_predictions(back_bevmap_vis, back_detections, configs.num_classes)
            back_bevmap_vis = cv2.rotate(back_bevmap_vis, cv2.ROTATE_90_CLOCKWISE)

            # Combine front and back maps
            combined_bev = np.concatenate((back_bevmap_vis, front_bevmap_vis), axis=1)

            # Show in a window
            cv2.imshow("SFA3D Front + Back Inference", combined_bev)

            # Print DL inference times only
            print(f"Frame {sample_idx} DL Inference:")
            print(f"  Front DL time: {elapsed_front*1000:.2f} ms → {1/elapsed_front:.2f} FPS")
            print(f"  Back DL time: {elapsed_back*1000:.2f} ms → {1/elapsed_back:.2f} FPS")

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    cv2.destroyAllWindows()
