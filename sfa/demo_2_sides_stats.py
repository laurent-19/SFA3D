#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SFA3D Inference + Jetson Hardware Stats Logging Script
Author: laurent-19
Date: 2025

This script:
- runs SFA3D inference on front + back BEV maps
- measures pure DL inference FPS
- uses jtop py api to log Jetson hardware stats
- logs power, GPU/CPU load, temps from Jetson hardware
- saves all results to a CSV for reporting
"""

import sys
import os
import warnings
import time
import pandas as pd
import subprocess
from datetime import datetime

warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import torch
import numpy as np

from jtop import jtop

# --- Import SFA3D modules ---
src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("sfa"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from data_process.demo_dataset import Demo_KittiDataset
from models.model_utils import create_model
import config.kitti_config as cnf
from utils.demo_utils import parse_demo_configs, do_detect

def set_jetson_clocks(enable=True):
    """Enable or disable Jetson maximum performance clocks"""
    try:
        if enable:
            print("Setting Jetson to maximum performance clocks...")
            result = subprocess.run(['sudo', 'jetson_clocks'], 
                                  capture_output=True, text=True, timeout=10)
        else:
            print("Disabling Jetson maximum performance clocks...")
            result = subprocess.run(['sudo', 'jetson_clocks', '--restore'], 
                                  capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            status = "enabled" if enable else "disabled"
            print(f"Jetson clocks successfully {status}")
            return True
        else:
            print(f"Failed to set jetson_clocks: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("Timeout while setting jetson_clocks")
        return False
    except FileNotFoundError:
        print("jetson_clocks command not found. Make sure you're running on a Jetson device.")
        return False
    except Exception as e:
        print(f"Error setting jetson_clocks: {e}")
        return False

def get_jetson_clock_status():
    """Check current jetson_clocks status"""
    try:
        result = subprocess.run(['sudo', 'jetson_clocks', '--show'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return "Unknown"
    except:
        return "Unable to check"

if __name__ == '__main__':
    configs = parse_demo_configs()

    # Set Jetson to maximum performance before benchmarking (if enabled)
    jetson_clocks_set = False
    if configs.jclock:
        print("Jetson clock control enabled via --jclock flag")
        print("Checking Jetson clock status...")
        clock_status = get_jetson_clock_status()
        print(f"Current clock status: {clock_status}")
        
        # Enable maximum performance clocks
        jetson_clocks_set = set_jetson_clocks(enable=True)
    else:
        print("Jetson clock control disabled (use --jclock to enable)")
    
    try:
        # Load model
        model = create_model(configs)
        assert os.path.isfile(configs.pretrained_path), f"No file at {configs.pretrained_path}"
        model.load_state_dict(torch.load(configs.pretrained_path, map_location='cpu'))
        print(f"Loaded weights from {configs.pretrained_path}")

        # Choose device
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        if cuda_available and not configs.no_cuda:
            print(f"CUDA is available. Using GPU {configs.gpu_idx}")
            configs.device = torch.device(f'cuda:{configs.gpu_idx}')
        else:
            print("CUDA is not available or disabled. Using CPU")
            configs.device = torch.device('cpu')
            
        model = model.to(device=configs.device)
        model.eval()

        # Prepare dataset
        demo_dataset = Demo_KittiDataset(configs)

        # Prepare CSV log with timestamp
        csv_rows = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = os.path.join(configs.results_dir, f"benchmark_results_{timestamp}.csv")
        print_true = False

        with jtop() as jetson:
            if not jetson.ok():
                print("Could not connect to jtop. Exiting.")
                sys.exit(1)

            print("\nStarting SFA3D benchmark...\n")

            t_total_start = time.time()

            for sample_idx in range(len(demo_dataset)):
                # --- Load Data ---
                metadatas, front_bevmap, back_bevmap, img_rgb = demo_dataset.load_bevmap_front_vs_back(sample_idx)
                
                # --- Front Inference ---
                front_detections, front_bevmap_out, front_fps_calc = do_detect(
                    configs, model, front_bevmap, is_front=True)

                # --- Back Inference ---
                back_detections, back_bevmap_out, back_fps_calc = do_detect(
                    configs, model, back_bevmap, is_front=False)

                # --- Jetson Hardware Stats ---
                stats = jetson.stats

                # Power
                power_W = stats.get('Power TOT', None)

                # GPU load
                gpu_load = None
                if 'GPU' in stats and stats['GPU']:
                    gpu_load = stats.get('GPU', None)

                # Temps
                temp_gpu = stats.get('Temp gpu', None)
                temp_cpu = stats.get('Temp cpu', None)
                temp_tj = stats.get('Temp tj', None)

                # RAM
                ram_used = None
                if 'RAM' in stats and stats['RAM']:
                    ram_used = stats.get('RAM', None)

                # CPU loads
                cpu_loads = {}
                for i in range(1, 7):
                    cpu_key = f'CPU{i}'
                    cpu_loads[cpu_key] = stats.get(cpu_key, {})

                if print_true:
                    # Print frame summary
                    print(f"\nFrame {sample_idx}")
                    print(f"  Front DL FPS: {front_fps_calc:.2f}")
                    print(f"  Back DL FPS: {back_fps_calc:.2f}")
                    print(f"  Power TOT: {power_W} W")
                    print(f"  GPU Load: {gpu_load} %")
                    print(f"  Temp GPU: {temp_gpu} °C | Temp CPU: {temp_cpu} °C | Temp TJ: {temp_tj} °C")
                    print(f"  RAM Used: {ram_used} MB")
                    for k, v in cpu_loads.items():
                        print(f"  {k}: {v} %")

                # --- Save to CSV ---
                row = {
                    "Frame": sample_idx,
                    "Front_DL_FPS": front_fps_calc,
                    "Back_DL_FPS": back_fps_calc,
                    "Power_W": power_W,
                    "GPU_Load_%": gpu_load,
                    "Temp_GPU_C": temp_gpu,
                    "Temp_CPU_C": temp_cpu,
                    "Temp_TJ_C": temp_tj,
                    "RAM_MB": ram_used
                }
                for cpu_key, load in cpu_loads.items():
                    row[f"{cpu_key}_Load_%"] = load

                csv_rows.append(row)

            # Save CSV
            t_total_end = time.time()
            total_runtime = t_total_end - t_total_start

            df = pd.DataFrame(csv_rows)
            df.to_csv(csv_filename, index=False)
            print(f"\nTotal Runtime: {total_runtime:.2f} seconds")
            print(f"Logged results saved to: {csv_filename}")
            print(f"Processed {len(csv_rows)} frames total")
    
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        print(f"\nError during benchmark: {e}")
    finally:
        # Restore original clock settings (only if they were changed)
        if jetson_clocks_set:
            print("\nRestoring original Jetson clock settings...")
            set_jetson_clocks(enable=False)
        elif configs.jclock:
            print("\nJetson clocks were not successfully set, no restoration needed")
