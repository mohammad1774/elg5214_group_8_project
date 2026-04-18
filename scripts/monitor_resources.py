#!/usr/bin/env python3
"""
Resource monitoring script for RL training
Logs GPU, CPU, and memory usage to a CSV file
Usage: python scripts/monitor_resources.py [--interval 5] [--output monitor.csv]
"""

import argparse
import csv
import time
import subprocess
import psutil
from datetime import datetime
from pathlib import Path


def get_gpu_stats():
    """Get GPU memory and utilization using nvidia-smi"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu,utilization.memory', 
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split()
            return {
                'gpu_mem_used': float(parts[0]),
                'gpu_mem_total': float(parts[1]),
                'gpu_util': float(parts[2]),
                'gpu_mem_util': float(parts[3]),
            }
    except Exception as e:
        print(f"Error getting GPU stats: {e}")
    
    return {
        'gpu_mem_used': 0,
        'gpu_mem_total': 0,
        'gpu_util': 0,
        'gpu_mem_util': 0,
    }


def get_system_stats():
    """Get CPU and RAM usage"""
    return {
        'cpu_percent': psutil.cpu_percent(interval=0.5),
        'ram_used': psutil.virtual_memory().used / (1024**3),  # GB
        'ram_total': psutil.virtual_memory().total / (1024**3),  # GB
        'ram_percent': psutil.virtual_memory().percent,
    }


def count_training_processes():
    """Count active training processes"""
    try:
        result = subprocess.run(
            ['pgrep', '-f', 'python.*test_dqn'],
            capture_output=True, text=True
        )
        return len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
    except:
        return 0


def monitor(output_file, interval):
    """Main monitoring loop"""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = [
        'timestamp', 'elapsed_seconds',
        'gpu_mem_used_mb', 'gpu_mem_total_mb', 'gpu_mem_percent',
        'gpu_utilization_percent', 'gpu_memory_utilization_percent',
        'cpu_percent', 'ram_used_gb', 'ram_total_gb', 'ram_percent',
        'training_processes'
    ]
    
    start_time = time.time()
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        print(f"Monitoring started. Logging to: {output_file}")
        print(f"Update interval: {interval}s")
        print()
        
        try:
            while True:
                elapsed = time.time() - start_time
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                gpu_stats = get_gpu_stats()
                sys_stats = get_system_stats()
                n_procs = count_training_processes()
                
                gpu_mem_percent = (gpu_stats['gpu_mem_used'] / max(gpu_stats['gpu_mem_total'], 1)) * 100 if gpu_stats['gpu_mem_total'] > 0 else 0
                
                row = {
                    'timestamp': timestamp,
                    'elapsed_seconds': int(elapsed),
                    'gpu_mem_used_mb': f"{gpu_stats['gpu_mem_used']:.0f}",
                    'gpu_mem_total_mb': f"{gpu_stats['gpu_mem_total']:.0f}",
                    'gpu_mem_percent': f"{gpu_mem_percent:.1f}",
                    'gpu_utilization_percent': f"{gpu_stats['gpu_util']:.1f}",
                    'gpu_memory_utilization_percent': f"{gpu_stats['gpu_mem_util']:.1f}",
                    'cpu_percent': f"{sys_stats['cpu_percent']:.1f}",
                    'ram_used_gb': f"{sys_stats['ram_used']:.2f}",
                    'ram_total_gb': f"{sys_stats['ram_total']:.2f}",
                    'ram_percent': f"{sys_stats['ram_percent']:.1f}",
                    'training_processes': n_procs,
                }
                
                writer.writerow(row)
                f.flush()
                
                # Print to console
                status = (
                    f"[{timestamp}] "
                    f"GPU: {row['gpu_mem_percent']}% mem ({row['gpu_mem_used_mb']}MB), "
                    f"{row['gpu_utilization_percent']}% util | "
                    f"CPU: {row['cpu_percent']}% | "
                    f"RAM: {row['ram_percent']}% ({row['ram_used_gb']}GB) | "
                    f"Processes: {n_procs}"
                )
                print(status)
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
            print(f"Results saved to: {output_path.absolute()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Monitor GPU and CPU resources during training')
    parser.add_argument('--interval', type=int, default=5, help='Update interval in seconds')
    parser.add_argument('--output', type=str, default='monitor.csv', help='Output CSV file')
    args = parser.parse_args()
    
    monitor(args.output, args.interval)
