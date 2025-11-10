#!/usr/bin/env python3
"""Monitor system resources during training."""

import time
import psutil
import datetime
import argparse

def monitor_resources(duration_minutes=30, interval_seconds=5):
    """Monitor CPU and memory usage."""
    print(f"Starting resource monitoring for {duration_minutes} minutes...")
    print(f"CPU cores: {psutil.cpu_count()}")
    print(f"Total memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print()
    
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)
    
    try:
        while time.time() < end_time:
            # Get current stats
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            
            print(f"[{timestamp}] CPU: {cpu_percent:5.1f}% | "
                  f"Memory: {memory.percent:5.1f}% "
                  f"({memory.used/(1024**3):.1f}/{memory.total/(1024**3):.1f} GB) | "
                  f"Available: {memory.available/(1024**3):.1f} GB")
            
            # Warning if resources are getting high
            if cpu_percent > 90:
                print("⚠️  WARNING: CPU usage very high!")
            if memory.percent > 90:
                print("⚠️  WARNING: Memory usage very high!")
            
            time.sleep(interval_seconds - 1)  # Subtract the 1 second from cpu_percent
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")
    
    print("Monitoring complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor system resources")
    parser.add_argument("--duration", type=int, default=30, help="Duration in minutes")
    parser.add_argument("--interval", type=int, default=5, help="Check interval in seconds")
    args = parser.parse_args()
    
    monitor_resources(args.duration, args.interval)