#!/bin/bash
# Real-time resource monitoring during training
# Usage: bash scripts/monitor_resources.sh [LOG_FILE]

LOG_FILE="${1:-monitor.log}"

echo "Starting resource monitoring - logging to: $LOG_FILE"
echo "Press Ctrl+C to stop"
echo

# Clear previous log
> "$LOG_FILE"

# Header
{
    echo "Timestamp | GPU_Mem(%) | GPU_Util(%) | CPU_Util(%) | RAM_Used(GB) | RAM_Util(%) | Processes"
    echo "-----------|-----------|-----------|-----------|-----------|-----------|----------"
} | tee -a "$LOG_FILE"

while true; do
    timestamp=$(date '+%H:%M:%S')
    
    # GPU Memory and Utilization
    gpu_info=$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits)
    gpu_mem_used=$(echo "$gpu_info" | awk '{print $1}')
    gpu_mem_total=$(echo "$gpu_info" | awk '{print $2}')
    gpu_util=$(echo "$gpu_info" | awk '{print $3}')
    
    gpu_mem_percent=$(awk "BEGIN {printf \"%.1f\", ($gpu_mem_used / $gpu_mem_total) * 100}")
    
    # CPU and RAM
    cpu_info=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}')
    ram_info=$(free -b | awk 'NR==2{printf "%.2f %.1f", $3/1024/1024/1024, ($3/$2)*100}')
    
    # Count Python processes (training processes)
    python_procs=$(pgrep -f "python.*test_dqn" | wc -l)
    
    # Output
    output=$(printf "%s | %6.1f%% | %6.1f%% | %6.1f%% | %6s GB | %6s%% | %d" \
        "$timestamp" "$gpu_mem_percent" "$gpu_util" "$cpu_info" "$ram_info" "$python_procs")
    
    echo "$output" | tee -a "$LOG_FILE"
    
    sleep 5  # Update every 5 seconds
done
