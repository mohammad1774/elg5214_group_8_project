#!/bin/bash
# Quick resource availability check

echo "============================================"
echo "  System Resource Check"
echo "============================================"
echo

echo "1. GPU Status:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free,utilization.gpu,temperature.gpu \
    --format=table
echo

echo "2. CPU and Memory:"
echo "  CPU Cores: $(nproc)"
echo "  CPU Usage: $(top -bn1 | grep 'Cpu(s)' | sed 's/.*, *\([0-9.]*\)%* id.*/\1/' | awk '{print 100 - $1}')%"
echo
free -h | tail -2
echo

echo "3. Storage:"
df -h | grep -E '(Filesystem|/$|/lustre)'
echo

echo "4. Active Processes:"
echo "  Python processes: $(pgrep python | wc -l)"
echo "  Training processes: $(pgrep -f 'test_dqn' | wc -l)"
echo

echo "============================================"
echo "  ✓ Check complete"
echo "============================================"
