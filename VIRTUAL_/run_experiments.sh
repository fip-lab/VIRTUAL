#!/bin/bash

# 创建日志目录
mkdir -p logs

# 获取当前时间作为日志文件名
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="logs/experiments_${timestamp}.log"

echo "Starting experiments at $(date)" | tee -a "$log_file"
echo "----------------------------------------" | tee -a "$log_file"

# 运行所有组合的实验，跳过(0,0)和(2,1)
for top_m in {0..3}; do
    for top_n in {0..3}; do
        # 跳过(0,0)和(2,1)组合
        if [ $top_m -eq 0 ] && [ $top_n -eq 0 ]; then
            continue
        fi
        if [ $top_m -eq 2 ] && [ $top_n -eq 1 ]; then
            continue
        fi
        
        echo "Running experiment with top_m=$top_m, top_n=$top_n" | tee -a "$log_file"
        echo "Start time: $(date)" | tee -a "$log_file"
        
        # 运行实验并记录输出
        python virtual_ensemble.py $top_m $top_n 2>&1 | tee -a "$log_file"
        
        echo "End time: $(date)" | tee -a "$log_file"
        echo "----------------------------------------" | tee -a "$log_file"
    done
done

echo "All experiments completed at $(date)" | tee -a "$log_file"