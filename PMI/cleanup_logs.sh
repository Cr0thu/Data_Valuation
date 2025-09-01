#!/bin/bash

# 清理脚本：删除旧的日志文件
# 保留最近7天的文件，删除更早的文件

echo "开始清理旧的日志文件..."

# 清理experiment_logs目录中的旧文件
if [ -d "experiment_logs" ]; then
    echo "清理 experiment_logs 目录中的旧文件..."
    
    # 删除7天前的successful_processes文件
    find experiment_logs/ -name "successful_processes_*.txt" -mtime +7 -delete 2>/dev/null
    
    # 删除7天前的实验日志文件
    find experiment_logs/ -name "exp_*.log" -mtime +7 -delete 2>/dev/null
    find experiment_logs/ -name "experiment_*.log" -mtime +7 -delete 2>/dev/null
    
    echo "experiment_logs 清理完成"
else
    echo "experiment_logs 目录不存在"
fi

# 清理experiment_results目录中的旧文件
if [ -d "experiment_results_*" ]; then
    echo "清理 experiment_results 目录中的旧文件..."
    
    # 删除7天前的实验结果目录
    find . -maxdepth 1 -name "experiment_results_*" -type d -mtime +7 -exec rm -rf {} \; 2>/dev/null
    
    echo "experiment_results 清理完成"
else
    echo "没有找到 experiment_results 目录"
fi

# 清理根目录中的临时文件
echo "清理根目录中的临时文件..."

# 删除7天前的output_copy文件
find . -maxdepth 1 -name "output_copy_*.txt" -mtime +7 -delete 2>/dev/null

# 删除7天前的successful_processes文件（如果还有的话）
find . -maxdepth 1 -name "successful_processes_*.txt" -mtime +7 -delete 2>/dev/null

echo "清理完成！"
echo "当前日志文件统计："
echo "  experiment_logs 目录中的文件数量: $(find experiment_logs/ -type f 2>/dev/null | wc -l)"
echo "  experiment_results 目录数量: $(find . -maxdepth 1 -name "experiment_results_*" -type d 2>/dev/null | wc -l)"