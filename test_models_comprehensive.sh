#!/bin/bash

# 全面测试脚本: Fisher guided NUF和NF量化方法测试
# 测试多个模型在不同数据集和group-size下的困惑度

# 设置基本参数
NSAMPLES=16
SEQLEN=2048
MAXSEQLEN=2048
CUDA_DEVICE=0

# 模型列表
MODELS=(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    "Qwen/Qwen3-14B"  
    "luodian/llama-7b-hf"
    "meta-llama/Meta-Llama-3-8B"
    "meta-llama/Llama-2-7b-hf"
)

# 模型输出目录名映射
MODEL_DIRS=(
    "deepseek_r1_distill_qwen_14b"
    "qwen3_14b"
    "llama_7b_hf"
    "meta_llama3_8b"
    "llama2_7b_hf"
)

# 数据集列表
DATASETS=("wikitext2" "c4")

# Group size列表
GROUP_SIZES=(0 32 128)

# 量化位数列表 (可根据需要调整)
ABITS=(4 2 1)

# 创建日志目录
mkdir -p logs
mkdir -p results

# 日志函数
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a logs/comprehensive_test.log
}

log_message "=== 开始全面测试 ==="
log_message "测试模型数量: ${#MODELS[@]}"
log_message "测试数据集: ${DATASETS[@]}"
log_message "测试Group Sizes: ${GROUP_SIZES[@]}"
log_message "量化位数: ${ABITS[@]}"

# 遍历每个模型
for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    MODEL_DIR="${MODEL_DIRS[$i]}"
    
    log_message "开始测试模型: $MODEL (输出目录: $MODEL_DIR)"
    
    # 遍历每个数据集
    for DATASET in "${DATASETS[@]}"; do
        log_message "  数据集: $DATASET"
        
        # 遍历每个group-size
        for GROUP_SIZE in "${GROUP_SIZES[@]}"; do
            log_message "    Group Size: $GROUP_SIZE"
            
            # 步骤1: 为当前group-size获取Fisher信息
            log_message "      步骤1: 为Group Size $GROUP_SIZE 获取Fisher信息..."
            FISHER_OUTPUT_DIR="${MODEL_DIR}_${DATASET}_gs${GROUP_SIZE}"
            
            CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python gradients/run-fisher.py \
                --model_name_or_path "$MODEL" \
                --output_dir "$FISHER_OUTPUT_DIR" \
                --dataset "$DATASET" \
                --seqlen $SEQLEN \
                --maxseqlen $MAXSEQLEN \
                --num_examples $NSAMPLES \
                --group_size $GROUP_SIZE \
                2>&1 | tee logs/fisher_${MODEL_DIR}_${DATASET}_gs${GROUP_SIZE}.log
                
            if [ $? -ne 0 ]; then
                log_message "      错误: Group Size $GROUP_SIZE 的Fisher信息获取失败，跳过此配置"
                continue
            fi
            
            log_message "      Fisher信息获取完成"
            
            # 遍历每个量化位数
            for ABITS_VAL in "${ABITS[@]}"; do
                log_message "      量化位数: $ABITS_VAL bits"
                
                # 测试NUF方法
                log_message "        测试NUF方法..."
                QUANTIZER_PATH_NUF="quantizers_${MODEL_DIR}_${DATASET}_nuq_gs${GROUP_SIZE}_${ABITS_VAL}bit.pickle"
                
                # 步骤2: 生成NUF量化器
                log_message "          生成NUF量化器..."
                CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python quant/llama_simquant.py "$MODEL" \
                    --abits $ABITS_VAL \
                    --nsamples $NSAMPLES \
                    --seqlen $SEQLEN \
                    --maxseqlen $MAXSEQLEN \
                    --dataset "$DATASET" \
                    --nuq \
                    --fisher "$FISHER_OUTPUT_DIR" \
                    --quantize \
                    --quantizer-path "$QUANTIZER_PATH_NUF" \
                    --group-size $GROUP_SIZE \
                    2>&1 | tee logs/quantize_nuq_${MODEL_DIR}_${DATASET}_gs${GROUP_SIZE}_${ABITS_VAL}bit.log
                
                if [ $? -eq 0 ]; then
                    # 步骤3: 评估NUF模型
                    log_message "          评估NUF模型..."
                    EVAL_LOG_NUF="results/eval_nuq_${MODEL_DIR}_${DATASET}_gs${GROUP_SIZE}_${ABITS_VAL}bit.log"
                    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python quant/llama_simquant.py "$MODEL" \
                        --abits $ABITS_VAL \
                        --nsamples $NSAMPLES \
                        --seqlen $SEQLEN \
                        --maxseqlen $MAXSEQLEN \
                        --dataset "$DATASET" \
                        --nuq \
                        --quantizer-path "$QUANTIZER_PATH_NUF" \
                        --group-size $GROUP_SIZE \
                        2>&1 | tee "$EVAL_LOG_NUF"
                    
                    # 提取困惑度结果
                    PPL_NUF=$(grep -o "Perplexity: [0-9.]*" "$EVAL_LOG_NUF" | tail -1 | cut -d' ' -f2)
                    if [ ! -z "$PPL_NUF" ]; then
                        log_message "          NUF结果 - 困惑度: $PPL_NUF"
                        echo "$MODEL,$DATASET,NUF,$GROUP_SIZE,$ABITS_VAL,$PPL_NUF" >> results/summary.csv
                    fi
                else
                    log_message "          NUF量化器生成失败"
                fi
                
                # 测试NF方法
                log_message "        测试NF方法..."
                QUANTIZER_PATH_NF="quantizers_${MODEL_DIR}_${DATASET}_nf_gs${GROUP_SIZE}_${ABITS_VAL}bit.pickle"
                
                # 步骤2: 生成NF量化器
                log_message "          生成NF量化器..."
                CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python quant/llama_simquant.py "$MODEL" \
                    --abits $ABITS_VAL \
                    --nsamples $NSAMPLES \
                    --seqlen $SEQLEN \
                    --maxseqlen $MAXSEQLEN \
                    --dataset "$DATASET" \
                    --nf \
                    --fisher "$FISHER_OUTPUT_DIR" \
                    --quantize \
                    --quantizer-path "$QUANTIZER_PATH_NF" \
                    --group-size $GROUP_SIZE \
                    2>&1 | tee logs/quantize_nf_${MODEL_DIR}_${DATASET}_gs${GROUP_SIZE}_${ABITS_VAL}bit.log
                
                if [ $? -eq 0 ]; then
                    # 步骤3: 评估NF模型
                    log_message "          评估NF模型..."
                    EVAL_LOG_NF="results/eval_nf_${MODEL_DIR}_${DATASET}_gs${GROUP_SIZE}_${ABITS_VAL}bit.log"
                    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python quant/llama_simquant.py "$MODEL" \
                        --abits $ABITS_VAL \
                        --nsamples $NSAMPLES \
                        --seqlen $SEQLEN \
                        --maxseqlen $MAXSEQLEN \
                        --dataset "$DATASET" \
                        --nf \
                        --quantizer-path "$QUANTIZER_PATH_NF" \
                        --group-size $GROUP_SIZE \
                        2>&1 | tee "$EVAL_LOG_NF"
                    
                    # 提取困惑度结果
                    PPL_NF=$(grep -o "Perplexity: [0-9.]*" "$EVAL_LOG_NF" | tail -1 | cut -d' ' -f2)
                    if [ ! -z "$PPL_NF" ]; then
                        log_message "          NF结果 - 困惑度: $PPL_NF"
                        echo "$MODEL,$DATASET,NF,$GROUP_SIZE,$ABITS_VAL,$PPL_NF" >> results/summary.csv
                    fi
                else
                    log_message "          NF量化器生成失败"
                fi
                
                log_message "      $ABITS_VAL bits 测试完成"
            done
            
            # 删除Fisher文件以节省空间
            log_message "      清理Fisher文件以节省空间..."
            if [ -d "$FISHER_OUTPUT_DIR" ]; then
                rm -rf "$FISHER_OUTPUT_DIR"
                log_message "      已删除 $FISHER_OUTPUT_DIR"
            fi
            
            log_message "    Group Size $GROUP_SIZE 测试完成"
        done
        
        log_message "  数据集 $DATASET 测试完成"
    done
    
    log_message "模型 $MODEL 测试完成"
    log_message "========================"
done

# 生成最终报告
log_message "=== 测试完成，生成最终报告 ==="

# 创建CSV头部（如果不存在）
if [ ! -f results/summary.csv ]; then
    echo "Model,Dataset,Method,GroupSize,Bits,Perplexity" > results/summary.csv
fi

# 生成可读性更好的报告
echo "=== 测试结果汇总 ===" > results/final_report.txt
echo "测试时间: $(date)" >> results/final_report.txt
echo "参数配置:" >> results/final_report.txt
echo "  - nsamples: $NSAMPLES" >> results/final_report.txt
echo "  - seqlen: $SEQLEN" >> results/final_report.txt
echo "  - maxseqlen: $MAXSEQLEN" >> results/final_report.txt
echo "" >> results/final_report.txt

# 按模型分组显示结果
for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    MODEL_DIR="${MODEL_DIRS[$i]}"
    echo "=== $MODEL ===" >> results/final_report.txt
    
    for DATASET in "${DATASETS[@]}"; do
        echo "数据集: $DATASET" >> results/final_report.txt
        grep "$MODEL,$DATASET" results/summary.csv | while IFS=',' read -r model dataset method groupsize bits ppl; do
            echo "  $method, Group-Size: $groupsize, ${bits}bit -> PPL: $ppl" >> results/final_report.txt
        done
        echo "" >> results/final_report.txt
    done
done

log_message "最终报告已生成: results/final_report.txt"
log_message "详细CSV结果: results/summary.csv"
log_message "=== 全部测试完成 ==="

# 显示统计信息
TOTAL_TESTS=$(wc -l < results/summary.csv)
log_message "总测试数量: $((TOTAL_TESTS - 1))" # 减去header行

# 检查是否有失败的测试
FAILED_TESTS=$(find logs -name "*.log" -exec grep -l "Error\|错误\|failed\|Failed" {} \; | wc -l)
if [ $FAILED_TESTS -gt 0 ]; then
    log_message "警告: 发现 $FAILED_TESTS 个失败的测试，请检查日志文件"
fi 