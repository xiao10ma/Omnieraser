#!/bin/bash

export WANDB_MODE=offline

# 定义要测试的rank和learning rate数组
ranks=(1 2 4 8 16)
learning_rates=(3e-4 3e-5 3e-6)

# 创建实验输出的基础目录
base_output_dir="output/flux_control_lora_grid_search_deepseed"
mkdir -p $base_output_dir

# 遍历所有组合进行训练
for rank in "${ranks[@]}"; do
    for lr in "${learning_rates[@]}"; do
        # 为每个实验创建唯一的输出目录和实验名称
        experiment_name="rank_${rank}_lr_${lr}"
        output_dir="${base_output_dir}/${experiment_name}"
        mkdir -p $output_dir
        
        # 定义日志文件路径
        log_file="${output_dir}/training.log"
        
        echo "Starting experiment with rank=${rank}, learning_rate=${lr}"
        
        # 设置WANDB_RUN_NAME环境变量
        export WANDB_RUN_NAME="${experiment_name}"
        
        # 将所有输出重定向到日志文件，同时显示在终端
        {
            echo "=== 实验配置 ===" 
            echo "开始时间: $(date)"
            echo "Rank: ${rank}"
            echo "Learning Rate: ${lr}"
            echo "WANDB Run Name: ${experiment_name}"
            echo "==================="
            echo
            
            accelerate launch --config_file accelerate.yaml finetune_control_lora_flux.py \
                --pretrained_model_name_or_path black-forest-labs/FLUX.1-dev \
                --output_dir "${output_dir}" \
                --group_name "deepseed-zero-stage-3" \
                --train_data_dir dataset \
                --resolution 512 \
                --guidance_scale 3.5 \
                --learning_rate ${lr} \
                --train_batch_size 1 \
                --num_train_epochs 100 \
                --rank ${rank} \
                --gaussian_init_lora \
                --logging_dir wandb \
                --tracker_project_name "flux_train_grid_search" \
                --report_to wandb \
                --validation_steps 10 \
                --checkpointing_steps 10 \
                --use_8bit_adam \
                --allow_tf32 \
                --scale_lr \
                --lr_scheduler cosine_with_restarts \
                --mixed_precision "bf16" \
                --gradient_accumulation_steps 8
            
            echo
            echo "结束时间: $(date)"
            echo "实验完成: rank=${rank}, learning_rate=${lr}"
        } 2>&1 | tee "${log_file}"
        
    done
done 