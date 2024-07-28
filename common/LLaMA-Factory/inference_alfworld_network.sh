#!/bin/bash

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 device_num model_path adapter_path template"
    exit 1
fi

CUDA_VISIBLE_DEVICES=$1 python ./src/alfworld_inference_server.py \
    --model_name_or_path $2 \
    --adapter_name_or_path $3 \
    --template $4 \
    --finetuning_type lora


# checkpoint-17659
# CUDA_VISIBLE_DEVICES=0 python ./src/alfworld_inference_server.py \
#     --model_name_or_path /dev/model_path/gemma-7b \
#     --adapter_name_or_path /root/project/model_path/self-rectify/checkpoints/gemma_alfworld/checkpoint-2156 \
#     --template gemma_muep \
#     --finetuning_type lora

# CUDA_VISIBLE_DEVICES=5 python ./src/alfworld_inference_server.py \
#     --model_name_or_path /dev/model_path/Mistral-7B-Instruct-v0.2 \
#     --adapter_name_or_path /root/project/model_path/self-rectify/checkpoints/mistral_alfworld/checkpoint-2156 \
#     --template mistral_muep \
#     --finetuning_type lora

# CUDA_VISIBLE_DEVICES=0 python ./src/alfworld_inference_server.py \
#     --model_name_or_path /dev/model_path/bloomz-7b1 \
#     --adapter_name_or_path /root/project/model_path/self-rectify/checkpoints/bloomz/checkpoint-5886 \
#     --template llama2_muep \
#     --finetuning_type lora

# CUDA_VISIBLE_DEVICES=1 python ./src/alfworld_inference_server.py \
#     --model_name_or_path /dev/model_path/Llama-2-7b-hf \
#     --adapter_name_or_path /root/project/model_path/self-rectify/checkpoints/llama2_alfworld/checkpoint-2156 \
#     --template llama2_muep \
#     --finetuning_type lora

# CUDA_VISIBLE_DEVICES=7 python ./src/alfworld_inference_server.py \
#     --model_name_or_path /dev/model_path/vicuna-7b-v1.5 \
#     --adapter_name_or_path /root/project/model_path/self-rectify/checkpoints/vicuna/checkpoint-5886 \
#     --template vicuna_muep \
#     --finetuning_type lora