
#!/bin/bash
set -euo pipefail

model_names="${MODEL_NAMES:-mamba_mil}"
backbones="${BACKBONES:-dinov2}"

declare -A in_dim
in_dim["CvT"]=384
in_dim["inception_v3"]=2048
in_dim["resnet50"]=2048
in_dim["dinov2"]=768
in_dim["revvit"]=1536
in_dim["twinssvt"]=768
in_dim["twinsvssm"]=1024
in_dim["vit"]=768
in_dim["swin"]=1024
in_dim["swinv2"]=1024
in_dim["VSSM"]=1024
in_dim["VSSMV2"]=1024

task="Lymph"
data_root="${DATA_ROOT:-./Lymph}"
results_dir="${RESULTS_DIR:-./results/$task}"
preloading="${PRELOADING:-no}"
patch_size="${PATCH_SIZE:-512}"
lr="${LR:-2e-4}"
mambamil_rate="${MAMBAMIL_RATE:-5}"
mambamil_layer="${MAMBAMIL_LAYER:-2}"
mambamil_type="${MAMBAMIL_TYPE:-SRMamba}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

for model in $model_names
do
    for backbone in $backbones
    do
        exp="${model}/${backbone}"
        echo "${exp}, GPU is: ${CUDA_VISIBLE_DEVICES}"
        k_start=-1
        k_end=-1
        python main.py \
            --data_root_dir "$data_root" \
            --drop_out 0 \
            --lr "$lr" \
            --k 1 \
            --k_start "$k_start" \
            --k_end "$k_end" \
            --label_frac 1.0 \
            --exp_code "$exp" \
            --patch_size "$patch_size" \
            --task "$task" \
            --backbone "$backbone" \
            --results_dir "$results_dir" \
            --model_type "$model" \
            --log_data \
            --split_dir "./splits/Lymph_100" \
            --in_dim "${in_dim[$backbone]}" \
            --preloading "$preloading" \
            --mambamil_rate "$mambamil_rate" \
            --mambamil_layer "$mambamil_layer" \
            --mambamil_type "$mambamil_type"
    done
done
