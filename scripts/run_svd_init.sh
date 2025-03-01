#
#1.min or max
#2.rank
#3.basemodel_path
#4.save_root_path
#5.hyperparameter, e.g., llm-adapters
base_model=$1

mkdir -p ./svd_init_models

CUDA_VISIBLE_DEVICES=0 python svd_init.py "min" 64 "$base_model" "./svd_init_models" "LLM-Adapters" &

# can do parallel inits with different rank and min/max