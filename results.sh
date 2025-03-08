ckpt_dir=/home/ubuntu/qujunlong/txyy/output/checkpoint/convnext_tiny_224/20241207_090334
model=$(basename $(dirname "$ckpt_dir"))

for file in "$ckpt_dir"/*.pth; do
    # Check if any files match the pattern
    if [ -f "$file" ]; then
        base_file=$(realpath "$file")
        python3 visu/features_extract.py --model $model --finetune $base_file --gpu_id 0 &
        python3 visu/results_analysis.py --model $model --finetune $base_file --gpu_id 1
    fi
done