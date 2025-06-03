### Manipulation Dataset Scripts

# Eval for base Qwen2.5 7B Instruct model on manipulation
python run_eval_local_model.py \
--base_model 'Qwen/Qwen2.5-7B-Instruct' \
--use_lora False \
--dataset_name 'GENESIS'

# Eval for base Qwen2.5 7B RL post-trained model on manipulation
python run_eval_local_model.py \
--base_model 'Qwen/Qwen2.5-7B-Instruct' \
--use_lora True \
--lora_name 'grpo_saved_lora_2' \
--dataset_name 'GENESIS'

# Eval for R1-reasoning distilled Qwen 7B RL post-trained model on manipulation
python run_eval_local_model.py \
--base_model 'unsloth/DeepSeek-R1-Distill-Qwen-7B' \
--use_lora False \
--dataset_name 'GENESIS'


### Math Dataset Scripts

# Eval for base Qwen2.5 7B Instruct model on manipulation
python run_eval_local_model.py \
--base_model 'Qwen/Qwen2.5-7B-Instruct' \
--use_lora False \
--dataset_name 'GSMK8'

# Eval for base Qwen2.5 7B RL post-trained model on manipulation
python run_eval_local_model.py \
--base_model 'Qwen/Qwen2.5-7B-Instruct' \
--use_lora True \
--lora_name 'grpo_saved_lora_2' \
--dataset_name 'GSMK8'

# Eval for R1-reasoning distilled Qwen 7B RL post-trained model on manipulation
python run_eval_local_model.py \
--base_model 'unsloth/DeepSeek-R1-Distill-Qwen-7B' \
--use_lora False \
--dataset_name 'GSMK8'