# This is the eval script for all the local models, so basiclly
# anything that's covered by Unsloth (so not OpenAI models etc. )

# imports etc.
import re
import sys
from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported
from datasets import Dataset, load_dataset
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams
from pandas import *
import pandas as pd
from genesis_env import *
import pyrallis
from dataclasses import asdict, dataclass
from tqdm import tqdm
def find_between(s, start, end):
    plan = ''
    try:
        plan = s.split(start)[1].split(end)[0]
    except:
        plan = ''
    return plan
def genesis_reward_func_local(env,prompt, completion, answer, **kwargs) -> list[float]:
    """Reward function that gets signal from Genesis simulator

    Args:
        prompts (_type_): prompts/questions as strings
        completions (_type_): Given plan in text
        answer (_type_): correct answer dictionary from questions

    Returns:
        list[float]: list of rewards for each prompt etc. 
    """
    env.reset(task_dictionary=answer)
    llm_plan = find_between(completion[0]["content"],'<answer>','</answer>')
    #print(llm_plan)
    reward = env.execute_llm_plan(llm_plan)
        
    return reward

def gsmk8_correctness_reward_func_local(prompt, completion, answer, **kwargs) -> list[float]:
    response = completion[0]['content']
    extracted_response = extract_xml_answer(response)
    return 1.0 if extracted_response == answer else 0.0

@dataclass
class LLMEvalConfig:
    ### Model definition
    base_model: str = 'Qwen/Qwen2.5-7B-Instruct'
    use_lora: bool = False
    lora_name: str = ''

    # Inference Parameters
    temperature: float = 0.8
    top_p: float = 0.95
    max_tokens: int = 4096
    
    # dataset stuf
    dataset_name: str = 'GSMK8' # can be tiehr GSMK8 or GENESIS

    def __post_init__(self):
        pass

@pyrallis.wrap()
def eval(config: LLMEvalConfig):
    model_name = config.base_model

    max_seq_length = config.max_tokens  # Can increase for longer reasoning traces
    lora_rank = 128  # Larger rank = smarter, but slower


    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=False,  # False for LoRA 16bit
        fast_inference=True,  # Enable vLLM fast inference
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.75,  # Reduce if out of memory
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ], # Remove QKVO if out of memory
        lora_alpha = lora_rank,
        use_gradient_checkpointing = "unsloth", # Enable long context finetuning
        random_state = 3407,
    )

    # loads dataset
    dataset = ''
    if config.dataset_name == "GSMK8":
        dataset = get_gsm8k_questions()
    elif config.dataset_name == 'GENESIS':
        dataset = get_manipulation_questions()
        env = FrankaManipEnv(render_video=False)
    else:
        print('Unknown Dataset!')

    # iterates through questions
    total_correct = 0
    for question in tqdm(dataset):
        prompts = question['prompt']
        text = tokenizer.apply_chat_template(
            prompts,
            tokenize=False,
            add_generation_prompt=True,
        )


        sampling_params = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
        )
        if config.use_lora:
            output = (
                model.fast_generate(
                    text,
                    sampling_params=sampling_params,
                    lora_request=model.load_lora(config.lora_name),
                    use_tqdm=False
                )[0]
                .outputs[0]
                .text
            )
        else:
            output = (
                model.fast_generate(
                    text,
                    sampling_params=sampling_params,
                    use_tqdm=False,
                )[0]
                .outputs[0]
                .text
            )

        # TESTING prints output
        print(output)
        # next line is for some stupid dumb thing 
        # to fit the formatting with the existing eval functions
        completion = [{'content':output}]
        # validates question
        if config.dataset_name == "GSMK8":
            total_correct += gsmk8_correctness_reward_func_local(prompts,completion,question['answer'])
        elif config.dataset_name == 'GENESIS':
            total_correct += genesis_reward_func_local(env,prompts,completion,question['answer'])
    print('*** Eval || Model:',config.base_model,'Lora:',config.use_lora,'Dataset:',config.dataset_name,'***')
    print('Total Correct:', total_correct,'out of ',len(dataset))
    print('Percent Correct:',total_correct/len(dataset))
if __name__ == "__main__":
    eval()