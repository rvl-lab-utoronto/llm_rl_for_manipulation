# This is the eval script for all the local models, so basiclly
# anything that's covered by Unsloth (so not OpenAI models etc. )

# imports etc.
import re
import sys
#from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported
from datasets import Dataset, load_dataset
from trl import GRPOConfig, GRPOTrainer
#from vllm import SamplingParams
from pandas import *
import pandas as pd
from genesis_env import *
import pyrallis
from dataclasses import asdict, dataclass
from tqdm import tqdm
import timeit
from openai import OpenAI
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

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
    llm_plan = find_between(completion,'<answer>','</answer>')
    #print(llm_plan)
    reward = env.execute_llm_plan(llm_plan)
        
    return reward

def gsmk8_correctness_reward_func_local(prompt, completion, answer, **kwargs) -> list[float]:
    response = completion
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

client = OpenAI()

@pyrallis.wrap()
def eval(config: LLMEvalConfig):
    model_name = config.base_model

    max_seq_length = config.max_tokens  # Can increase for longer reasoning traces
    lora_rank = 128  # Larger rank = smarter, but slower

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

    models = [
        'o3-mini',
        'gpt-4o-mini',
        'gpt-3.5-turbo',
        'gpt-4o',
    ]

    results_list = []

    def call_openai_gsmk8_parallel(run):
        start_time = timeit.default_timer()
        completion = client.chat.completions.create(
            model=model,
            messages=prompts
        )
        run_time = timeit.default_timer() - start_time
        response = completion.choices[0].message.content

        if config.dataset_name == "GSMK8":
            reward = gsmk8_correctness_reward_func_local(prompts, response, question['answer'])

        return [question_count, model, run, prompts[-1], response, reward, run_time]

    for model in models:
        question_count = 0
        for question in tqdm(dataset):
            question_count += 1
            prompts = question['prompt']

            if config.dataset_name == "GSMK8":
                with ThreadPoolExecutor(max_workers=3) as executor:
                    futures = [executor.submit(call_openai_gsmk8_parallel, run) for run in range(3)]

                    for future in as_completed(futures):
                        result = future.result()
                        results_list.append(result)
                
            elif config.dataset_name == 'GENESIS':
                for run in range(3):
                    start_time = timeit.default_timer()
                    completion = client.chat.completions.create(
                        model=model,
                        messages=prompts
                    )
                    run_time = timeit.default_timer() - start_time
                    response = completion.choices[0].message.content
                    reward = genesis_reward_func_local(env,prompts,response,question['answer'])
                    results_list.append([question_count, model, run, prompts[-1], response, reward, run_time])
            
            pd.DataFrame(results_list, columns=['scenario_idx', 'model', 'question', 'run', 'answer', 'reward', 'run_time']).to_csv('./openai_results_gsmk8.csv')
    
    print('*** Eval || Model:',config.base_model,'Lora:',config.use_lora,'Dataset:',config.dataset_name,'***')
    print('Total Correct:', total_correct,'out of ',len(dataset))
    print('Percent Correct:',total_correct/len(dataset))
if __name__ == "__main__":
    eval()