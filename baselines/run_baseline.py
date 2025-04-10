import ast
import re
import sys
import timeit

sys.path.append('../')
from datasets import Dataset, load_dataset
from pandas import *
from genesis_env import FrankaManipEnv
from openai import OpenAI
import numpy as np
import pandas as pd

from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

# starts simulator 
env = FrankaManipEnv(render_video=False, show_viewer=False)

# Load and prep dataset
with open("../prompts/base_manip_prompt.txt","r") as f:
    SYSTEM_PROMPT = f.read()

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def get_manipulation_questions(path = '../data/manipulation_tasks_full.xlsx'):
    # makes the initial Dataset
    xls = ExcelFile(path)
    df = xls.parse(xls.sheet_names[0])
    print(df.head())
    questions = df['Text Question'].to_list()
    rcg = df['red_cube_goal'].to_list()
    bcg = df['blue_cube_goal'].to_list()
    ycg = df['yellow_cube_goal'].to_list()
    gcg = df['green_cube_goal'].to_list()
    categories = df['category'].to_list()
    answers = []
    for i in range(len(rcg)):
        answers.append({'red_cube_goal':rcg[i],
                        'blue_cube_goal':bcg[i],
                        'yellow_cube_goal':ycg[i],
                        'green_cube_goal':gcg[i]})
    data = Dataset.from_dict({'question':questions,'answer':answers, 'category': categories})

    data = data.map(
        lambda x: {  # type: ignore
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": x["question"]},
            ],
            "answer":(x["answer"]),
            "category": (x['category'])
        }
    )  # type: ignore
    return data  # type: ignore

# Reward functions
def genesis_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """Reward function that gets signal from Genesis simulator

    Args:
        prompts (_type_): prompts/questions as strings
        completions (_type_): Given plan in text
        answer (_type_): correct answer dictionary from questions

    Returns:
        list[float]: list of rewards for each prompt etc. 
    """
    rewards = []
    for prompt,completion,goal in zip(prompts,completions,answer):
        env.reset(goal_location=goal)
        reward = env.execute_llm_plan(completion)
        rewards.append(reward)
        print(
            "-" * 20,
            f"Question:\n{prompt}",
            f"\nAnswer:\n{completion}",
            f"\nReward:\n{reward}",
        )
    return rewards

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

# ### Inference

dataset = get_manipulation_questions()

results_list = []

scenario_count = 0

models = [
    'o3-mini',
    'gpt-4o-mini',
    'gpt-3.5-turbo',
    'gpt-4o',
]

for model in models:
    for scenario in dataset:
        scenario_count += 1
        for run in range(3):
            # Run the task 3 times
            goal_dict = scenario['answer']
            goal_dict = {
                'red_cube_goal': np.array(ast.literal_eval(goal_dict['red_cube_goal'])),
                'blue_cube_goal': np.array(ast.literal_eval(goal_dict['blue_cube_goal'])),
                'green_cube_goal': np.array(ast.literal_eval(goal_dict['green_cube_goal'])),
                'yellow_cube_goal': np.array(ast.literal_eval(goal_dict['yellow_cube_goal']))
            }
            env.reset(goal_location=goal_dict, task_idx_str=str(scenario_count) + str(model) + str(run))
            # print(scenario['prompt'])
            start_time = timeit.default_timer()
            completion = client.chat.completions.create(
                model=model,
                messages=scenario['prompt']
            )
            run_time = timeit.default_timer() - start_time
            answer = extract_xml_answer(completion.choices[0].message.content)
            reward = env.execute_llm_plan(answer)
            results_list.append([scenario_count, scenario['category'], model, run, scenario['question'], answer, reward, run_time])
            pd.DataFrame(results_list, columns=['scenario_idx', 'category', 'model', 'question', 'run', 'answer', 'reward', 'run_time']).to_csv('./results.csv')
            print('Correct Sequence Reward:', reward)