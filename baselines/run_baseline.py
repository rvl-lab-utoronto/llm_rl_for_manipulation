"""
Unsloth memory-efficient TRL-vLLM GRPO
Exported from
https://docs.unsloth.ai/basics/reasoning-grpo-and-rl
"""
# Use `PatchFastRL` before all functions to patch GRPO and other RL algorithms!

import ast
import re
import sys

sys.path.append('../')
from datasets import Dataset, load_dataset
from pandas import *
from genesis_env import FrankaManipEnv
from openai import OpenAI
import numpy as np
from dotenv import load_dotenv

load_dotenv()


client = OpenAI()

max_seq_length = 4096  # Can increase for longer reasoning traces
lora_rank = 64  # Larger rank = smarter, but slower


# ### Data Prep
# <a name="Data"></a>
#
# We directly leverage
# [@willccbb](https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb)
# for data prep and all reward functions. You are free to create your own!

# starts simulator 
env = FrankaManipEnv(render_video=False, show_viewer=True)

# Load and prep dataset
with open("../prompts/base_manip_prompt.txt","r") as f:
    SYSTEM_PROMPT = f.read()

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


# uncomment middle messages for 1-shot prompting
def get_manipulation_questions(path = '../data/manipulation_tasks.xlsx'):
    # makes the initial Dataset
    xls = ExcelFile(path)
    df = xls.parse(xls.sheet_names[0])
    questions = df['Text Question'].to_list()
    rcg = df['red_cube_goal'].to_list()
    bcg = df['blue_cube_goal'].to_list()
    ycg = df['yellow_cube_goal'].to_list()
    gcg = df['green_cube_goal'].to_list()
    answers = []
    for i in range(len(rcg)):
        answers.append({'red_cube_goal':rcg[i],
                        'blue_cube_goal':bcg[i],
                        'yellow_cube_goal':ycg[i],
                        'green_cube_goal':gcg[i]})
    data = Dataset.from_dict({'question':questions,'answer':answers})

    # does some other fucky thing idk
    data = data.map(
        lambda x: {  # type: ignore
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": x["question"]},
            ],
            "answer":(x["answer"]),
        }
    )  # type: ignore
    return data  # type: ignore



dataset = get_manipulation_questions()


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
for scenario in dataset:
    # print(len(dataset))
    goal_dict = scenario['answer']
    goal_dict = {
        'red_cube_goal': np.array(ast.literal_eval(goal_dict['red_cube_goal'])),
        'blue_cube_goal': np.array(ast.literal_eval(goal_dict['blue_cube_goal'])),
        'green_cube_goal': np.array(ast.literal_eval(goal_dict['green_cube_goal'])),
        'yellow_cube_goal': np.array(ast.literal_eval(goal_dict['yellow_cube_goal']))
    }
    env.reset(goal_location=goal_dict)
    # print(scenario['prompt'])
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=scenario['prompt']
    )
    answer = extract_xml_answer(completion.choices[0].message.content)

    reward = env.execute_llm_plan(answer)
    print('Correct Sequence Reward:', reward)
    env.reset(goal_location=goal_dict)
    print(answer)