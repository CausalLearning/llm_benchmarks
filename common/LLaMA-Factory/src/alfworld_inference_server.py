from llamafactory.chat import ChatModel
from llamafactory.extras.misc import torch_gc

# inference 
import time
import logging
import json
import subprocess
import os
import re
import gc
import ast
import torch
import numpy as np
import random
from typing import Dict, Any, Union
from textworld_server import Textworld_Env
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge

try:
    import platform

    if platform.system() != "Windows":
        import readline  # noqa: F401
except ImportError:
    print("Install `readline` for a better experience.")
	
########## Log Init  #############
now = int(time.time())
time_array = time.localtime(now)
format_time = time.strftime("%Y%m%d%H%M%S", time_array) 

# model_name = args.adapter_name_or_path.split("/")[-1].strip()
# checkpoint = args.adapter_name_or_path.split("/")[-1].strip()
	
# method = f"eval-{model_name}-{args.template}"
# llama3 gemma1.1 baichuan2 chatglm3 mistral qwen
model_name = "llama3"
scene_type = "unseen"
# scene_type = "seen"
	
method = f"eval-{model_name}"
# model_retified = "with-myself"
output_dir = f"log/{method}/{scene_type}-{format_time}/"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
		 
logger= logging.getLogger("Evaluation_LLMs")
logger.setLevel(logging.INFO)
handler = logging.FileHandler(os.path.join(output_dir, "evl.log"), 'a', 'utf-8')
handler.setFormatter(logging.Formatter("%(levelname)s:%(asctime)s | %(message)s"))
logger.addHandler(handler)

########## Log Init END  #############

########################  Loading Model ########################
chat_model = ChatModel()
textworld_env = Textworld_Env(port=9000)
def model_generate(query_prompt):
    messages = []
    messages.append({"role": "user", "content": query_prompt})
    # print("Prompt: ", messages)

    action = ""
    for new_text in chat_model.stream_chat(messages):
        # print(new_text, end="", flush=True)
        action += new_text
    # breakpoint()
    if "\n" in action:
        action = action.split("\n")[-1]
    # print(action)
    return action

########################  Loading Model END ########################

########################  Function Define ########################
go_pattern = r"go to (\w+) (\d+)"
open_pattern = r"open (\w+) (\d+)"
close_pattern = r"close (\w+) (\d+)"
use_pattern = r"use (\w+) (\d+)"
examine_pattern = r"examine (\w+) (\d+)"
take_pattern = r"take (\w+) (\d+) from (\w+) (\d+)"
put_pattern = r"put (\w+) (\d+) in/on (\w+) (\d+)"
clean_pattern = r"clean (\w+) (\d+) with (\w+) (\d+)"
cool_pattern = r"cool (\w+) (\d+) with (\w+) (\d+)"
heat_pattern = r"heat (\w+) (\d+) with (\w+) (\d+)"

def is_valid_action(action):
    valid_action = False
    if "look" == action:
        return True
    elif "inventory" == action:
        return True
    elif "go to" in action:
        match = re.match(go_pattern, action)
        if match:
            valid_action = True
    elif "open" in action:
        match = re.match(open_pattern, action)
        if match:
            valid_action = True
    elif "close" in action:
        match = re.match(close_pattern, action)
        if match:
            valid_action = True
    elif "use" in action:
        match = re.match(use_pattern, action)
        if match:
            valid_action = True
    elif "examine" in action:
        match = re.match(examine_pattern, action)
        if match:
            valid_action = True
    elif "take" in action:
        match = re.match(take_pattern, action)
        if match:
            valid_action = True
    elif "put" in action:
        match = re.match(put_pattern, action)
        if match:
            valid_action = True
    elif "clean" in action:
        match = re.match(clean_pattern, action)
        if match:
            valid_action = True
    elif "cool" in action:
        match = re.match(cool_pattern, action)
        if match:
            valid_action = True
    elif "heat" in action:
        match = re.match(heat_pattern, action)
        if match:
            valid_action = True
    else:
        valid_action = False
		
    return valid_action

def return_task_type(task_id):
    if task_id.startswith("pick_and_place_simple"):
        return "pick_and_place_simple"
    elif task_id.startswith("pick_two_obj_and_place"):
        return "pick_two_obj_and_place"
    elif task_id.startswith("pick_cool_then_place_in_recep"):
        return "pick_cool_then_place_in_recep"
    elif task_id.startswith("pick_clean_then_place_in_recep"):
        return "pick_clean_then_place_in_recep"
    elif task_id.startswith("pick_heat_then_place_in_recep"):
        return "pick_heat_then_place_in_recep"
    elif task_id.startswith("look_at_obj_in_light"):
        return "look_at_obj_in_light"
    
def save_result(success, step, task_type,Task_Index, success_rate, long_horizon_action):
    result = {}
    result["memory"] =''
    result["is_success"] = success
    result["path_length"] = step
    result["type"] = task_type
    result["long_horizon_action"] = long_horizon_action
    result["finish_rate"] = float(success_rate)
    result_data[f"env_{Task_Index-1}"] = result
    # 将字典保存为json文件
    with open(os.path.join(output_dir, 'env_results_trial_0.json'), 'w') as file:
        json.dump(result_data, file, indent=4)
        
def post_process(action, max_len=30):
    def max_len_process(string, max_len=30):
        # 去除句子两端的符号
        symbols = ".,!?;:-_~*"
        string = string.strip(symbols)
        words = string.split()
        word_count = len(words)
        if word_count > max_len:
            words = words[:max_len]
            
        return ' '.join(words)

    act = max_len_process(action)
    if len(act) > max_len:
        return act[:max_len]
    return act    

def action_retified(valid_actions, action):
    # 假设is_valid_action函数用来检查一个动作是否有效
    if action not in valid_actions:
        bleu_scores = {}
        rouge_scores = {}
        cc = SmoothingFunction()
        
        # 计算BLEU分数
        for valid_action in valid_actions:
            score = sentence_bleu([valid_action.split()], action.split(), smoothing_function=cc.method1)
            bleu_scores[valid_action] = score
        
        # 选出最高BLEU分数的动作
        max_bleu_score = max(bleu_scores.values())
        candidates = [action for action, score in bleu_scores.items() if score == max_bleu_score]
        
        # 如果有多个候选，进一步使用ROUGE分数筛选
        if len(candidates) > 1:
            rouge = Rouge()
            for candidate in candidates:
                scores = rouge.get_scores(candidate, action)
                rouge_score = scores[0]['rouge-l']['f']  # 使用ROUGE-L F分数作为参考
                rouge_scores[candidate] = rouge_score
            
            max_rouge_score = max(rouge_scores.values())
            final_candidates = [action for action, score in rouge_scores.items() if score == max_rouge_score]
            
            # 如果还是有多个候选，随机选一个
            action = random.choice(final_candidates)
        else:
            action = candidates[0]
    
    return action

def repeat_in_circle(action, previous_actions):
    # 检查AA形式的重复
    if len(previous_actions) > 0 and action == previous_actions[-1]:
        return True
    # 检查ABAB形式的重复
    if len(previous_actions) > 2 and action == previous_actions[-2] and previous_actions[-1] == previous_actions[-3]:
        return True
    return False

def filter_useless_action(valid_actions, previous_actions):
    unvalid_actions = ['inventory', 'look']
    actions_to_remove = []
    # if act in valid_actions and repeat_in_circle(act, previous_actions):
    #     actions_to_remove.append(act)
    for action in valid_actions:
        if action in unvalid_actions or action.startswith("examine") or repeat_in_circle(action, previous_actions):
            actions_to_remove.append(action)
    
    for action in actions_to_remove:
        valid_actions.remove(action)

    return valid_actions

def get_advise_from_myself(planing_context, candidates):
    # query_prompt = get_action_from_other_model.construct_query_prompt(planing_context, candidates)
    # action = post_process(model_generate(query_prompt))
    # print("--------------:", action)
    return "" 
	
def action_retified_v2(valid_actions, action, planing_context, long_horizon_action, total_action_retified, action_retified_by_model):
    
    if action not in valid_actions:
        total_action_retified += 1
        original_action = action
        # 首先做预处理,排除RDI
        valid_actions = filter_useless_action(valid_actions, long_horizon_action) 
        
        bleu_scores = {}
        rouge_scores = {}
        cc = SmoothingFunction()
        
        # 2. 计算BLEU分数
        for valid_action in valid_actions:
            score = sentence_bleu([valid_action.split()], action.split(), smoothing_function=cc.method1)
            bleu_scores[valid_action] = score

        # 选出最高BLEU分数的动作
        max_bleu_score = max(bleu_scores.values())
        candidates = [action for action, score in bleu_scores.items() if score == max_bleu_score]
        # breakpoint()
		
        # 3. 如果仍有多个候选，则使用模型进行来做选择
        if len(candidates) > 1:
            # action = get_action_from_other_model.from_gpt(planing_context, candidates)
            action = get_advise_from_myself(planing_context, candidates)
            if action not in candidates: # 有内容返回,但是还是无效的
                if action: 
                    rouge = Rouge()
                    for candidate in candidates:
                        scores = rouge.get_scores(candidate, action)
                        rouge_score = scores[0]['rouge-l']['f']  # 使用ROUGE-L F分数作为参考
                        rouge_scores[candidate] = rouge_score
                    # breakpoint()
                    max_rouge_score = max(rouge_scores.values())
                    final_candidates = [action for action, score in rouge_scores.items() if score == max_rouge_score]
                    
                    # 如果还是有多个候选，随机选一个
                    action = random.choice(final_candidates)
                else:
                    action = random.choice(candidates)
            else:
                print("Action Retified By Model:")
                action_retified_by_model += 1
            
        else:
            action = candidates[0]
        logger.info(f'###action_retified: [{original_action}] --> [{action}]')
        print(f'\n###action_retified: [{original_action}] --> [{action}]\n')
    return action, total_action_retified, action_retified_by_model    
########################  Function END ########################

Task_Index = 0 
Task_ID_List=[]
result_data = {}
Succeed_Count = 0
success_flag = False
success_rate = 0.0
# TOTAL_TASK=140 #seen
TOTAL_TASK=134 #unseen
action_retified_flag = True
step = 0
start_time = time.time()
task_scene = ""
long_horizon_action = []
task_type = None
total_action_retified = 0
action_retified_by_model = 0
print("##### Action Alignment:", output_dir+str(action_retified_flag))

while True:
    # Wait for requestion
    feedback_data, sr = textworld_env.get_feedback()

    # for new task
    if feedback_data.task_type:
        if Task_Index >= TOTAL_TASK:
            save_result(success_flag, step-1, Task_ID_List[-1], Task_Index, success_rate, long_horizon_action)
            logger.info(f'###action_retified: Taotal_number:[{total_action_retified}]; from_model:[{action_retified_by_model}]')
            print(f'###action_retified: Taotal_number:[{total_action_retified}]; from_model:[{action_retified_by_model}]')
            break
        if feedback_data.task_type not in Task_ID_List:
            if Task_Index != 0: #保存上一个环境执行的结果
                save_result(success_flag, step-1, Task_ID_List[-1], Task_Index, success_rate, long_horizon_action)
            Task_ID_List.append(feedback_data.task_type)
            task_scene = feedback_data.observation.split("\n\n")[1]
            step = 0
            success_flag = False
            task_type = return_task_type(feedback_data.task_type)
            logger.info(f"###current task: [{Task_Index}]:[{feedback_data.task_type}], Success:[{Succeed_Count}] SR:[{0 if Succeed_Count==0 else Succeed_Count/Task_Index}]")
            logger.info(f'###action_retified: Taotal_number:[{total_action_retified}]; from_model:[{action_retified_by_model}]')
            print(f'###action_retified: Taotal_number:[{total_action_retified}]; from_model:[{action_retified_by_model}]')
            print(f"###current task: [{Task_Index}]:[{feedback_data.task_type}], Success:[{Succeed_Count}] SR:[{0 if Succeed_Count==0 else Succeed_Count/Task_Index}]")
            Task_Index += 1
            long_horizon_action = []
        else:
            textworld_env.send_action("SKIP")
            continue
        
    if feedback_data.done:
        success_flag = True
        Succeed_Count +=1
		
    success_rate = sr
	
    # print(feedback_data.history)

    action = model_generate(feedback_data.history)
    if action_retified_flag:
        # action = action_retified(list(feedback_data.information['admissible_commands']), action)
        action, total_action_retified, action_retified_by_model = action_retified_v2(list(feedback_data.information['admissible_commands']), action, feedback_data.history, long_horizon_action, total_action_retified, action_retified_by_model)
    
    print(f"Task: [{Task_Index}], Iter: [{step}], Obs: {feedback_data.observation}, Action: {action}")
    logger.info(f"Task: [{Task_Index}], Iter: [{step}], Obs: {feedback_data.observation}, Action: {action}")
    step += 1
    long_horizon_action.append(action)
    textworld_env.send_action([action])

logger.info(f"---###--- Succeed: [{Succeed_Count}], SR:[{0 if Succeed_Count==0 else Succeed_Count/Task_Index}], TIME: {time.time() - start_time:.2f} s")
textworld_env.release_port
######################### END  ############################