import json
import os
import re

# Metric: SR(成功率), IS(交互步数), SGSR(字目标成功率)，IAR(无效动作比例), IAL(无效动作循环)

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

max_interaction_step = 30

def is_valid_action(action):
    valid_action = False
    if action.startswith('think:'): # for Reflexion
        return True
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

    if not valid_action:
        print(action)
    return valid_action

def has_repeating_pattern(action_queue, max_pattern_length=6):
    
    for pattern_length in range(1, max_pattern_length + 1):
        for i in range(len(action_queue) - 2 * pattern_length + 1):
            pattern = action_queue[i:i+pattern_length]
            
            # 检查序列中的下一个子序列是否与当前模式相同
            for j in range(i + pattern_length, len(action_queue) - pattern_length + 1, pattern_length):
                if pattern_length == 1: 
                    # if pattern == action_queue[j:j+pattern_length] == action_queue[j+1:j+pattern_length+1]: # AAA
                    if pattern == action_queue[j:j+pattern_length]: # AA
                        return pattern
                    else:
                        break 
                else:
                    if pattern_length == 2: # AABB
                        if (action_queue[i] == action_queue[i+1]) and  (action_queue[i+2] == action_queue[i+3]):
                            return pattern
                    if action_queue[j:j+pattern_length] == pattern: # ABCABC ABAB
                        # 如果找到重复模式，返回该模式
                        return pattern
                    else:
                        # 如果当前位置的子序列不匹配，跳出内循环
                        break
    # 如果没有找到任何重复模式，返回None
    return None
    
# metrics for avg
def avg_performance_analysis(result_datas):
    total_task = len(result_datas)
    success_task_count = 0
    success_step_count = 0
    total_step_count = 0
    invalid_action_count = 0
    invalid_action_loop_count = 0
    sgis = float(0) # 测试集上所有的任务完成率
    action_length = 0
    
    for key, result in result_datas.items():
    # for result in result_datas: # for reflexion log
        action_length += result["path_length"]
        
        if result["is_success"]: #success task
            success_task_count += 1
            success_step_count += result["path_length"]
            total_step_count += result["path_length"]
            sgis += float(1)
        else: #failure task                 
            sgis += result["finish_rate"]
            # 1. count invalid actions
            for action in result["long_horizon_action"]:
                if not is_valid_action(action):
                    invalid_action_count += 1
            
            # count rdi
            # if result["path_length"] < max_interaction_step: # unvalid action loop
                #     invalid_action_loop_count += 1
            if has_repeating_pattern(result["long_horizon_action"]) != None:
                invalid_action_loop_count += 1
              
            total_step_count += min(30, len(result["long_horizon_action"]))
            
    agent_sr = float(success_task_count)/total_task
    agent_is = float(success_step_count)/success_task_count
    agent_lc = float(invalid_action_count)/(action_length-success_step_count)
    agent_rdi = float(invalid_action_loop_count)/(total_task-success_task_count)
    agent_gcs = float(sgis)/total_task
    # breakpoint()

    print("{:<10s} {:<10s} {:<10s} {:<10s} {:<10s} {:10s}".format("SR", "IS", "GCS", "LC", "RDI", "seq_len"))
    print("{:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f} {}".format(agent_sr*100, agent_is, agent_gcs*100, (1-agent_lc)*100, agent_rdi*100, action_length))
    print()
	
	
def extract_task_type(task):
	if task.startswith("pick_and_place_simple"):
		return "Pick"
	elif task.startswith("look_at_obj_in_light"):
		return "Examine"
	elif task.startswith("pick_clean_then_place_in_recep"):
		return "Clean"
	elif task.startswith("pick_cool_then_place_in_recep"):
		return "Cool"
	elif task.startswith("pick_heat_then_place_in_recep"):
		return "Heat"
	elif task.startswith("pick_two_obj_and_place"):
		return "Pick Two"
	else:
		print("###[ERR]: TASK TYPE")
		return ""

def task_type_performance_analysis(result_datas, task_types):
    task_type_stats = {task_type: {"success_task_count": 0, "total_task_count": 0, "total_step_count": 0, "success_step_count": 0, "invalid_action_count": 0, "invalid_action_loop_count": 0, "sgis": 0} for task_type in task_types}

    total_task = len(result_datas)
    total_step_count = 0

    for key, result in result_datas.items():
    # for result in result_datas: # for reflexion log
        task_type = extract_task_type(result["type"])
        task_type_stats[task_type]["total_task_count"] += 1
        # task_type_stats[task_type]["total_step_count"] += result["path_length"]
                
        # 2. count rdi
        # if result["path_length"] < max_interaction_step: # unvalid action loop
        #     task_type_stats[task_type]["invalid_action_loop_count"] += 1
        if has_repeating_pattern(result["long_horizon_action"]) != None:
            task_type_stats[task_type]["invalid_action_loop_count"] += 1
                        
        if result["is_success"]: # success task
            task_type_stats[task_type]["success_task_count"] += 1
            task_type_stats[task_type]["success_step_count"] += result["path_length"]
            task_type_stats[task_type]["total_step_count"] += result["path_length"]
            task_type_stats[task_type]["sgis"] += 1
            # total_step_count += result["path_length"]

        else: #failure task
            # 1. count invalid actions 
            for action in result["long_horizon_action"]:
                if not is_valid_action(action):
                    task_type_stats[task_type]["invalid_action_count"] += 1
                    
            task_type_stats[task_type]["sgis"] += result["finish_rate"]
            task_type_stats[task_type]["total_step_count"] += min(30, len(result["long_horizon_action"]))
            # total_step_count += min(30, len(result["long_horizon_action"]))

    for task_type, stats in task_type_stats.items():
        success_task_count = stats["success_task_count"]
        total_task_count = stats["total_task_count"]
        total_step_count = stats["total_step_count"]
        success_step_count = stats["success_step_count"]
        invalid_action_count = stats["invalid_action_count"]
        invalid_action_loop_count = stats["invalid_action_loop_count"]
        sgis = stats["sgis"]

        agent_sr = float(success_task_count) / total_task_count
        agent_is = float(success_step_count) / success_task_count if success_task_count > 0 else 0
        agent_lc = float(invalid_action_count) / (total_step_count - success_step_count)
        agent_rdi = float(invalid_action_loop_count) / total_task_count
        agent_gcs = float(sgis) / total_task_count

        print("Task Type: ", task_type)
        print("{:<10s} {:<10s} {:<10s} {:<10s} {:<10s}".format("SR", "IS", "GCS", "LC", "RDI"))
        print("{:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f}".format(agent_sr*100, agent_is, agent_gcs*100, (1-agent_lc)*100, agent_rdi*100))
        print()

# 读取train.json文件
dir_path = '/home/fist_user2/LLaMA-Factory/log/muep_peft/eval-qwen/unseen-20240724160750'
with open(os.path.join(dir_path, 'env_results_trial_0.json'), 'r') as file:
    result_datas = json.load(file)
	
avg_performance_analysis(result_datas)
# 使用示例  按类别统计
# task_types = ["Pick", "Examine", "Clean", "Heat", "Cool", "Pick Two"]

# task_type_performance_analysis(result_datas, task_types)