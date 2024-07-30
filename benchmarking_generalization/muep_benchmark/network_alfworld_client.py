import os
import copy
import hashlib
import requests
import json
import numpy as np
import tqdm
import cv2
import time
import argparse
from PIL import Image
from os.path import join as pjoin

import alfworld.agents
import alfworld.agents.environment as environment
import alfworld.agents.modules.generic as generic
from dataclasses import dataclass
from typing import Dict, Any

from alfworld.info import ALFWORLD_DATA
import single_alfworld_tw
from single_alfworld_thor import ThorEnvironment
from collections import deque

# url = 'http://proxy-7860-yangyijun14-notebook-p40-4.kuplus.jd.com'
# url = 'http://proxy-7860-yangyijun14-notebook-p40.kuplus.jd.com'
# url = 'http://proxy-7860-yangyijun14-training.kuplus.jd.com'
# url = 'http://proxy-7860-yangyijun14-notebook-04.kuplus.jd.com'ssssss
# url = 'http://proxy-7860-yangyijun14-notebook-05.kuplus.jd.com'
# url = 'http://proxy-7860-likanxue1-reflexion.kuplus.jd.com'
url = 'http://proxy-7860-likanxue1-notebook-04.kuplus.jd.com'
# url = 'http://proxy-7860-likanxue1-self-instruct.kuplus.jd.com'
# url = 'http://proxy-7860-likanxue1-benchmark.kuplus.jd.com'
# url = 'http://proxy-7860-likanxue1-tidybot.kuplus.jd.com'
# PORT=7862
# url = 'http://127.0.0.1:'+str(PORT)

# define feedback data struct
@dataclass
class FeedbackData:
    history: str = ''
    observation: str = ''
    task_type: str = ''
    information: Dict[str, Any]  = ''
    done: bool = False
    image: np.ndarray = None

def send_command(command):
    
    # generate MD5
    command_md5 = hashlib.md5(command.encode()).hexdigest()
    
    headers = {'MD5': command_md5}
    response = requests.post(url, data=command, headers=headers)

    # check response
    received_md5 = response.headers.get('MD5')
    response_data = response.content

    calculated_md5 = hashlib.md5(response_data).hexdigest()
    if received_md5 != calculated_md5:
        return False, ''
    
    return True, response_data.decode()

def get_next_action(feedback_data, wait_forever=False, env_reset=False, sr=0.0):
    request_data = f'[#OBSERVATION]{feedback_data.observation}[#HISTORY]{feedback_data.history}[#INFORMATION]{feedback_data.information}[#TYPE]{feedback_data.task_type}[#DONE]{feedback_data.done, str(sr)}[#IMAGE]{feedback_data.image}'
    # request_data = f'[#OBSERVATION]{feedback_data.observation}[#IMAGE]{feedback_data.image}'
    if wait_forever and env_reset:
        while True:
            success, action = send_command(request_data)
            if success:
                break
            # print('waiting for server...')
            time.sleep(2)
    else:
        success, action = send_command(request_data)
        
    if success:
        action_str = action[2:-2]
        return True, " ".join(action_str.split())
    else:
        return False, ""

def get_frame_data():
    # 从npy文件加载数据
    loaded_data = np.load('data.npy')  
    # 将numpy数组转换为JSON字符串
    return json.dumps(loaded_data.tolist())

def get_numpy_from_OPCV(file_path):
    # 读取图片
    img = cv2.imread(file_path)

    # 将图像转换为RGB格式
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 转换为numpy.ndarray格式
    img_array = np.array(img)

    return img_array

def get_numpy_from_PIL(file_path):
    # 读取图片
    image = Image.open(file_path).convert("RGB")

    # 转为numpy格式
    image_np = np.array(image)
    
    return image_np

def construt_action_history(history, new_action):
    if new_action == "START":
        history += ' > '
    else:
        # if new_action.startswith("think:"):
        #     result = "ok"
        # else:
            # if ob == "Nothing happens.":
            #     result = "fail"
            # else:
            #     result = "success"
            
        history = history + new_action + ' > '
    return history

def construt_action_history_pair_with_fb(history, new_action, ob):
    if new_action == "START":
        history += ' > '
    else:
        if ob == "Nothing happens.":
            result = "fail"
        else:
            result = "success"
            
        history = history + new_action + f' ({result})' +' > '
    return history

def insert_element(queue, new_element):  # 左出右进
    if len(queue) == 4:
        queue.popleft()
    queue.append(new_element)

def is_aa_or_abab(queue, action):
    if len(queue)>0 and action==queue[-1]: #aa
        print('AA:', action)
        return True
    if len(queue) >= 3: #abab
        if queue[-1] == queue[-3] and queue[-2] == action:
            print('ABAB:', action)
            return True
            
    return False

env_type = "AlfredThorEnv"  # 'AlfredTWEnv' or 'AlfredThorEnv'
if env_type == "AlfredThorEnv":
    env = ThorEnvironment()

unseen_valid_path = "valid_unseen.json"
seen_valid_path = "valid_seen.json"
env_file_list = []
with open(seen_valid_path,'r') as load_f:
    file_list = json.load(load_f)
for json_path in file_list:
    env_file_list.append(ALFWORLD_DATA+"/"+json_path)
    
# testing
# env_file_list = [env_file_list[i] for i in range(5)]

file_transf_test_flag = False
wait_forever_flag = True

if wait_forever_flag:
    collect_num = 1000000
else:
    collect_num = len(env_file_list)

env_steps = 30    
success_count = 0
progress_bar = tqdm.tqdm(total=collect_num)
fail_list = []

for i in range(0, collect_num):
    
    if env_type == 'AlfredTWEnv':
        env_args = dict()
        env_args['problem'] = os.path.dirname(env_file_list[i % len(env_file_list)])
        env_args['domain'] = pjoin(ALFWORLD_DATA, "logic", "alfred.pddl")
        env_args['grammar'] = pjoin(ALFWORLD_DATA, "logic", "alfred.twl2")
        env = single_alfworld_tw.get_one_env(env_args)
        obs, infos = env.reset()
    elif env_type == 'AlfredThorEnv':
        description = "Play the abstract text version of an ALFRED environment."
        parser = argparse.ArgumentParser(description=description)
        parser.add_argument("--problem", default=os.path.dirname(env_file_list[i % len(env_file_list)]))
        parser.add_argument('--load_receps', default=False)
        parser.add_argument('--reward_config', type=str, default=pjoin(alfworld.agents.__path__[0], 'config', 'rewards.json'))
        args = parser.parse_args()
        obs, infos = env.reset(args)
        
    # interact
    feedback_data = FeedbackData()
    if file_transf_test_flag:
        image = get_numpy_from_OPCV('000000000.jpg')
        feedback_data.image = json.dumps(image.tolist())
        feedback_data.history = "your task is to: put a potato in countertop > "
    else:
        text_input = construt_action_history(obs.split('\n\n')[2], "START")
        feedback_data.history = text_input
        # feedback_data.observation = "What is unusual about this image?"
        if env_type == 'AlfredThorEnv':
            feedback_data.image = json.dumps(env.get_frame_image().tolist())
            feedback_data.task_type = '/'.join(infos['extra.gamefile'].split('/')[-2:])
        else:
            feedback_data.image = get_frame_data()
        feedback_data.observation = obs
        feedback_data.information = infos
        if env_type == 'AlfredTWEnv':
            feedback_data.task_type = '/'.join(infos['extra.gamefile'].split('/')[-3:-1])
        
    success, action_data = get_next_action(feedback_data, wait_forever=wait_forever_flag, env_reset=True)
    if not success or action_data == "SKIP":
        continue
    cur_step = 0
    game_success = False
    success_rate = 0.0

    history_action_queue = deque(maxlen=4)
    insert_element(history_action_queue, action_data)
    if action_data:
        while cur_step < env_steps:
            if file_transf_test_flag:
                image = get_numpy_from_PIL('000000000.jpg')
                feedback_data.image = json.dumps(image.tolist())
                feedback_data.history = "your task is to: put a potato in countertop > "
            else:
                obs, success_rate, dones, infos = env.step(action_data)
                print("Action[{}]: {}, Obs: {}".format(cur_step, [action_data], obs))
                # print('admissible_commands:', infos['admissible_commands'])
                text_input = construt_action_history(text_input, action_data)
                # text_input = construt_action_history_pair_with_fb(text_input, action_data, obs)
                feedback_data.history = text_input
                # feedback_data.observation = "What is unusual about this image?"
                if env_type == 'AlfredThorEnv':
                    feedback_data.image = json.dumps(env.get_frame_image().tolist())
                else:
                    feedback_data.image = get_frame_data()
                feedback_data.observation = obs
                feedback_data.information = infos
                feedback_data.done = dones
                feedback_data.task_type = ''
                
                if dones:
                    success, action_data = get_next_action(feedback_data, wait_forever=wait_forever_flag, env_reset=False, sr=success_rate)
                    game_success = True
                    break
                
            success, action_data = get_next_action(feedback_data, wait_forever=wait_forever_flag, env_reset=False, sr=success_rate)
            cur_step += 1
            
            if is_aa_or_abab(history_action_queue, action_data):
                break
            else:
                insert_element(history_action_queue, action_data)
            
            if not success:
                break
            
        if game_success:
            success_count += 1
            
        print("tested_num: {}  success: {}".format(i+1, success_count))
    progress_bar.update(1)
    
    if env_type == "AlfredTWEnv":
        env.close()
    
# with open('failed_list.json', 'w') as f:
#     json.dump(fail_list, f)

