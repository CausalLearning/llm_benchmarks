import hashlib
import requests
import json
import numpy as np
# import cv2
import time
from PIL import Image

from dataclasses import dataclass
from typing import Dict, Any

url = 'http://proxy-7860-likanxue1-notebook-04.kuplus.jd.com'

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

def get_next_action(feedback_data, wait_forever=False):
    request_data = f'[#OBSERVATION]{feedback_data.observation}[#HISTORY]{feedback_data.history}[#INFORMATION]{feedback_data.information}[#TYPE]{feedback_data.task_type}[#DONE]{feedback_data.done}[#IMAGE]{feedback_data.image}'
    if wait_forever:  # 一直等待，直到模型返回响应
        while True:
            success, action = send_command(request_data)
            if success:
                break
            time.sleep(2)
    else:
        success, action = send_command(request_data)
        
    if success:
        action_str = action[2:-2]
        return True, " ".join(action_str.split())
    else:
        return False, ""

# # Opencv库读取图片
# def get_numpy_from_OPCV(file_path):
#     # 读取图片
#     img = cv2.imread(file_path)

#     # 将图像转换为RGB格式
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     # 转换为numpy.ndarray格式
#     img_array = np.array(img)

#     return img_array

# PIL库读取图片
def get_numpy_from_PIL(file_path):
    # 读取图片
    image = Image.open(file_path).convert("RGB")

    # 转为numpy格式
    image_np = np.array(image)
    
    return image_np

def consturct_command(image_path, historical_action):
    feedback_data = FeedbackData()
    image = get_numpy_from_PIL(image_path)
    feedback_data.image = json.dumps(image.tolist())
    feedback_data.history = historical_action

    return feedback_data

image = '000000000.jpg'
action = ""
task = 'your task is to: put a potato in countertop'

# task里面包含： “任务” + “历史动作序列”
while True:
    task += action + ' > '
    print(task)
    feedback_data = consturct_command(image, task)

    # interact with EMMA
    success, action = get_next_action(feedback_data, wait_forever=True)
    
    # 机器人底层实现action

    # 完成一个动作后的图像观察
    image = '000000001.jpg' 
