import hashlib
import http.server
import socketserver
import threading
import queue
import time
import json
import subprocess
import os
import re
import ast
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import Dict, Any

# 在eventlet中，常规的Queue可能不会按预期工作，因为eventlet是基于协程的，用于实现并发，
# 它拦截了标准库中的很多阻塞调用并将它们转换成非阻塞调用。所以不使用queue进行队列初始化
import eventlet
eventlet.monkey_patch()
from eventlet.queue import Queue
# import queue as Queue

@dataclass
class FeedbackData:
    history: str = ''
    observation: str = ''
    task_type: str = ''
    information: Dict[str, Any] = ''
    done: bool = False
    image: np.ndarray = None

class Textworld_Env:
    def __init__(self, port=7860, ip="0.0.0.0"):
        self.action_queue = Queue()
        self.feedback_queue = Queue()
        self.server_thread = self.ServerThread(ip, port, self.action_queue, self.feedback_queue)
        self.server_thread.start()

    def get_feedback(self):
        data = self.feedback_queue.get()
        feedback_data, sr = self.process_feedback(data)
        return feedback_data, sr

    def send_action(self, action):
        self.action_queue.put(action)
        
     # 调用命令kill之前的监听进程
    @staticmethod
    def release_port():
        command = "ss -lptn 'sport = :7860'"
        output = subprocess.check_output(command, shell=True).decode()
        # 解析输出结果获取PID
        lines = output.strip().split('\n')
        if len(lines) > 1:
            line = lines[1]
            # 使用正则表达式提取PID
            pid_pattern = r',pid=(\d+),'
            match = re.search(pid_pattern, line)
            if match:
                pid = match.group(1)
                # 杀死对应的进程
                os.kill(int(pid), 9)
                time.sleep(5)

    @staticmethod
    def process_feedback(text):
        # 使用正则表达式匹配并提取字段
        pattern = r'\[#OBSERVATION\](.*?)\[#HISTORY\](.*?)\[#INFORMATION\](.*?)\[#TYPE\](.*?)\[#DONE\](.*?)\[#IMAGE\](.*)'
        mat = re.match(pattern, text, re.DOTALL)
        feedback_data = FeedbackData()
        if mat:
            feedback_data.observation = mat.group(1)
            feedback_data.history = mat.group(2)
            feedback_data.information = ast.literal_eval(mat.group(3))
            feedback_data.task_type = mat.group(4)
            # 划分done与reward
            str_spilt = mat.group(5)[1:-1].split(", ") # remove "( )"
            feedback_data.done = str_spilt[0].strip() == "True"
            sr = float(str_spilt[1][1:-1].strip()) # remove "' '"
            py_data = json.loads(mat.group(6))
            feedback_data.image = Image.fromarray(np.asarray(py_data, dtype=np.uint8))
            # feedback_data.image = Image.fromarray(feedback_data.image)
        return feedback_data, sr

    class ServerThread(threading.Thread):
        def __init__(self, host, port, action_queue, feedback_queue):
            super().__init__()
            self.host = host
            self.port = port
            self.action_queue = action_queue
            self.feedback_queue = feedback_queue

        def run(self):
            server_thread = self

            class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
                def log_message(self, format: str, *args):
                    pass

                def do_POST(self):
                    try:
                        content_length = int(self.headers['Content-Length'])
                        data = self.rfile.read(content_length)
                        
                        # check data integrity
                        received_md5 = self.headers.get('MD5')
                        calculated_md5 = hashlib.md5(data).hexdigest()
                        if received_md5 != calculated_md5:
                            self.send_response(400)
                            self.end_headers()
                        else:
                            # put feedback into feedback_queue
                            server_thread.feedback_queue.put(data.decode())
                            
                            # wait for feedback
                            action = server_thread.action_queue.get()
                            action = str(action)
                            response_data = action.encode()
                            response_md5 = hashlib.md5(response_data).hexdigest()
                            self.send_response(200)
                            self.send_header('MD5', response_md5)
                            self.end_headers()
                            self.wfile.write(response_data)
                    except Exception as e:
                        self.send_error(400, str(e))

            with socketserver.TCPServer((self.host, self.port), MyHTTPRequestHandler) as httpd:
                print(f"Server running on IP: {self.host}, PORT: {self.port}")
                httpd.serve_forever()