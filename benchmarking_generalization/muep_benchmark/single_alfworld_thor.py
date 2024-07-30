#!/usr/bin/env python

import os
import json

from alfworld.env.thor_env import ThorEnv
from alfworld.agents.controller import OracleAgent


class ThorEnvironment(object):
    
    def __init__(self):

        # start THOR
        self.env = ThorEnv(
            player_screen_height=300,
            player_screen_width=300
        )

        
    def _setup_scene(self, traj_data, args, reward_type='dense'):
        # scene setup
        scene_num = traj_data['scene']['scene_num']
        object_poses = traj_data['scene']['object_poses']
        dirty_and_empty = traj_data['scene']['dirty_and_empty']
        object_toggles = traj_data['scene']['object_toggles']

        scene_name = 'FloorPlan%d' % scene_num
        self.env.reset(scene_name)
        self.env.restore_scene(object_poses, object_toggles, dirty_and_empty)

        # initialize to start position
        self.env.step(dict(traj_data['scene']['init_action']))

        # setup task for reward
        self.env.set_task(traj_data, args, reward_type=reward_type)
        
    def reset(self, args):
        
        print(f"Playing '{args.problem}'.")
        # load traj_data
        self.root = args.problem
        self.info = {"extra.gamefile": args.problem}
        json_file = os.path.join(args.problem, 'traj_data.json')
        with open(json_file, 'r') as f:
            traj_data = json.load(f)

        # setup scene
        self._setup_scene(traj_data, args)
        
        # setup controller
        # self.agent = OracleAgent(
        #     self.env, traj_data, traj_root=self.root, 
        #     load_receps=args.load_receps, debug=False,
        #     goal_desc_human_anns_prob=1.0,
        #     use_gt_relations=True
        # )
        self.agent = OracleAgent(
            self.env, traj_data, traj_root=self.root, 
            load_receps=args.load_receps, debug=False, use_gt_relations=True
        )
        
        obs = self.agent.feedback
        self.info['admissible_commands'] = self.agent.get_admissible_commands()
        return obs, self.info
    
    def step(self, action):
        self.agent.step(action)
        next_obs = self.agent.feedback
        done = self.env.get_goal_satisfied()
        # 计算当前任务的完成率
        pcs = self.env.get_goal_conditions_met()
        goal_condition_success_rate = pcs[0] / float(pcs[1])
        self.info['admissible_commands'] = self.agent.get_admissible_commands()
        return next_obs, goal_condition_success_rate, done, self.info
    
    def get_frame_image(self):
        return self.env.last_event.frame