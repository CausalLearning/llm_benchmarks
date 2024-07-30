#!/usr/bin/env python

import os
import json

import textworld

import gym
import textworld.gym

from alfworld.agents.utils.misc import Demangler, add_task_to_grammar


class AlfredDemangler(textworld.core.Wrapper):

    def load(self, *args, **kwargs):
        super().load(*args, **kwargs)

        demangler = Demangler(game_infos=self._game.infos)
        for info in self._game.infos.values():
            info.name = demangler.demangle_alfred_name(info.id)
            

class AlfredInfos(textworld.core.Wrapper):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._gamefile = None

    def load(self, *args, **kwargs):
        super().load(*args, **kwargs)
        self._gamefile = args[0]

    def reset(self, *args, **kwargs):
        state = super().reset(*args, **kwargs)
        state["extra.gamefile"] = self._gamefile
        return state


def get_one_env(args):
    print(f"Playing '{args['problem']}'.")
    GAME_LOGIC = {
        "pddl_domain": open(args['domain']).read(),
        "grammar": open(args['grammar']).read(),
    }

    # load state and trajectory files
    pddl_file = os.path.join(args['problem'], 'initial_state.pddl')
    json_file = os.path.join(args['problem'], 'traj_data.json')
    with open(json_file, 'r') as f:
        traj_data = json.load(f)
    GAME_LOGIC['grammar'] = add_task_to_grammar(GAME_LOGIC['grammar'], traj_data)

    # dump game file
    gamedata = dict(**GAME_LOGIC, pddl_problem=open(pddl_file).read())
    gamefile = os.path.join(os.path.dirname(pddl_file), 'game.tw-pddl')
    json.dump(gamedata, open(gamefile, "w"))

    # register a new Gym environment.
    infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
    env_id = textworld.gym.register_game(gamefile, infos,
                                         max_episode_steps=1000000,
                                         wrappers=[AlfredDemangler, AlfredInfos])

    # reset env
    return gym.make(env_id)

    # obs, infos = env.reset()

    # # human agent
    # agent = HumanAgent(True)
    # agent.reset(env)

    # while True:
    #     print(obs)
    #     cmd = agent.act(infos, 0, False)

    #     if cmd == "ipdb":
    #         from ipdb import set_trace; set_trace()
    #         continue

    #     obs, score, done, infos = env.step(cmd)

    #     if done:
    #         print("You won!")
    #         break