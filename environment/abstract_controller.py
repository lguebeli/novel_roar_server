from abc import ABC, abstractmethod

import numpy as np

from agent import get_agent
from environment.reward import prepare_reward_computation


class AbstractController(ABC):
    @abstractmethod
    def loop_episodes(self, agent):
        pass

    def run_c2(self):
        print("==============================\nPrepare Reward Computation\n==============================")
        prepare_reward_computation()

        cont = input("Results ok? Start C2 Server? [y/n]\n")
        if cont.lower() == "y":
            print("\n==============================\nStart C2 Server\n==============================")
            agent = get_agent()
            rewards = self.loop_episodes(agent)
            print("\n==============================\n! Done !\n==============================")
            print(rewards)

    @staticmethod
    def transform_fp(fp):
        split_to_floats = list(map(lambda feat: float(feat), fp.split(",")))
        return np.asarray(split_to_floats).reshape(-1, 1)
