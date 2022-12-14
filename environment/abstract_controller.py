from abc import ABC, abstractmethod
from time import sleep

import numpy as np

from agent import get_agent
from environment.reward.abstract_reward import AbstractReward
from environment.state_handling import is_api_running

WAIT_FOR_CONFIRM = False


class AbstractController(ABC):
    @abstractmethod
    def loop_episodes(self, agent):
        pass

    def __start_training(self):
        print("\nWaiting for API...")
        while not is_api_running():
            sleep(1)

        print("\n==============================\nStart C2 Server\n==============================")
        agent = get_agent()
        rewards = self.loop_episodes(agent)
        print("\n==============================\n! Done !\n==============================")
        print(rewards)

    def run_c2(self):
        print("==============================\nPrepare Reward Computation\n==============================")
        AbstractReward.prepare_reward_computation()

        if WAIT_FOR_CONFIRM:
            cont = input("Results ok? Start C2 Server? [y/n]\n")
            if cont.lower() == "y":
                self.__start_training()
        else:
            self.__start_training()

    @staticmethod
    def transform_fp(fp):
        split_to_floats = list(map(lambda feat: float(feat), fp.split(",")))
        return np.asarray(split_to_floats).reshape(-1, 1)  # shape (F, 1)
