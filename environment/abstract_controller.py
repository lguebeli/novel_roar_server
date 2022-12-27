import json
import os
from abc import ABC, abstractmethod
from time import sleep

import numpy as np

from agent.constructor import get_agent
from environment.reward.abstract_reward import AbstractReward
from environment.state_handling import is_api_running, is_simulation, get_prototype, get_storage_path

WAIT_FOR_CONFIRM = False


class AbstractController(ABC):
    @abstractmethod
    def loop_episodes(self, agent):
        pass

    def __start_training(self):
        if not is_simulation():
            print("\nWaiting for API...")
            while not is_api_running():
                sleep(1)

        print("\n==============================\nStart Training\n==============================")
        agent = get_agent()
        q_values, rewards = self.loop_episodes(agent)
        if int(get_prototype()) > 1:
            print("==============================", "Rewards:", rewards, "\nFinal Q-Values:", q_values, sep="\n")
        print("\n==============================\n! Done !\n==============================")

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
    def save_agent(weights1, weights2, bias_weights1, bias_weights2, epsilon, agent, description):
        agent_file = os.path.join(get_storage_path(), "agent={}.json".format(description))
        content = {
            "weights1": weights1.tolist(),
            "weights2": weights2.tolist(),
            "bias_weights1": bias_weights1.tolist(),
            "bias_weights2": bias_weights2.tolist(),
            "epsilon": epsilon,
            "learn_rate": agent.learn_rate,
            "num_input": agent.num_input,
            "num_hidden": agent.num_hidden,
            "num_output": agent.num_output,
        }
        with open(agent_file, "w+") as file:
            json.dump(content, file)
        return agent_file

    @staticmethod
    def transform_fp(fp):
        split_to_floats = list(map(lambda feat: float(feat), fp.split(",")))
        return np.asarray(split_to_floats).reshape(-1, 1)  # shape (F, 1)
