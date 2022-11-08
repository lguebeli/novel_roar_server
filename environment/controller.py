from agent import get_agent
from environment.reward import prepare_reward_computation


def run_c2():
    print("==============================\nPrepare Reward Computation\n==============================")
    prepare_reward_computation()

    cont = input("Results ok? Start C2 Server? [y/n]\n")
    if cont.lower() == "y":
        print("\n==============================\nStart C2 Server\n==============================")
        agent = get_agent()
        agent.loop_episodes()
        print("\n==============================\n! Done !\n==============================")
