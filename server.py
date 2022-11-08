from argparse import ArgumentParser
from multiprocessing import Process
from time import sleep

from api import create_app
from environment.controller import run_c2
from environment.state_handling import is_multi_fp_collection, set_multi_fp_collection, initialize_storage


def parse_args():
    parser = ArgumentParser(description='C2 Server')
    parser.add_argument('-c', '--collect',
                        help='Indicator to only collect incoming fingerprints instead of running the full C2 server.',
                        action="store_true")

    return parser.parse_args()


def start_api():
    app = create_app()
    app.run(host="0.0.0.0", port=5000)


def kill_process(proc):
    print("Kill Process", proc)
    proc.kill()
    proc.join()


if __name__ == "__main__":
    print("==============================\nInstantiate Storage\n")
    initialize_storage()

    # Parse arguments
    args = parse_args()
    collect = args.collect or False
    set_multi_fp_collection(collect)

    # Start API listener
    print("\n==============================\nStart API\n==============================")
    procs = []
    proc_api = Process(target=start_api)
    procs.append(proc_api)
    proc_api.start()

    # Start C2 server
    try:
        if not is_multi_fp_collection():
            run_c2()
        else:
            while True:
                sleep(600)  # sleep until process is terminated by user keyboard interrupt
    finally:
        for proc in procs:
            kill_process(proc)
