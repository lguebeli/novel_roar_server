import os
import numpy as np
import pandas as pd

from agent.agent_representation import AgentRepresentation
from environment.anomaly_detection.constructor import get_preprocessor
from environment.settings import ALL_CSV_HEADERS, TRAINING_CSV_FOLDER_PATH
from environment.state_handling import get_num_configs, get_storage_path
import json

LEARN_RATE = 0.0050
DISCOUNT_FACTOR = 0.75

class AgentSarsaTabular:
    def __init__(self, representation=None):
        self.representation = representation
        if isinstance(representation, AgentRepresentation):  # build from representation
            self.actions = list(range(representation.num_output))
            self.learn_rate = representation.learn_rate
        else:  # init from scratch
            num_configs = get_num_configs()
            self.actions = list(range(num_configs))
            self.learn_rate = LEARN_RATE

        self.q_table = {}  # Q-table: key = state, value = array of Q-values for each action
        self.fp_features = self.__get_fp_features()

    def __get_fp_features(self):
        df_normal = pd.read_csv(os.path.join(TRAINING_CSV_FOLDER_PATH, "normal-behavior.csv"))
        preprocessor = get_preprocessor()
        ready_dataset = preprocessor.preprocess_dataset(df_normal)
        return ready_dataset.columns

    def __preprocess_fp(self, fp):
        headers = ALL_CSV_HEADERS.split(",")
        indexes = [headers.index(header) for header in self.fp_features]
        return fp[indexes]

    def discretize_state(self, state):
        return tuple(np.round(state, decimals=1).astype(float).flatten())

    def predict(self, epsilon, state):
        d_state = self.discretize_state(state)
        if d_state not in self.q_table:
            self.q_table[d_state] = np.zeros(len(self.actions))

        q_values = self.q_table[d_state]
        if np.random.random() < epsilon:  # Explore
            action = np.random.choice(self.actions)
        else:  # Exploit
            action = np.argmax(q_values)

        return action, q_values

    def update_q_table(self, state, action, reward, next_state, next_action, done):
        d_state = self.discretize_state(state)
        d_next_state = self.discretize_state(next_state)

        if d_next_state not in self.q_table:
            self.q_table[d_next_state] = np.zeros(len(self.actions))

        current_q = self.q_table[d_state][action]
        next_q = self.q_table[d_next_state][next_action] if not done else 0
        target = reward + DISCOUNT_FACTOR * next_q
        self.q_table[d_state][action] += self.learn_rate * (target - current_q)

    def init_error(self):
        return np.zeros(len(self.actions))

    def update_error(self, error, reward, selected_action, selected_q_value, next_action, next_q_value, is_done):
        if is_done:
            error[selected_action] = reward - selected_q_value
        else:
            error[selected_action] = reward + (DISCOUNT_FACTOR * next_q_value) - selected_q_value
        return error

    def save_q_table(self, description="sarsa_q_table"):
        """Saves the Q-table as a JSON file."""
        # Convert the keys of the q_table (tuples) to strings
        q_table_serializable = {str(key): value.tolist() for key, value in self.q_table.items()}

        q_table_path = os.path.join(get_storage_path(), f"{description}.json")
        with open(q_table_path, "w") as file:
            json.dump(q_table_serializable, file)
        #print(f"Q-table saved at: {q_table_path}")
        return q_table_path

    def load_q_table(self, q_table_path):
        """Loads the Q-table from a JSON file."""
        if os.path.exists(q_table_path):
            with open(q_table_path, "r") as file:
                q_table_serializable = json.load(file)

            # Convert the keys back from strings to tuples
            self.q_table = {tuple(map(float, key.strip("()").split(", "))): np.array(value)
                            for key, value in q_table_serializable.items()}

            #print(f"Q-table loaded from: {q_table_path}")
        else:
            print("No existing Q-table found, starting fresh.")
