import os
import numpy as np
import pandas as pd

from agent.agent_representation import AgentRepresentation
from environment.settings import LEARN_RATE_V22, DISCOUNT_FACTOR_V22
from environment.anomaly_detection.constructor import get_preprocessor
from environment.settings import ALL_CSV_HEADERS, TRAINING_CSV_FOLDER_PATH
from environment.state_handling import get_num_configs, get_storage_path
import json

LEARN_RATE = LEARN_RATE_V22
DISCOUNT_FACTOR = DISCOUNT_FACTOR_V22

class AgentSarsaTabular:
    def __init__(self, representation=None):
        self.representation = representation
        if isinstance(representation, AgentRepresentation):
            self.actions = list(range(representation.num_output))
            self.learn_rate = representation.learn_rate
            fp_features, min_vals, max_vals = self.__get_fp_features()
            self.fp_features = representation.get("fp_features", fp_features)
            self.min = np.array(representation.get("min", min_vals))
            self.max = np.array(representation.get("max", max_vals))
        else:  # Init from scratch
            num_configs = get_num_configs()
            self.actions = list(range(num_configs))
            self.learn_rate = LEARN_RATE
            self.fp_features, self.min, self.max = self.__get_fp_features()

        self.q_table = {}  # Q-table: key = state, value = array of Q-values for each action

    def __get_fp_features(self):
        """Get selected fingerprint features and compute min/max for all features from normal behavior."""
        df_normal = pd.read_csv(os.path.join(TRAINING_CSV_FOLDER_PATH, "normal-behavior.csv"))
        min_vals = df_normal.min().values  # Min for all features from raw data
        max_vals = df_normal.max().values  # Max for all features from raw data
        preprocessor = get_preprocessor()
        ready_dataset = preprocessor.preprocess_dataset(df_normal)
        fp_features = ready_dataset.columns.tolist()  # Selected features after preprocessing
        return fp_features, min_vals, max_vals

    def __preprocess_fp(self, fp):
        """Select relevant features from the fingerprint."""
        headers = ALL_CSV_HEADERS.split(",")
        indexes = [headers.index(header) for header in self.fp_features]
        return fp[:, indexes]

    def standardize_fp(self, fp):
        """Standardize the full fingerprint using min-max scaling, then select relevant features."""
        if fp.ndim == 1:
            fp = fp.reshape(1, -1)
        standardized_fp = (fp - self.min) / (self.max - self.min + 1e-6)
        selected_fp = self.__preprocess_fp(standardized_fp)
        return selected_fp

    def discretize_state(self, state):
        """Discretize the selected standardized state."""
        standardized_state = self.standardize_fp(state)
        return tuple(np.round(standardized_state, decimals=2).astype(float).flatten())

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

    def save_q_table(self, description="sarsa_q_table"):
        """Saves the Q-table as a JSON file."""
        q_table_serializable = {
            ",".join(map(str, key)): value.tolist() for key, value in self.q_table.items()
        }
        q_table_path = os.path.join(get_storage_path(), f"{description}.json")
        with open(q_table_path, "w") as file:
            json.dump(q_table_serializable, file)
        return q_table_path

    def load_q_table(self, q_table_path):
        """Loads the Q-table from a JSON file."""
        if os.path.exists(q_table_path):
            with open(q_table_path, "r") as file:
                q_table_serializable = json.load(file)
            self.q_table = {
                tuple(map(float, key.split(","))): np.array(value)
                for key, value in q_table_serializable.items()
            }
        else:
            print("No existing Q-table found, starting fresh.")