# ROAR - Server
Master Thesis on Ransomware Optimized with AI for Resource-constrained devices

## Configuration
Adjust constants in the following files:

| File                               | Constant                                                          |
|------------------------------------|-------------------------------------------------------------------|
| `fp-to-csv.py`                     | - CSV directory path<br>- CSV file names<br>- Verify CSV headers  |
| `api/configurations.py`            | - Client IP address                                               |
| `environment/anomaly-detection.py` | - CSV folder path<br>- Contamination factor                       |

# Versions

***Preliminary!***

| Prototype | Description                                                                                                                                                                                                                                                                                                   |
|-----------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1         | Proof of Concept: agent manually selects each action exactly once, then terminates after the last action.<br>Every incoming fingerprint is evaluated by anomaly detection, but the computed reward is not further processed in any way.                                                                       |
| 2         | Incorporate intelligence: replace manual selection of actions with a neural network that predicts the optimal action to take in the current state.<br>In this version, the agent stops its episode once the ransomware is done encrypting.                                                                    |
| 3         | Optimize framework: to further improve on accuracy and performance, the single neural network for action prediction is replaced with two actor-critic networks.<br>Upon positive anomaly detection, the episode loop will consider presumable detection of the ransomware as failure and start a new episode. |

# TODO

- version 2 with Q-learning and single episode
- version 3 with SARSA and multiple episodes
- add neutral config
- version 4 reducing fingerprint input dimensionality with auto-encoder (learn important features representation)
- version 5 with actor-critic
