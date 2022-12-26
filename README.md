# ROAR - Server
Master Thesis on Ransomware Optimized with AI for Resource-constrained devices

## Configuration
Adjust constants in the following files:

| File                               | Constant                                                                                                      |
|------------------------------------|---------------------------------------------------------------------------------------------------------------|
| `fp-to-csv.py`                     | - CSV file names<br>- Verify CSV headers                                                                      |
| `environment/anomaly-detection.py` | - Contamination factor                                                                                        |
| `environment/settings.py`          | - CSV folder path<br>- Verify CSV headers<br>- Client IP address<br>- AD features<br>- C2 simulation settings |
| `vX/agent/agent.py`                | - Agent specific constants and starting values                                                                |
| `vX/agent/model.py`                | - Model specific constants and starting values                                                                |
| `vX/environment/controller.py`     | - Episode specific constants and values                                                                       |

# Prototypes

***Preliminary!***

| Prototype | Description                                                                                                                                                                                                                                                                                                                             |
|-----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1         | Proof of Concept: agent manually selects each action exactly once, then terminates after the last action. Every incoming fingerprint is evaluated by anomaly detection, but the computed reward is not further processed in any way.                                                                                                    |
| 2         | Incorporate intelligence: replace manual selection of actions with a neural network that predicts the optimal action to take in the current state. In this version, the agent performs a single episode and stops it once the ransomware is done encrypting.                                                                            |
| 3         | Considering Performance: the reward system now also considers the performance of the chosen action by incorporating the encryption rate. The learning process is continued over multiple episodes to a maximum number of steps per episode.                                                                                             |
| 4         | Improving simulations: simulating the encryption on the client device heavily speeds up the learning process. Additionally, modeling the simulation as close to the real environment as possible is key to transferring the learnings. Therefore, the simulation was adjusted to consider an artificial corpus of data to be encrypted. |
| x         | Comparing algorithms: This version implements the SARSA algorithm to compared the results with previous Q-learning versions.                                                                                                                                                                                                            |
| x         | Optimize framework: to further improve on accuracy and performance, the single neural network for action prediction is replaced with two actor-critic networks. Upon positive anomaly detection, the episode loop will consider presumable detection of the ransomware as failure and start a new episode.                              |

# TODO

- SARSA and multiple episodes
- reducing fingerprint input dimensionality with auto-encoder (learn important features representation)
- actor-critic
