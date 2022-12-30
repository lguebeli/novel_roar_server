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


## Prototypes

***Preliminary!***

| Prototype | Description                                                                                                                                                                                                                                                                                                                             |
|-----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1         | Proof of Concept: agent manually selects each action exactly once, then terminates after the last action. Every incoming fingerprint is evaluated by anomaly detection, but the computed reward is not further processed in any way.                                                                                                    |
| 2         | Incorporate intelligence: replace manual selection of actions with a neural network that predicts the optimal action to take in the current state. In this version, the agent performs a single episode and stops it once the ransomware is done encrypting.                                                                            |
| 3         | Considering Performance: the reward system now also considers the performance of the chosen action by incorporating the encryption rate. The learning process is continued over multiple episodes to a maximum number of steps per episode.                                                                                             |
| 4         | Improving simulations: simulating the encryption on the client device heavily speeds up the learning process. Additionally, modeling the simulation as close to the real environment as possible is key to transferring the learnings. Therefore, the simulation was adjusted to consider an artificial corpus of data to be encrypted. |
| x         | Comparing algorithms: This version implements the SARSA algorithm to compared the results with previous Q-learning versions.                                                                                                                                                                                                            |
| x         | Optimize framework: to further improve on accuracy and performance, the single neural network for action prediction is replaced with two actor-critic networks. Upon positive anomaly detection, the episode loop will consider presumable detection of the ransomware as failure and start a new episode.                              |


## Folder Structure

All scripts contained in this repository can only work if the required data can be found, i.e., the collected fingerprints need to be stored in a very specific way.

```
FOLDER                  DESCRIPTION

fingerprints            # The local folder containing all respective subdirectories. This folder and its children are not required to be located in this repository as long as the corresponding settings are correctly set.
-- evaluation           # The subfolder where the portion of fingerprints explicitly used only in accuracy computation is stored. The corresponding setting is called `EVALUATION_CSV_FOLDER_PATH`.
    -- infected-cX      # Directory for all infected-behavior fingerprints belonging to ransomware configuration X. There should be one folder for every configuration.
    -- normal           # Directory for normal-behavior fingerprints. There should be exactly one such folder here.
-- training             # The subfolder where all other fingerprints used during training will be stored. The corresponding setting is called `TRAINING_CSV_FOLDER_PATH`.
    -- infected-cX      # Directory for all infected-behavior fingerprints belonging to ransomware configuration X. There should be one folder for every configuration.
    -- normal           # Directory for normal-behavior fingerprints. There should be exactly one such folder here.
```


## Run the Server

There are different ways to use the `server.py` according to the intended purpose.

### Collect Fingerprints
Fitting if the intention is to only listen to the API for collection of incoming fingerprints.

Start the server as follows: `python server.py -c` (the flag `-c` marks collection of fingerprints).

### Train Agent (Production)
The agent can be trained in an online step-by-step fashion with fingerprints directly originating from the target device.
This is the most realistic scenario, but it trades higher authenticity for slower execution as the fingerprints collected on the device arrive roughly every 7 seconds.
Accordingly, the time between every step the agent can take is quite high, and learning is very slow - independent of the underlying implementation.

Start the server as follows: `python server.py -p <x>` (the flag `-p` marks prototype selection whereas `<x>` represents the selected version number).

### Train Agent (Simulation)
To tackle the problem of slow learning and dependency on a target device, the agent can be trained in a simulation with fingerprints originating from a corpus of previously collected fingerprints.
This allows to invest the time for collecting fingerprints only once and then run many simulations on the collected data.
While this comes with a much faster execution, it trades speed for authenticity because the environment is entirely artificial and may not represent the actual environment on the target device.

Start the server as follows: `python server.py -p <x> -s` (in addition to the flags mentioned above, the flag `-s` marks simulated execution).



## Auxiliary Scripts
Additionally, there are some auxiliary scripts used for everything around the C2 server.
If a script does not require any parameters or flags, it may also be run in an IDE of your choice for your convenience.

### Convert Fingerprints to CSV Files
Run this script to convert the collected fingerprints from the target device to a CSV file for further usage, e.g., in simulated execution.

Run the script as follows: `python fp-to-csv.py`

### Evaluate Anomaly Detection (AD)
To evaluate the quality of the collected fingerprints, or their underlying ransomware configuration, respectively, we can pass the collected fingerprints through AD.
This will first evaluate anomaly detection over all previously collected fingerprints (CSV datasets) once with SimplePreprocessor and once with the CorrelationPreprocessor (highly correlated features are removed).
Finally, the script will evaluate all collected fingerprints that still reside in the collection folder.
This is especially helpful during collection as to detect unexpected behavior or results as early as possible.

Run the script as follows: `python evaluate_AD.py`

### Compare Agent Accuracy
When verification is required that the agent is really learning how to properly select good actions, this script is what you want to run.

It first creates a fresh instance of an untrained agent and feeds all collected fingerprints for all available ransomware configs through its prediction.
The expected result is very bad as the agent's initial prediction skills are more or less equivalent to random guessing.
Then, the untrained agent is trained according to the current environment settings.
Lastly, the now trained agent is again evaluated by feeding all fingerprints through its prediction.
This time, however, the expected results are much better than before and clearly show that the agent is no longer guessing but demonstrates being able to select good actions for all possible states.

Run the script as follows: `python accuracy.py`


# TODO

- SARSA and multiple episodes
- reducing fingerprint input dimensionality with auto-encoder (learn important features representation)
- actor-critic
