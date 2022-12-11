# Discarded Ransomware Configurations

There are different versions of ransomware configurations that were created along with development of the RL agent.
The latest set corresponding to the current version used in production is not listed here but contained in the intended [config folder](../rw-configs).

## V1 - Rate Limitation After File

| Configuration | JSON                                                                                 | Detection           |
|---------------|--------------------------------------------------------------------------------------|---------------------|
| 0 (neutral)   | { "algo": "AES-CTR", "rate": "0", "burst_duration": "f1", "burst_pause": "5" }       | [0 1] - [1025  226] |
| 1             | { "algo": "AES-CBC", "rate": "0", "burst_duration": "s0", "burst_pause": "0" }       | [0 1] - [   2  339] |
| 2             | { "algo": "AES-CTR", "rate": "0", "burst_duration": "s0", "burst_pause": "0" }       | [0 1] - [   2  312] |
| 3             | { "algo": "Salsa20", "rate": "0", "burst_duration": "s0", "burst_pause": "0" }       | [0 1] - [   1  291] |
| 4             | { "algo": "ChaCha20", "rate": "0", "burst_duration": "s0", "burst_pause": "0" }      | [0 1] - [   1  359] |
| 5             | { "algo": "AES-CTR", "rate": "0", "burst_duration": "s5", "burst_pause": "2" }       | [0 1] - [   1  373] |
| 6             | { "algo": "AES-CTR", "rate": "0", "burst_duration": "s1", "burst_pause": "5" }       | [0 1] - [ 117  549] |
| 7             | { "algo": "AES-CTR", "rate": "400000", "burst_duration": "s0", "burst_pause": "0" }  | [0 1] - [ 130  446] |
| 8             | { "algo": "AES-CTR", "rate": "100000", "burst_duration": "f50", "burst_pause": "2" } | [0 1] - [ 602  365] |
| 9             | { "algo": "Salsa20", "rate": "50000", "burst_duration": "s10", "burst_pause": "20" } | [0 1] - [1386  478] |
