import os

import numpy as np
import pandas as pd
from pyod.models.iforest import IForest
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from environment.settings import CSV_FOLDER_PATH, ALL_CSV_HEADERS, DUPLICATE_HEADERS
from environment.state_handling import get_num_configs

# ========================================
# ==========   CONFIG   ==========
# ========================================
CONTAMINATION_FACTOR = 0.05

# ========================================
# ==========   GLOBALS   ==========
# ========================================
CLASSIFIER = None
SCALER = None


def __get_classifier():
    global CLASSIFIER
    if not CLASSIFIER:
        CLASSIFIER = IForest(random_state=42, contamination=CONTAMINATION_FACTOR)
    return CLASSIFIER


def __init_scaler(train_set):
    global SCALER
    SCALER = StandardScaler().fit(train_set)  # Feature scaling


def __get_scaler():
    global SCALER
    assert SCALER is not None, "Must first initialize scaler and fit to training set!"
    return SCALER


def preprocess_dataset(dataset):
    # Remove duplicates
    dataset.drop(list(map(lambda header: header+".1", DUPLICATE_HEADERS)), inplace=True, axis=1)  # read_csv adds the .1
    # Remove temporal features
    dataset.drop(["time", "timestamp", "seconds"], inplace=True, axis=1)

    # Remove highly-correlated features
    correlated = ["cpu_ni", "cpu_hi", "tasks_stopped", "alarmtimer:alarmtimer_fired", "alarmtimer:alarmtimer_start",
                  "cachefiles:cachefiles_create", "cachefiles:cachefiles_lookup", "cachefiles:cachefiles_mark_active",
                  "dma_fence:dma_fence_init", "udp:udp_fail_queue_rcv_skb"]
    dataset.drop(correlated, inplace=True, axis=1)

    # Remove vectors generated when the rasp did not have connectivity
    if len(dataset) > 1:  # avoid dropping single entries causing empty dataset
        dataset = dataset.loc[(dataset['connectivity'] == 1)]
    # Remove the connectivity feature because now it is constant
    dataset.drop(['connectivity'], inplace=True, axis=1)

    # Reset index
    dataset.reset_index(inplace=True, drop=True)
    return dataset


def prepare_training_test_sets(dataset):
    # Split into training and test sets
    train_set, test_set = train_test_split(dataset, test_size=0.20, random_state=42, shuffle=True)
    # print("prep", train_set.shape, test_set.shape)

    # Remove train data with Z-score higher than 3
    train_set = train_set[(np.abs(stats.zscore(train_set)) < 3).all(axis=1)]

    return train_set, test_set


def scale_dataset(scaler, dataset):
    dataset = scaler.transform(dataset)
    return dataset


def evaluate_dataset(name, dataset):
    clf = __get_classifier()
    pred = clf.predict(dataset)
    unique_elements, counts_elements = np.unique(pred, return_counts=True)
    print(name, unique_elements, counts_elements,
          "%.2f" % (counts_elements[0] / (counts_elements[0] + counts_elements[1]) * 100), sep="\t")


def train_anomaly_detection():
    # ==============================
    # LOAD, PROCESS, EVALUATE NORMAL DATA
    # ==============================

    # Load data
    # print("Loading CSV data.")
    csv_path_template = os.path.join(CSV_FOLDER_PATH, "{}-behavior.csv")
    df_normal = pd.read_csv(csv_path_template.format("normal"))
    # print("load", df_normal.shape)

    # Preprocess data for ML
    # print("Preprocessing datasets.")
    normal_data = preprocess_dataset(df_normal)
    # print("proc", normal_data.shape)

    # print("Split normal behavior data into training and test set.")
    train_set, test_set = prepare_training_test_sets(dataset=normal_data)
    # print("sets", train_set.shape, test_set.shape)

    # Scale the datasets, turning them into ndarrays
    # print("Scaling dataset features to fit training set.")
    __init_scaler(train_set)
    scaler = __get_scaler()
    train_set = scale_dataset(scaler, train_set)
    test_set = scale_dataset(scaler, test_set)
    # print("scaled", train_set.shape, test_set.shape)

    # Instantiate ML Isolation Forest instance
    # print("Instantiate classifier.")
    clf = __get_classifier()

    # Train model
    # print("Train classifier on training set.")
    clf.fit(train_set)

    # Evaluate model
    print("Evaluate test set and infected behavior datasets.")
    evaluate_dataset("normal", test_set)

    # ==============================
    # REPEAT FOR INFECTED SAMPLES
    # ==============================

    for conf_nr in range(get_num_configs()):
        df_inf = pd.read_csv(csv_path_template.format("infected-c{}".format(conf_nr)))
        inf_data = preprocess_dataset(df_inf)
        inf_data = scale_dataset(scaler, inf_data)
        evaluate_dataset("inf-c{}".format(conf_nr), inf_data)


def detect_anomaly(fingerprint):  # string
    # print("Detecting anomaly.")

    # Transforming FP string to pandas DataFrame
    fp_data = fingerprint.reshape(1, -1)

    headers = ALL_CSV_HEADERS.split(",")
    for header in DUPLICATE_HEADERS:
        found = headers.index(header)
        headers[found+1] = headers[found+1]+".1"  # match the .1 for duplicates appended by read_csv()

    df_fp = pd.DataFrame(fp_data, columns=headers)

    # Sanitizing FP to match IsolationForest
    preprocessed = preprocess_dataset(df_fp)
    scaler = __get_scaler()
    scaled = scale_dataset(scaler, preprocessed)
    # print("Scaled FP to", scaled.shape)

    # Evaluate fingerprint
    clf = __get_classifier()
    pred = clf.predict(scaled)
    assert type(pred) == np.ndarray and len(pred) == 1
    return pred[0]
