import os

import numpy as np
import pandas as pd
from pyod.models.iforest import IForest
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ========================================
# ==========   CONFIG   ==========
# ========================================
CSV_FILES_FOLDER = "C:/Users/jluec/Desktop/fingerprints"
CONTAMINATION_FACTOR = 0.05

# ========================================
# ==========   GLOBALS   ==========
# ========================================
CLASSIFIER = None


def get_classifier():
    global CLASSIFIER
    if not CLASSIFIER:
        CLASSIFIER = IForest(random_state=42, contamination=CONTAMINATION_FACTOR)
    return CLASSIFIER


def preprocess_dataset(dataset):
    # Remove duplicates: 'qdisc:qdisc_dequeue', 'skb:consume_skb', 'skb:kfree_skb'
    dataset.drop(["qdisc:qdisc_dequeue.1", "skb:consume_skb.1", "skb:kfree_skb.1"], inplace=True, axis=1)
    # Remove temporal features
    dataset.drop(["time", "timestamp", "seconds"], inplace=True, axis=1)

    # Remove highly-correlated features
    # corr = df_normal.corr()
    correlated = ["cpu_ni", "cpu_hi", "tasks_stopped", "alarmtimer:alarmtimer_fired", "alarmtimer:alarmtimer_start",
                  "cachefiles:cachefiles_create", "cachefiles:cachefiles_lookup", "cachefiles:cachefiles_mark_active",
                  "dma_fence:dma_fence_init", "udp:udp_fail_queue_rcv_skb"]
    dataset.drop(correlated, inplace=True, axis=1)

    # Remove vectors generated when the rasp did not have connectivity
    dataset = dataset.loc[(dataset['connectivity'] == 1)]
    # Remove the connectivity feature because now it is constant
    dataset.drop(['connectivity'], inplace=True, axis=1)

    # Reset index
    dataset.reset_index(inplace=True, drop=True)
    return dataset


def prepare_training_test_sets(dataset):
    # Split into training and test sets
    train_set, test_set = train_test_split(dataset, test_size=0.10, random_state=42, shuffle=False)
    print(train_set.shape, test_set.shape)

    # Remove train data with Z-score higher than 3
    train_set = train_set[(np.abs(stats.zscore(train_set)) < 3).all(axis=1)]

    return train_set, test_set


def scale_dataset(scaler, dataset):
    dataset = scaler.transform(dataset)
    return dataset


def evaluate_dataset(name, dataset):
    clf = get_classifier()
    pred = clf.predict(dataset)
    unique_elements, counts_elements = np.unique(pred, return_counts=True)
    print(name, "\t", unique_elements, "\t", counts_elements)


def train_anomaly_detection():
    # Load data
    print("Loading CSV data.")
    csv_path_template = os.path.join(CSV_FILES_FOLDER, "{}-behavior.csv")
    df_normal = pd.read_csv(csv_path_template.format("normal"))
    # df_inf_c0 = pd.read_csv(csv_path_template.format("infected-c0"))
    df_inf_c1 = pd.read_csv(csv_path_template.format("infected-c1"))

    # Preprocess data for ML
    print("Preprocessing datasets.")
    normal_data = preprocess_dataset(df_normal)
    infected_c1_data = preprocess_dataset(df_inf_c1)

    print("Split normal behavior data into training and test set.")
    train_set, test_set = prepare_training_test_sets(dataset=normal_data)

    # Scale the datasets, turning them into ndarrays
    print("Scaling dataset features to fit training set.")
    scaler = StandardScaler().fit(train_set)  # Feature scaling
    train_set = scale_dataset(scaler, train_set)
    test_set = scale_dataset(scaler, test_set)
    infected_c1_data = scale_dataset(scaler, infected_c1_data)

    # Instantiate ML Isolation Forest instance
    print("Instantiate classifier.")
    clf = get_classifier()

    # Train model
    print("Train classifier on training set.")
    clf.fit(train_set)

    # Evaluate model
    print("Evaluate test set and infected behavior datasets.")
    evaluate_dataset("normal", test_set)
    # evaluate_dataset("inf-c0", infected_c0_data)
    evaluate_dataset("inf-c1", infected_c1_data)


def detect_anomaly(fingerprint):
    clf = get_classifier()

    # Evaluate fingerprint
    single_sample = fingerprint.reshape(1, -1)
    pred = clf.predict(single_sample)
    assert type(pred) == np.ndarray and len(pred) == 1
    return pred[0]
