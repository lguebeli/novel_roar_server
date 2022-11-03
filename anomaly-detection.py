import os

import numpy as np
import pandas as pd
from pyod.models.iforest import IForest
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CSV_FILES_FOLDER = os.path.abspath(os.path.curdir)
CONTAMINATION_FACTOR = 0.05


def preprocess_dataset(dataset):
    # Remove duplicates: 'qdisc:qdisc_dequeue', 'skb:consume_skb', 'skb:kfree_skb'
    dataset.drop(["qdisc:qdisc_dequeue.1", "skb:consume_skb.1", "skb:kfree_skb.1"], inplace=True, axis=1)
    # Remove temporal features
    dataset.drop(["time", "timestamp", "seconds"], inplace=True, axis=1)

    # Remove highly-correlated features
    dataset.dropna(how="all", axis=0, inplace=True).dropna(how="all", axis=1, inplace=True)
    # corr = df_normal.corr()
    # correlated = ["cpu_ni", "cpu_hi", "tasks_stopped", "alarmtimer:alarmtimer_fired", "alarmtimer:alarmtimer_start",
    #               "cachefiles:cachefiles_create", "cachefiles:cachefiles_lookup", "cachefiles:cachefiles_mark_active",
    #               "dma_fence:dma_fence_init", "udp:udp_fail_queue_rcv_skb"]
    # dataset.drop(correlated, inplace=True, axis=1)

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

    # Scale the datasets
    # Feature scaling
    scaler = StandardScaler().fit(train_set)
    # Transform training set
    train_set = scaler.transform(train_set)
    # Transform test set
    test_set = scaler.transform(test_set)

    return train_set, test_set


def evaluate_dataset(clf, dataset):
    pred = clf.predict(dataset)
    unique_elements, counts_elements = np.unique(pred, return_counts=True)
    print("\t", unique_elements, "    ", counts_elements)


def train_anomaly_detection():
    # Load data
    csv_path_template = os.path.join(CSV_FILES_FOLDER, "{}-behavior.csv")
    df_normal = pd.read_csv(csv_path_template.format("normal"))
    # df_inf_c0 = pd.read_csv(csv_path_template.format("infected-c0"))
    df_inf_c1 = pd.read_csv(csv_path_template.format("infected-c1"))

    # Preprocess data for ML
    normal_data = preprocess_dataset(df_normal)
    infected_c1_data = preprocess_dataset(df_inf_c1)

    train_set, test_set = prepare_training_test_sets(dataset=normal_data)

    # Instantiate ML Isolation Forest instance
    clf = IForest(random_state=42, contamination=CONTAMINATION_FACTOR)

    # Train model
    clf.fit(train_set)

    # Evaluate model
    evaluate_dataset(clf, test_set)
    # evaluate_dataset(clf, infected_c0_data)
    evaluate_dataset(clf, infected_c1_data)

    return clf


def detect_anomaly(clf, fingerprint):
    # Evaluate fingerprint
    single_sample = fingerprint.reshape(1, -1)
    pred = clf.predict(single_sample)
    assert type(pred) == np.ndarray and len(pred) == 1
    return pred[0]
