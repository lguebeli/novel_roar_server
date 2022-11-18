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
RPI_MODEL_PREFIX = "mod4-"
CONTAMINATION_FACTOR = 0.05
ALL_CSV_HEADERS = "time,timestamp,seconds,connectivity,cpu_us,cpu_sy,cpu_ni,cpu_id,cpu_wa,cpu_hi,cpu_si,tasks_total,tasks_running,tasks_sleeping,tasks_stopped,tasks_zombie,mem_free,mem_used,mem_cache,swap_avail,net_lo_rx,net_lo_tx,net_eth_rx,net_eth_tx,cpu_temp,alarmtimer:alarmtimer_fired,alarmtimer:alarmtimer_start,block:block_bio_backmerge,block:block_bio_remap,block:block_dirty_buffer,block:block_getrq,block:block_touch_buffer,block:block_unplug,cachefiles:cachefiles_create,cachefiles:cachefiles_lookup,cachefiles:cachefiles_mark_active,clk:clk_set_rate,cpu-migrations,cs,dma_fence:dma_fence_init,fib:fib_table_lookup,filemap:mm_filemap_add_to_page_cache,gpio:gpio_value,ipi:ipi_raise,irq:irq_handler_entry,irq:softirq_entry,jbd2:jbd2_handle_start,jbd2:jbd2_start_commit,kmem:kfree,kmem:kmalloc,kmem:kmem_cache_alloc,kmem:kmem_cache_free,kmem:mm_page_alloc,kmem:mm_page_alloc_zone_locked,kmem:mm_page_free,kmem:mm_page_pcpu_drain,mmc:mmc_request_start,net:net_dev_queue,net:net_dev_xmit,net:netif_rx,page-faults,pagemap:mm_lru_insertion,preemptirq:irq_enable,qdisc:qdisc_dequeue,qdisc:qdisc_dequeue,random:get_random_bytes,random:mix_pool_bytes_nolock,random:urandom_read,raw_syscalls:sys_enter,raw_syscalls:sys_exit,rpm:rpm_resume,rpm:rpm_suspend,sched:sched_process_exec,sched:sched_process_free,sched:sched_process_wait,sched:sched_switch,sched:sched_wakeup,signal:signal_deliver,signal:signal_generate,skb:consume_skb,skb:consume_skb,skb:kfree_skb,skb:kfree_skb,skb:skb_copy_datagram_iovec,sock:inet_sock_set_state,task:task_newtask,tcp:tcp_destroy_sock,tcp:tcp_probe,timer:hrtimer_start,timer:timer_start,udp:udp_fail_queue_rcv_skb,workqueue:workqueue_activate_work,writeback:global_dirty_state,writeback:sb_clear_inode_writeback,writeback:wbc_writepage,writeback:writeback_dirty_inode,writeback:writeback_dirty_inode_enqueue,writeback:writeback_dirty_page,writeback:writeback_mark_inode_dirty,writeback:writeback_pages_written,writeback:writeback_single_inode,writeback:writeback_write_inode,writeback:writeback_written"
DUPLICATE_HEADERS = ["qdisc:qdisc_dequeue", "skb:consume_skb", "skb:kfree_skb"]

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
    # Remove duplicates: 'qdisc:qdisc_dequeue', 'skb:consume_skb', 'skb:kfree_skb'
    dataset.drop(list(map(lambda header: header+".1", DUPLICATE_HEADERS)), inplace=True, axis=1)  # read_csv adds the .1
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
    # print(train_set.shape, test_set.shape)

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
    print(name, "\t", unique_elements, "\t", counts_elements)


def train_anomaly_detection():
    # Load data
    # print("Loading CSV data.")
    csv_path_template = os.path.join(CSV_FILES_FOLDER, RPI_MODEL_PREFIX + "{}-behavior.csv")
    df_normal = pd.read_csv(csv_path_template.format("normal"))
    df_inf_c0 = pd.read_csv(csv_path_template.format("infected-c0"))
    df_inf_c1 = pd.read_csv(csv_path_template.format("infected-c1"))
    df_inf_c2 = pd.read_csv(csv_path_template.format("infected-c2"))
    df_inf_c3 = pd.read_csv(csv_path_template.format("infected-c3"))
    df_inf_c4 = pd.read_csv(csv_path_template.format("infected-c4"))
    df_inf_c5 = pd.read_csv(csv_path_template.format("infected-c5"))
    df_inf_c6 = pd.read_csv(csv_path_template.format("infected-c6"))
    df_inf_c7 = pd.read_csv(csv_path_template.format("infected-c7"))

    # Preprocess data for ML
    # print("Preprocessing datasets.")
    normal_data = preprocess_dataset(df_normal)
    infected_c0_data = preprocess_dataset(df_inf_c0)
    infected_c1_data = preprocess_dataset(df_inf_c1)
    infected_c2_data = preprocess_dataset(df_inf_c2)
    infected_c3_data = preprocess_dataset(df_inf_c3)
    infected_c4_data = preprocess_dataset(df_inf_c4)
    infected_c5_data = preprocess_dataset(df_inf_c5)
    infected_c6_data = preprocess_dataset(df_inf_c6)
    infected_c7_data = preprocess_dataset(df_inf_c7)

    # print("Split normal behavior data into training and test set.")
    train_set, test_set = prepare_training_test_sets(dataset=normal_data)

    # Scale the datasets, turning them into ndarrays
    # print("Scaling dataset features to fit training set.")
    __init_scaler(train_set)
    scaler = __get_scaler()
    train_set = scale_dataset(scaler, train_set)
    test_set = scale_dataset(scaler, test_set)
    infected_c0_data = scale_dataset(scaler, infected_c0_data)
    infected_c1_data = scale_dataset(scaler, infected_c1_data)
    infected_c2_data = scale_dataset(scaler, infected_c2_data)
    infected_c3_data = scale_dataset(scaler, infected_c3_data)
    infected_c4_data = scale_dataset(scaler, infected_c4_data)
    infected_c5_data = scale_dataset(scaler, infected_c5_data)
    infected_c6_data = scale_dataset(scaler, infected_c6_data)
    infected_c7_data = scale_dataset(scaler, infected_c7_data)

    # Instantiate ML Isolation Forest instance
    # print("Instantiate classifier.")
    clf = __get_classifier()

    # Train model
    # print("Train classifier on training set.")
    clf.fit(train_set)

    # Evaluate model
    print("Evaluate test set and infected behavior datasets.")
    evaluate_dataset("normal", test_set)
    evaluate_dataset("inf-c0", infected_c0_data)
    evaluate_dataset("inf-c1", infected_c1_data)
    evaluate_dataset("inf-c2", infected_c2_data)
    evaluate_dataset("inf-c3", infected_c3_data)
    evaluate_dataset("inf-c4", infected_c4_data)
    evaluate_dataset("inf-c5", infected_c5_data)
    evaluate_dataset("inf-c6", infected_c6_data)
    evaluate_dataset("inf-c7", infected_c7_data)


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
