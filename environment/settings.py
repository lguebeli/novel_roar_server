# Global settings of the environment

# ==============================
# CSV FILES OF COLLECTED FINGERPRINTS
# ==============================

EVALUATION_CSV_FOLDER_PATH = r".\fingerprints\evaluation"
TRAINING_CSV_FOLDER_PATH = r".\fingerprints\training"
ALL_CSV_HEADERS = "time,timestamp,seconds,connectivity,cpu_us,cpu_sy,cpu_ni,cpu_id,cpu_wa,cpu_hi,cpu_si,tasks_total,tasks_running,tasks_sleeping,tasks_stopped,tasks_zombie,mem_free,mem_used,mem_cache,swap_avail,net_lo_rx,net_lo_tx,net_eth_rx,net_eth_tx,cpu_temp,alarmtimer:alarmtimer_fired,alarmtimer:alarmtimer_start,block:block_bio_backmerge,block:block_bio_remap,block:block_dirty_buffer,block:block_getrq,block:block_touch_buffer,block:block_unplug,cachefiles:cachefiles_create,cachefiles:cachefiles_lookup,cachefiles:cachefiles_mark_active,clk:clk_set_rate,cpu-migrations,cs,dma_fence:dma_fence_init,fib:fib_table_lookup,filemap:mm_filemap_add_to_page_cache,gpio:gpio_value,ipi:ipi_raise,irq:irq_handler_entry,irq:softirq_entry,jbd2:jbd2_handle_start,jbd2:jbd2_start_commit,kmem:kfree,kmem:kmalloc,kmem:kmem_cache_alloc,kmem:kmem_cache_free,kmem:mm_page_alloc,kmem:mm_page_alloc_zone_locked,kmem:mm_page_free,kmem:mm_page_pcpu_drain,mmc:mmc_request_start,net:net_dev_queue,net:net_dev_xmit,net:netif_rx,page-faults,pagemap:mm_lru_insertion,preemptirq:irq_enable,qdisc:qdisc_dequeue,qdisc:qdisc_dequeue,random:get_random_bytes,random:mix_pool_bytes_nolock,random:urandom_read,raw_syscalls:sys_enter,raw_syscalls:sys_exit,rpm:rpm_resume,rpm:rpm_suspend,sched:sched_process_exec,sched:sched_process_free,sched:sched_process_wait,sched:sched_switch,sched:sched_wakeup,signal:signal_deliver,signal:signal_generate,skb:consume_skb,skb:consume_skb,skb:kfree_skb,skb:kfree_skb,skb:skb_copy_datagram_iovec,sock:inet_sock_set_state,task:task_newtask,tcp:tcp_destroy_sock,tcp:tcp_probe,timer:hrtimer_start,timer:timer_start,udp:udp_fail_queue_rcv_skb,workqueue:workqueue_activate_work,writeback:global_dirty_state,writeback:sb_clear_inode_writeback,writeback:wbc_writepage,writeback:writeback_dirty_inode,writeback:writeback_dirty_inode_enqueue,writeback:writeback_dirty_page,writeback:writeback_mark_inode_dirty,writeback:writeback_pages_written,writeback:writeback_single_inode,writeback:writeback_write_inode,writeback:writeback_written"
DUPLICATE_HEADERS = ["qdisc:qdisc_dequeue", "skb:consume_skb", "skb:kfree_skb"]

# ==============================
# RASPBERRY CLIENT
# ==============================

CLIENT_IP = "127.0.0.1"

# ==============================
# ANOMALY DETECTION
# ==============================

MAX_ALLOWED_CORRELATION_IF = 0.99
MAX_ALLOWED_CORRELATION_AE = 0.98
DROP_CONNECTIVITY = ["connectivity"]
DROP_TEMPORAL = ["time", "timestamp", "seconds"]
DROP_CONSTANT = ["cpu_ni", "cpu_hi", "tasks_stopped", "alarmtimer:alarmtimer_fired",
                 "alarmtimer:alarmtimer_start", "cachefiles:cachefiles_create",
                 "cachefiles:cachefiles_lookup", "cachefiles:cachefiles_mark_active",
                 "dma_fence:dma_fence_init", "udp:udp_fail_queue_rcv_skb"]
DROP_UNSTABLE = ["cpu_wa", "mem_free", "mem_used", "mem_cache", "swap_avail", "net_lo_rx", "net_lo_tx", "net_eth_rx",
                 "net_eth_tx", "cpu_temp", "filemap:mm_filemap_add_to_page_cache", "ipi:ipi_raise",
                 "jbd2:jbd2_start_commit", "kmem:kfree", "kmem:kmalloc", "kmem:mm_page_alloc_zone_locked",
                 "kmem:mm_page_pcpu_drain", "net:net_dev_queue", "net:net_dev_xmit", "qdisc:qdisc_dequeue",
                 "random:mix_pool_bytes_nolock", "rpm:rpm_suspend", "skb:consume_skb", "tcp:tcp_probe",
                 "timer:timer_start", "workqueue:workqueue_activate_work", "writeback:writeback_pages_written",
                 "writeback:writeback_written"]

# ==============================
# C2 SIMULATION
# ==============================

MAX_STEPS_V2 = 1000

MAX_EPISODES_V3 = 250
MAX_STEPS_V3 = 20  # avg 4.5 steps

MAX_EPISODES_V4 = 10_000
SIM_CORPUS_SIZE_V4 = 4000  # 4000 for 20 1s steps with 200 bytes/s

MAX_EPISODES_V5 = 10_000
SIM_CORPUS_SIZE_V5 = 4000

MAX_EPISODES_V6 = 10_000
SIM_CORPUS_SIZE_V6 = 4000

MAX_EPISODES_V7 = 10_000
SIM_CORPUS_SIZE_V7 = 4000

MAX_EPISODES_V8 = 500
SIM_CORPUS_SIZE_V8 = 4000

"""Hyperparameters for V20 (DDQL mit Normal AD)"""
LEARN_RATE_V20 = 0.005
DISCOUNT_FACTOR_V20 = 0.9
EPSILON_V20 = 0.4
DECAY_RATE_V20 = 0.01
SIM_CORPUS_SIZE_V20 = 4000
MAX_EPISODES_V20 = 300

"""Hyperparameters for V21 (DDQL mit Ideal AD)"""
LEARN_RATE_V21 = 0.005
DISCOUNT_FACTOR_V21 = 0.9
EPSILON_V21 = 0.4
DECAY_RATE_V21 = 0.01
SIM_CORPUS_SIZE_V21 = 4000
MAX_EPISODES_V21 = 1000

"""Hyperparameters for V22 (Sarsa mit Normal AD)"""
LEARN_RATE_V22 = 0.005
DISCOUNT_FACTOR_V22 = 0.9
EPSILON_V22 = 0.1
DECAY_RATE_V22 = 0.01
SIM_CORPUS_SIZE_V22 = 4000
MAX_EPISODES_V22 = 1000

"""Hyperparameters for V23 (Sarsa mit Ideal AD)"""
LEARN_RATE_V23 = 0.005
DISCOUNT_FACTOR_V23 = 0.9
EPSILON_V23 = 0.1
DECAY_RATE_V23 = 0.01
SIM_CORPUS_SIZE_V23 = 4000
MAX_EPISODES_V23 = 100

"""Hyperparameters for V24 (PPO mit Normal AD)"""
LEARN_RATE_V24 = 0.0001                 # Optimizer step size
CLIP_EPSILON_V24 = 0.1                  # PPO update limit
GAMMA_V24 = 0.99                        # Reward discount
LAMBDA_V24 = 0.95                       # GAE smoothing
VALUE_COEF_V24 = 0.5                    # Critic loss weight
ENTROPY_COEF_INITIAL_V24 = 0.5          # Start exploration
ENTROPY_COEF_DECAY_V24 = 0.995          # Reduce exploration -- higher value = slower decay
MIN_ENTROPY_COEF_V24 = 0.0001           # Keep some exploration
EPOCHS_V24 = 5                          # Training passes per batch
BATCH_SIZE_V24 = 32                     # Samples per training step
SINGLE_EPISODE_LENGTH_V24 = 4000
MAX_EPISODES_V24 = 10_000

"""Hyperparameters for V25 (PPO mit Ideal AD)"""
LEARN_RATE_V25 = 0.0001                 # Optimizer step size
CLIP_EPSILON_V25 = 0.1                  # PPO update limit
GAMMA_V25 = 0.99                        # Reward discount
LAMBDA_V25 = 0.95                       # GAE smoothing
VALUE_COEF_V25 = 0.5                    # Critic loss weight
ENTROPY_COEF_INITIAL_V25 = 0.5          # Start exploration
ENTROPY_COEF_DECAY_V25 = 0.995          # Reduce exploration
MIN_ENTROPY_COEF_V25 = 0.0001           # Keep some exploration
EPOCHS_V25 = 5                          # Training passes per batch
BATCH_SIZE_V25 = 32                     # Samples per training step
SINGLE_EPISODE_LENGTH_V25 = 4000
MAX_EPISODES_V25 = 100