# Global settings of the environment

# ==============================
# CSV FILES OF COLLECTED FINGERPRINTS
# ==============================

CSV_FOLDER_PATH = "<local-path-to-collected-fingerprints-folders>"
ALL_CSV_HEADERS = "time,timestamp,seconds,connectivity,cpu_us,cpu_sy,cpu_ni,cpu_id,cpu_wa,cpu_hi,cpu_si,tasks_total,tasks_running,tasks_sleeping,tasks_stopped,tasks_zombie,mem_free,mem_used,mem_cache,swap_avail,net_lo_rx,net_lo_tx,net_eth_rx,net_eth_tx,cpu_temp,alarmtimer:alarmtimer_fired,alarmtimer:alarmtimer_start,block:block_bio_backmerge,block:block_bio_remap,block:block_dirty_buffer,block:block_getrq,block:block_touch_buffer,block:block_unplug,cachefiles:cachefiles_create,cachefiles:cachefiles_lookup,cachefiles:cachefiles_mark_active,clk:clk_set_rate,cpu-migrations,cs,dma_fence:dma_fence_init,fib:fib_table_lookup,filemap:mm_filemap_add_to_page_cache,gpio:gpio_value,ipi:ipi_raise,irq:irq_handler_entry,irq:softirq_entry,jbd2:jbd2_handle_start,jbd2:jbd2_start_commit,kmem:kfree,kmem:kmalloc,kmem:kmem_cache_alloc,kmem:kmem_cache_free,kmem:mm_page_alloc,kmem:mm_page_alloc_zone_locked,kmem:mm_page_free,kmem:mm_page_pcpu_drain,mmc:mmc_request_start,net:net_dev_queue,net:net_dev_xmit,net:netif_rx,page-faults,pagemap:mm_lru_insertion,preemptirq:irq_enable,qdisc:qdisc_dequeue,qdisc:qdisc_dequeue,random:get_random_bytes,random:mix_pool_bytes_nolock,random:urandom_read,raw_syscalls:sys_enter,raw_syscalls:sys_exit,rpm:rpm_resume,rpm:rpm_suspend,sched:sched_process_exec,sched:sched_process_free,sched:sched_process_wait,sched:sched_switch,sched:sched_wakeup,signal:signal_deliver,signal:signal_generate,skb:consume_skb,skb:consume_skb,skb:kfree_skb,skb:kfree_skb,skb:skb_copy_datagram_iovec,sock:inet_sock_set_state,task:task_newtask,tcp:tcp_destroy_sock,tcp:tcp_probe,timer:hrtimer_start,timer:timer_start,udp:udp_fail_queue_rcv_skb,workqueue:workqueue_activate_work,writeback:global_dirty_state,writeback:sb_clear_inode_writeback,writeback:wbc_writepage,writeback:writeback_dirty_inode,writeback:writeback_dirty_inode_enqueue,writeback:writeback_dirty_page,writeback:writeback_mark_inode_dirty,writeback:writeback_pages_written,writeback:writeback_single_inode,writeback:writeback_write_inode,writeback:writeback_written"
DUPLICATE_HEADERS = ["qdisc:qdisc_dequeue", "skb:consume_skb", "skb:kfree_skb"]

# ==============================
# RASPBERRY CLIENT
# ==============================

CLIENT_IP = "<Client-IP>"

# ==============================
# C2 SIMULATION
# ==============================

MAX_STEPS_V2 = 1000
MAX_STEPS_V3 = 5
MAX_EPISODES_V4 = 3
MAX_STEPS_V4 = 5
MAX_STEPS_V99 = 500
