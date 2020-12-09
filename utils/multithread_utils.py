import multiprocessing

def processed_by_multi_thread(function, multi_range):
    num_thread = int(multiprocessing.cpu_count()/2)
    pool = multiprocessing.Pool(num_thread)
    res = pool.map(function, multi_range)
    pool.close()
    pool.join()
    return res