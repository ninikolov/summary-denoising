
import sys
from summary_rewriting.summarization_systems.oracle import copy_rate
import numpy as np
import logging
import itertools
from multiprocessing import Process, Manager, cpu_count
from tqdm import *
from distance import nlevenshtein


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
source_file = open(sys.argv[1]).readlines()
target_file = open(sys.argv[2]).readlines()


def get_copy_rate(in_queue, out_list):
    while True:
        id, pair = in_queue.get()
        if pair is None: # exit signal
            return
        src_line, tgt_line = pair
        out_list.append(copy_rate(src_line, tgt_line))


if __name__ == '__main__':
    logging.info("Computing copy rate for {}-{}".format(sys.argv[1], sys.argv[2]))

    num_workers = int(cpu_count() * 0.5)
    manager = Manager()
    output = manager.list()
    work = manager.Queue(100)

    pool = []
    for i in range(num_workers):
        p = Process(target=get_copy_rate, args=(work, output))
        p.start()
        pool.append(p)

    iters = itertools.chain(zip(iter(source_file), iter(target_file)),
                            (None, ) * num_workers)

    with tqdm(total=len(source_file), desc="Copy rate") as pbar:
        for id_line_pair in enumerate(iters):
            work.put(id_line_pair)
            pbar.update()

    for p in pool:
        p.join()

    print("Copy rate: {} ({})".format(np.round(np.mean(output) ** 2, 2), np.round(np.std(output) ** 2, 2)))
