
import sys
import nltk
from summary_rewriting.summarization_systems.oracle import jaccard_similarity, copy_rate, repeat_rate
import numpy as np
import logging
from multiprocessing import Process, Manager, cpu_count
from tqdm import *
import itertools

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

target_file = open(sys.argv[1]).readlines()
# sent_window = int(sys.argv[2])


def compute_repeat_rate(in_queue, out_list):
    while True:
        id, txt = in_queue.get()
        if txt is None: # exit signal
            return
        # sents = nltk.sent_tokenize(txt)
        sents = txt.split(" <s> ")
        repeat = repeat_rate(sents)
        out_list.append(repeat)


if __name__ == '__main__':
    num_workers = int(cpu_count() * 0.5)
    manager = Manager()
    repeat_rates = manager.list()
    work = manager.Queue(100)

    pool = []
    for i in range(num_workers):
        p = Process(target=compute_repeat_rate, args=(work, repeat_rates))
        p.start()
        pool.append(p)

    iters = itertools.chain(iter(target_file), (None, ) * num_workers)
    with tqdm(total=len(target_file), desc="Repeat rate:") as pbar:
        for id_line in enumerate(iters):
            work.put(id_line)
            pbar.update()

    for p in pool:
        p.join()

    if len(repeat_rates) == 0:
        print("None")
    print("Repeat rate for {}: {} ({})".format(
        sys.argv[1], np.round(np.mean(repeat_rates) * 100, 2), np.round(np.std(repeat_rates) * 100, 2)))
