#
# Moved functions from memorysort.py to this module in order to support multiprocessing
# within interactie jupyter notebooks (the functions are not picklable, which doesn't matter
# when running from the command line, but does matter when running interactie jupyter notebooks)
# 

import logging
import tracelogger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import numpy as np
import pandas as pd
pd.options.mode.copy_on_write = True # avoid warnings
import os
import time
from multiprocessing import Process, Pipe

# Random number generator seed
seed = np.random.seed(123456789)

def wrapper(child_end, f, *args, **kwargs):
    child_end.send(f(*args, **kwargs))


def run_safely(f, *args, **kwargs):

    (parent_end, child_end) = Pipe()
    p = Process(target=wrapper, args=(child_end, f, *args), kwargs=kwargs)
    p.start()
    logging.info(f'Waiting for process {p.pid}')
    p.join()

    logging.info(f'Process {p.pid} exited {p.exitcode}')

    if(p.exitcode == 0):
        elapsed = parent_end.recv()
    else:
        elapsed = None
    p.close()
    parent_end.close()
    child_end.close()

    return elapsed

def mapped_sort(name, n, dir):

    logger.info("Using Memmap\n")

    filename = f'{dir}data{name}.dat'
    logger.info(f"Creating {name} dataset with {n} elements in mapped file {filename}")
    data = np.memmap(filename, dtype='float64', mode='w+', shape=(n,))
    rng = np.random.default_rng(seed)
    data[:] = rng.uniform(low=0.0, high=2**64+0.1, size=n)
    logger.debug(f"\n{data[0:3]}\n{data[-3:]}")
    if logger.isEnabledFor(logging.TRACE):
        logger.trace(f"{data.min()}, {data.max()}")
    data.flush()
    del data  # Close the memmap

    logger.info(f"Sorting {name} dataset with {n} elements in mapped file")
    data = np.memmap(filename, dtype='float64', mode='r+', shape=(n,))
    start_perf_counter = time.perf_counter()
    data.sort(kind='quicksort')
    end_perf_counter = time.perf_counter()
    elapsed_perf_counter = end_perf_counter - start_perf_counter
    
    logger.info(f"Elapsed time (perf_counter): {elapsed_perf_counter:.6f} seconds")
    logger.debug(f"\n{data[0:3]}\n{data[-3:]}")

    data.flush()
    os.remove(filename)

    return elapsed_perf_counter

def virtual_sort(name, n):

    logger.info("Using Virtual Memory\n")

    logger.info(f"Creating {name} dataset with {n} elements in memory")
    data = np.ndarray(n, dtype='float64')
    rng = np.random.default_rng(seed)
    data[:] = rng.uniform(low=0.0, high=2**64+0.1, size=n)
    logger.info(f"\n{data[0:3]}\n{data[-3:]}")
    if logger.isEnabledFor(logging.TRACE):
        logger.trace(f"{data.min()}, {data.max()}")
    
    logger.info(f"Sorting {name} dataset with {n} elements in memory")
    start_perf_counter = time.perf_counter()
    data.sort(kind='quicksort')
    end_perf_counter = time.perf_counter()
    elapsed_perf_counter = end_perf_counter - start_perf_counter
    
    logger.info(f"Elapsed time (perf_counter): {elapsed_perf_counter:.6f} seconds")
    logger.debug(f"\n{data[0:3]}\n{data[-3:]}")

    return elapsed_perf_counter

