import logging

import pandas as pd
import time

from project2.gridworld import GridWorld
from project2.movingCar import MovingCar
from project2.secret import Secret

DATA_DIR = 'data'
SMALL = DATA_DIR + '/small.csv'
MEDIUM = DATA_DIR + '/medium.csv'
LARGE = DATA_DIR + '/large.csv'

def small():
    small_data = pd.read_csv(SMALL)

    # rewards for a given tile
    sp_r = small_data[['s', 'r']].drop_duplicates().sort_values('s').r.values.reshape((10, 10))

    # grid is of size N x N
    N = 10

    g = GridWorld(N, sp_r, 0.95)

    start = time.time()
    g.update(50)
    logging.info('took {}s for 50 iterations'.format(time.time() - start))
    g.output_policy()


def logging_config():
    numeric_level = getattr(logging, 'INFO')
    logging.basicConfig(level=numeric_level,
                        format='%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S')

if __name__ == '__main__':
    logging_config()
    small()
