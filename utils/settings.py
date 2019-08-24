from os.path import abspath, dirname, join
import os

PROJ_DIR = join(abspath(dirname(__file__)), '..')
DATA_DIR = join(PROJ_DIR, 'data')
OUT_DIR = join(PROJ_DIR, 'out')
EMB_DATA_DIR = join(DATA_DIR, 'emb')
GLOBAL_DATA_DIR = join(DATA_DIR, 'global')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(EMB_DATA_DIR, exist_ok=True)



LEARNING_RATE = 0.001
EMB_DIM = 100
WEIGHT_SIZE = 64
EPOCHES = 50
BATCH_SIZE = 80
TRAIN_SCALE = 12000
TEST_SCALE = 2800
VERBOSE = 1
MAX_AUTHOR = 100
MAX_WORD = 500
MAX_PAPER = 100
NEG_SAMPLE = 8
TEST_SAMPLE = 19
TEST_BATCH_SIZE = 160

# parameter for regular
ALPHA = 1e-6  
# parameter for exponential sum
BETA = 0.5

ALGORITHM = 1

