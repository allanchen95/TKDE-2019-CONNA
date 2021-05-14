from os.path import abspath, dirname, join
import os

PROJ_DIR = join(abspath(dirname(__file__)), '..')
# DATA_DIR = join(PROJ_DIR, 'data')
DATA_DIR = 'Essential_Embeddings/'
# OUT_DIR = join(PROJ_DIR, 'out')
# EMB_DATA_DIR = join(DATA_DIR, 'emb')
EMB_DATA_DIR = 'Essential_Embeddings/emb/'
# GLOBAL_DATA_DIR = join(DATA_DIR, 'global')
GLOBAL_DATA_DIR = 'Essential_Embeddings/global/'

NEW_DATA_DIR = 'WhoIsWho_data/'
# os.makedirs(OUT_DIR, exist_ok=True)
# os.makedirs(EMB_DATA_DIR, exist_ok=True)



LEARNING_RATE = 0.001
FINE_TUNE_LR = 0.0005
EMB_DIM = 100
WEIGHT_SIZE = 72
EPOCHES = 20
BATCH_SIZE = 36
TRAIN_SCALE = 9000
TEST_SCALE = 2000
# TRAIN_SCALE = 200
# TEST_SCALE = 100
VERBOSE = 1
MAX_PER_AUTHOR = 100
MAX_PER_WORD = 50
MAX_AUTHOR = 500
MAX_WORD = 1000
MAX_PAPER = 100
# NEG_SAMPLE = 8
# TEST_SAMPLE = 19
NEG_SAMPLE = 9
TEST_SAMPLE = 19
TEST_BATCH_SIZE = 40 

# parameter for regular
ALPHA = 1e-6  
# parameter for exponential sum
BETA = 0.5

ALGORITHM = 1

