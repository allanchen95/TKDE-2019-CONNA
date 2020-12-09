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

NEW_DATA_DIR = 'OAG_WhoIsWho_data/'
# os.makedirs(OUT_DIR, exist_ok=True)
# os.makedirs(EMB_DATA_DIR, exist_ok=True)

# FOR TEST
LEARNING_RATE = 0.001
EMB_DIM = 100
WEIGHT_SIZE = 64
EPOCHES = 50
BATCH_SIZE = 80
TRAIN_SCALE = 500
TEST_SCALE = 100
TEST_SCALE_T = 800
VERBOSE = 1

MAX_PER_AUTHOR = 100
MAX_PER_WORD = 50
MAX_AUTHOR = 640
MAX_WORD = 2600
MAX_PAPER = 80

NEG_SAMPLE = 8
TEST_SAMPLE = 9
TEST_SAMPLE_T = 19
TEST_BATCH_SIZE = 40
TEST_BATCH_SIZE_T = 160

RANK_TEST_SCALE = 2000
RANK_TEST_SAMPLE = 19
RANK_TEST_BATCH_SIZE = 40

FEATURE_SIZE = 44

# parameter for regular
ALPHA = 1e-6  
# parameter for exponential sum
BETA = 0.5

ALGORITHM = 1
