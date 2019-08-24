from os.path import abspath, dirname, join
import os

PROJ_DIR = join(abspath(dirname(__file__)), '..')
DATA_DIR = join(PROJ_DIR, 'data')
OUT_DIR = join(PROJ_DIR, 'out')
EMB_DATA_DIR = join(DATA_DIR, 'emb')
GLOBAL_DATA_DIR = join(DATA_DIR, 'global')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(EMB_DATA_DIR, exist_ok=True)


# For fine_tune
# LEARNING_RATE = 0.001
# EMB_DIM = 100
# WEIGHT_SIZE = 64
# EPOCHES = 50
# BATCH_SIZE = 80
# TRAIN_SCALE = 12000
# TEST_SCALE = 16000
# TEST_SCALE_T = 5600
# VERBOSE = 1
# MAX_AUTHOR = 100
# MAX_WORD = 500
# MAX_PAPER = 100
# NEG_SAMPLE = 8
# TEST_SAMPLE = 9
# TEST_SAMPLE_T = 20
# TEST_BATCH_SIZE = 180
# TEST_BATCH_SIZE_T = 160

# RANK_TEST_SCALE = 2800
# RANK_TEST_SAMPLE = 20
# RANK_TEST_BATCH_SIZE = 160

# FEATURE_SIZE = 88

# # parameter for regular
# ALPHA = 1e-6  
# # parameter for exponential sum
# BETA = 0.5

# ALGORITHM = 1

# FOR TEST
LEARNING_RATE = 0.001
EMB_DIM = 100
WEIGHT_SIZE = 64
EPOCHES = 50
BATCH_SIZE = 80
TRAIN_SCALE = 12000
TEST_SCALE = 16000
TEST_SCALE_T = 1600
VERBOSE = 1
MAX_AUTHOR = 100
MAX_WORD = 500
MAX_PAPER = 100
NEG_SAMPLE = 8
TEST_SAMPLE = 9
TEST_SAMPLE_T = 20
TEST_BATCH_SIZE = 180
TEST_BATCH_SIZE_T = 160

RANK_TEST_SCALE = 800
RANK_TEST_SAMPLE = 20
RANK_TEST_BATCH_SIZE = 160

FEATURE_SIZE = 88

# parameter for regular
ALPHA = 1e-6  
# parameter for exponential sum
BETA = 0.5

ALGORITHM = 1
