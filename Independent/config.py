# =============== Basic Configurations ===========
TEXTURE_W = 512
TEXTURE_H = 512
TEXTURE_DIM = 16
USE_PYRAMID = True
VIEW_DIRECTION = True


# =============== Train Configurations ===========
DATA_DIR = ''
CHECKPOINT_DIR = ''
LOG_DIR = ''
TRAIN_SET = ['{:05d}'.format(i) for i in range(10)]
EPOCH = 1000
BATCH_SIZE = 2
CROP_W = 512
CROP_H = 512
LEARNING_RATE = 1e-3
BETAS = '0.9, 0.999'
L2_WEIGHT_DECAY = '0.01, 0.001, 0.0001, 0'
EPS = 1e-8
LOAD = None
LOAD_STEP = 0
EPOCH_PER_CHECKPOINT = 10
MASK_EPOCH = 10
SAMPLES = 10

# =============== Test Configurations ============
TEST_LOAD = ''
TEST_DATA_DIR = ''
TEST_SET = ['{:05d}'.format(i) for i in range(10)]
SAVE_DIR = ''
OUT_MODE = 'image'
FPS = 60
