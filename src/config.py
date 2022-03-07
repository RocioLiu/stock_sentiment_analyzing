from os.path import abspath, dirname, split, join

# import transformers


ROOT_DIR = join(*split(abspath(dirname(__file__)))[:-1])
# ROOT_DIR = abspath(dirname(__file__))


# BASE_MODEL_PATH = "../inputs/bert_base_uncased/"
# BASE_MODEL_NAME = 'bert-base-uncased'
TRAINING_FILE = join(ROOT_DIR, "inputs", "twits.json")
TEST_FILE = join(ROOT_DIR, "inputs", "test_twits.json")
OUTPUT_PATH = join(ROOT_DIR, "outputs")
MODEL_PATH = join(OUTPUT_PATH, "checkpoint", "model.pth")
IMG_PATH = join(OUTPUT_PATH, "images", "loss_metric.png")
OUTPUT_JSON = join(OUTPUT_PATH, "history.json")


EPOCHS = 1
EVERY_N_STEP = 1000

CLS_ID = 101
SEP_ID = 102
PAD_ID = 0

LOW_CUTOFF = 5e-6
HIGH_CUTOFF = 15
NEUTRAL_SCORE = 2
DATA_SPLIT_RATIO = 0.15

# TOKENIZER = transformers.BertTokenizer.from_pretrained(
#     BASE_MODEL_NAME,
#     do_lower_case=True
# )

MAX_LEN = 32
TRAIN_BATCH_SIZE = 128 # 32
VAL_BATCH_SIZE = 128 # 8
TEST_BATCH_SIZE = 64  # 8

WEIGHT_DECAY = 0.001
LEARNING_RATE = 0.001 # 3e-5
WARMUP_PROPORTION = 0.1
GRAD_CLIP = 5.0

EMBED_SIZE = 1024
LSTM_SIZE = 512
OUTPUT_SIZE = 5
LSTM_LAYER = 2
DROP_RATE = 0.2

MSG1 = "Google is working on self driving cars, I'm bullish on $goog"
MSG2 = "$MSFT to acquire $ATVI"