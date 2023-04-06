# import transformers
from os.path import abspath, dirname, split, join, realpath

from transformers import (
    BloomForCausalLM, 
    GPT2LMHeadModel,
    GPTNeoForCausalLM,
    GPTJForCausalLM, 
    T5ForConditionalGeneration,
    MT5ForConditionalGeneration,
    MT5Config
)

# from bert_chinese_ner.models.transformers.models.bert.tokenization_bert import BertTokenizer
# from .models.transformers.models.bert.tokenization_bert import BertTokenizer


class Config:
    # ROOT_DIR = join(*split(abspath(dirname("__file__")))[:-1])
    ROOT_DIR = join(*split(abspath(dirname("__file__"))))
    # ROOT_DIR = dirname(realpath("__file__"))
    #ROOT_DIR = abspath(dirname(__file__))

    DATA_DIR = join(ROOT_DIR, "data")
    OUTPUT_PATH = join(ROOT_DIR, "outputs")
    MODEL_PATH = join(OUTPUT_PATH, "mt5-small-checkpoint")
    FILENAME = "soft_prompt.pt"
    # IMG_PATH = join(OUTPUT_PATH, "images", "loss_metric.png")
    # OUTPUT_JSON = join(OUTPUT_PATH, "history.json")
    # OUTPUT_CSV = join(OUTPUT_PATH, "history.csv")

    # The name of the dataset to use (via the datasets library).
    # * Notice that ensure to make DATASET_NAME = None if we want to train with our own dataset. 
    DATASET_NAME = None
    # DATASET_NAME = "amazon_reviews_multi"
    DATASET_CONFIG_NAME = "en"
    summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
    }
    TEXT_COLUMN = "text"
    SUMMARY_COLUMN = "summary"
    TRAINING_FILE = "summary_train.json"
    VALIDATION_FILE = "summary_validation.json"
    TEST_FILE = None # "summary_data_test.json"
    TRAIN_VALID_SPLIT_RATIO = 0.15

    CONFIG_NAME = MT5Config
    MODEL_NAME_OR_PATH = "google/mt5-small"
    MODEL_TYPE = None
    TOKENIZER_NAME = None
    
    MODEL_TYPE = "mt5"
    MODEL_MAPPING = {
        "gpt2": ("gpt2", GPT2LMHeadModel),
        "gpt-neo": ("EleutherAI/gpt-neo-125M", GPTNeoForCausalLM),
        "gpt-j": ("EleutherAI/gpt-j-6B", GPTJForCausalLM),
        "t5": ("t5-small", T5ForConditionalGeneration),
        "mt5": ("google/mt5-small", MT5ForConditionalGeneration),
        "bloom": ("bigscience/bloom-560m", BloomForCausalLM),
        "mt0": ("bigscience/mt0-small", None)
    }

    # class ModelMapping(Enum):
    #     gpt2 = GPT2LMHeadModel
    #     gpt_neo = GPTNeoForCausalLM

    TRAIN_BATCH_SIZE = 1
    EVAL_BATCH_SIZE = 1
    TEST_BATCH_SIZE = None
    SOURCE_PREFIX = None
    MAX_INPUT_LEN = 512
    MAX_TARGET_LEN = 32
    PAD_TO_MAX_LENGTH = True
    IGNORE_PAD_TOKEN_FOR_LOSS = True
    
    NUM_WORKERS = 4
    EPOCHS = 2
    EVERY_N_STEP = 2

    # How to initialize the soft prompt: "from_sample_of_embeddings", "from_class_label", "from_string"
    INITIALIZE_METHOD = "from_sample_of_embeddings"
    # the initialize length of tokens of soft prompt
    N_TOKENS = 20
    # initialize from class labels
    CLASS_LABELS = ["True", "False"]
    # CLASS_LABELS = ["commonsense", "pronoun", "resolution"]
    # CLASS_LABELS = ["commonsense", "reasoning", "reading", "comprehension"]
    # TypeError: TextEncodeInput must be Union[TextInputSequence, Tuple[InputSequence, InputSequence]]
    # initialize from string
    INITIAL_STRING = "Summarize the following transcript"

    # BASE_MODEL_NAME = "bert-base-chinese"
    # VOCAB_FILE = join("data", BASE_MODEL_NAME, "vocab.txt")
    # # LABELS = ['B-LOC', 'B-ORG', 'B-PER', 'I-LOC', 'I-ORG', 'I-PER', 'O']


    # TOKENIZER = BertTokenizer.from_pretrained(
    #     BASE_MODEL_NAME,
    #     do_lower_case=False
    # )

    
    # optimizer parameters
    WEIGHT_DECAY = 0.01
    LEARNING_RATE = 0.01 # 3e-5
    GRAD_CLIP = 1.0

    # scheduler parameters
    SCHEDULER = "linear" # "linear" or "cosine"
    GRADIENT_ACCUMULATION_STEPS = None
    WARMUP_PROPORTION = 0.1
    NUM_CYCLES=0.5

    INFERENCE_TEXT = "Initially, I wasn't sure what I will find, but I was surprised to find that it was very simple to unpack, download the app, pair the watch to your phone and have it working effectively. It's not loaded with a lot of functions that will confuse you, but the basic format of tracking your steps and your sleep is excellent. It is quite easy to set a goal for your daily workout or exercise routine. It also allows you to get a little more sophisticated and pair your phone where you can answer calls from your watch. I personally believe this is a good product for anyone who wants to start tracking their workout and more."


