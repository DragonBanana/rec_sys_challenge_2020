
DEBUG=True

USE_CUDA=False

#DATA_DIR="tweet_tokens/training/tweet_tokens/padded"  # complete dataset
DATA_DIR="tweet_tokens/subsample/tweet_tokens/train_day_1"  # single day
OUTPUT_DIR="tweet_tokens/embeddings/train_day_1"

BERT_MODEL="models/distilbert-base-multilingual-cased"

SENTENCES_NUMBER=1000000

#MAX_SEQ_LENGTH=281

BATCH_SIZE=8

#EMBEDDINGS_TYPE="sentence"

#LAYERS="all"

PADDED=False
TOKENIZED=True

python produce_embeddings.py \
    --bert_model=$BERT_MODEL \
    --input_dir=$DATA_DIR \
    --output_dir=$OUTPUT_DIR \
    --batch_size=$BATCH_SIZE \
    --sentences_number=$SENTENCES_NUMBER \
    --debug=$DEBUG \
    --tokenized=$TOKENIZED \
    #--padded=$PADDED \
    #--use_cuda=$USE_CUDA \