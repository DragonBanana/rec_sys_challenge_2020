
DEBUG=True

USE_CUDA=False

#INPUT="tweet_tokens/training/tweet_tokens"         # complete dataset
INPUT="tweet_tokens/evaluation/tweet_tokens_clean_unique.csv"   # single day
OUTPUT="tweet_tokens/embeddings/test_embeddings.csv.gz"

BERT_MODEL="models/distilbert-base-multilingual-cased"

SENTENCES_NUMBER=10000

#MAX_SEQ_LENGTH=281

BATCH_SIZE=128

#EMBEDDINGS_TYPE="sentence"

#LAYERS="all"

PADDED=False
TOKENIZED=True

python produce_embeddings.py \
    --bert_model=$BERT_MODEL \
    --input=$INPUT \
    --output=$OUTPUT \
    --batch_size=$BATCH_SIZE \
    --sentences_number=$SENTENCES_NUMBER \
    --debug=$DEBUG \
    --tokenized=$TOKENIZED \
    #--padded=$PADDED \
    #--use_cuda=$USE_CUDA \