import os
import argparse
import json
import numpy as np
import torch
import logging
import gzip

from collections import OrderedDict

from Utils import read_sentences

from sentence_transformers.sentence_transformers.models import BERT, DistilBERT, Pooling
from sentence_transformers.sentence_transformers.SentenceTransformer import SentenceTransformer
from sentence_transformers.sentence_transformers.LoggingHandler import LoggingHandler

from PCA import IPCA


def write_result_file_header(path):
    with gzip.open(path, "wt") as writer:
        string = "tweet_features_tweet_id"
        for i in range(768):
            string += ",embedding_" + str(i)
        string += '\n'
        writer.write(string)


def save_embeddings(path, tweet_ids, embeddings):
    assert len(tweet_ids) == len(embeddings)
    with gzip.open(path, "at") as writer:
        for i in range(len(tweet_ids)):
            string = tweet_ids[i] + ',' + ','.join(map(str, embeddings[i])) + '\n'
            writer.write(string)
        '''embeddings_list = []  # preserve keys order
        for i in range(len(tweet_ids)):
            _dict = {}
            _dict['tweet_id'] = int(tweet_ids[i])
            _dict['embeddings'] = embeddings[i].tolist()
            embeddings_list.append(_dict)
        json_output = json.dumps(embeddings_list)
        writer.write(json_output)'''


def select_device(local_rank, use_cuda):
    if not use_cuda:
        device = torch.device("cpu")
        n_gpu = torch.cuda.device_count()
    elif local_rank == -1 or use_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
        print('GPUs available: %d' % n_gpu)
        print('Using GPU:', torch.cuda.get_device_name())
    else:
        device = torch.device("cuda", local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    
    print("Device: {} - Number  GPUs: {}  - Distributed training: {} \n".format(device, n_gpu, bool(local_rank != -1)))
    
    return device


def parse_arguments():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--input", default=None, type=str, required=True)
    parser.add_argument("--output", default=None, type=str, required=True)
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    
    ## Other parameters
    parser.add_argument("--embeddings_type", default="sentence", type=str)  # 'sentence' or 'word' embeddings
    parser.add_argument("--sentences_number", default=1000, type=int)
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences longer "
                            "than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for predictions.")
    parser.add_argument("--use_cuda",
                        default=False,
                        help="Whether to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help = "local_rank for distributed training on GPUs")
    parser.add_argument("--tokenized", default=False, type=bool, help="Whether or not text is already tokenized")
    parser.add_argument("--padded", default=False, type=bool, help="Whether or not text tokens are already padded to the same length")
    parser.add_argument("--debug", default=False, type=bool)

    return parser.parse_args()


def main():
    
    args = parse_arguments()
    
    print("\nPadded : ", args.padded)
    print("Tokenized : ", args.tokenized)

    device = select_device(args.local_rank, args.use_cuda)
    
    np.set_printoptions(threshold=100)
    
    logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
    
    # Use BERT for mapping tokens to embeddings
    embedding_model = DistilBERT(args.bert_model)

    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = Pooling(embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)

    model = SentenceTransformer(modules=[embedding_model, pooling_model], device=device)
    
    # cycle over data files (batches)
    #for directory, subdirectories, files in os.walk(args.input):
        
    print('\nEmbeddings for : ', args.input)
    
        #for index, file in enumerate(files):

        #    if "00" in file:
        #        header_in_first_line = True
        #    else:
        #        header_in_first_line = False

        #    input_file = os.path.join(directory, file)
        #    output_file = os.path.join(args.output, 'embeddings_' + file)  #.replace('.csv','') + '_COMPLETE.csv')

    write_result_file_header(args.output)

            #print("\nFile number : ", index)
            #print("Embeddings for : ", input_file)
            #print("Output file : ", output_file)
                
    input_reader = open(args.input, "r")
    
    finished = False
    i = 0
    
    while not finished:
        
        print('\nBATCH NUMBER : ', i)
        
        #if i == 2:
        #    tweet_ids, sentences = read_sentences(input_reader, 5, header_first_line=True)
        #else:
        tweet_ids, sentences = read_sentences(input_reader, args.sentences_number, header_first_line=True)
                #for s in sentences:
                #    print(s)
                
        if len(tweet_ids) < args.sentences_number:
            finished = True

        embeddings = model.encode(sentences, already_tokenized=args.tokenized, already_padded=args.padded, batch_size=args.batch_size, convert_to_numpy=True)

        if args.debug:
            #print(embeddings)
            #print('\tEmbeddings number : ', len(embeddings))
            print('\tEmbeddings shape : ', embeddings[0].shape, 'each')

        save_embeddings(args.output, tweet_ids, embeddings)
        
        i += 1
                
    print("\nDone.")

if __name__ == "__main__":
    main()