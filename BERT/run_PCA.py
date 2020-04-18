
import numpy as np

from PCA import IPCA


TRAIN_PATH = "tweet_tokens/embeddings/day_1/embeddings_text_tokens_clean_days_1_unique_04_COMPLETE.csv"
#VALID_PATH = "tweet_tokens/embeddings/day_1/embeddings_text_tokens_clean_days_1_unique_10.csv"

DEVICE = "cpu"

BATCH_SIZE = 64

NUM_COMPONENTS = 64


def read_text_embeddings(emb_file):
    if i == 0:
        f = open(path, 'r')
        text_embeddings = np.genfromtxt(emb_file, delimiter=",", skip_header=1, usecols=range(1,768), max_rows=100000)
    
    return text_embeddings


if __name__ == "__main__":

    inc_pca = IPCA(num_components=NUM_COMPONENTS)
    
    # for 10 times, read 100k rows
    for i in range(10):
        
        print("Iteration : ", i)
        
        if i == 0:
            emb_file = open(TRAIN_PATH, "r")
            text_embeddings = np.genfromtxt(emb_file, delimiter=",", skip_header=1, usecols=range(1,768), max_rows=100000)
        else:
            text_embeddings = np.genfromtxt(emb_file, delimiter=",", usecols=range(1,768), max_rows=100000)

        transformed_embeddings, reconstructed_embeddings, loss = inc_pca.fit_transform(text_embeddings, batch_size=BATCH_SIZE, with_loss=True)

        #print("Transformed embeddings : ", transformed_embeddings)
        print("PCA projection loss : ", loss)
        
    emb_file.close()