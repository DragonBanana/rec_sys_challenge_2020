
import numpy as np
import argparse
import torch
from torch import nn, optim
from torch.nn import functional

from tqdm import tqdm

from VAE import *

'''
parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)'''


torch.manual_seed(42)   # args.seed


TRAIN_PATH = "tweet_tokens/embeddings/train_day_1/embeddings_text_tokens_clean_days_1_unique_04_COMPLETE.csv"
VALID_PATH = "tweet_tokens/embeddings/train_day_1/embeddings_text_tokens_clean_days_1_unique_10_COMPLETE.csv"

MODEL_PATH = "models/first_VAE.model"

DEVICE = "cpu"

LEARNING_RATE = 1e-4

BATCH_SIZE = 128

N_EPOCHS = 1000

PATIENCE = 20

INPUT_DIM = 768
HIDDEN_DIM = 128
LATENT_DIM = 32


encoder = Encoder(INPUT_DIM, HIDDEN_DIM, LATENT_DIM)
decoder = Decoder(LATENT_DIM, HIDDEN_DIM, INPUT_DIM)
model = VAE(encoder, decoder).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


#model = torch.load(MODEL_PATH)


train_set = torch.from_numpy(read_text_embeddings("train"))
print("Training set : ", train_set.size())
valid_set = torch.from_numpy(read_text_embeddings("valid"))
print("Validation set : ", valid_set.size(), "\n")

model = model.float()

best_valid_loss = float('inf')

epoch = 1
early_stopped = False

while epoch <= N_EPOCHS and not early_stopped:
    
    print(f'Epoch {epoch} \n')

    train_loss = train(train_set, BATCH_SIZE, epoch)
    valid_loss = test(valid_set, BATCH_SIZE, epoch)

    train_loss /= len(train_set)
    valid_loss /= len(valid_set)

    print(f'\tTrain Loss: {train_loss:.4f}, Test Loss: {valid_loss:.4f}')

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        counter = 1
    else:
        counter += 1

    if counter > PATIENCE:
        early_stopped = True
        print(f"\nEarly stopped at epoch {epoch}")
        
    epoch += 1
    
    
torch.save(model, MODEL_PATH)