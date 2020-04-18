
from __future__ import print_function
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional


class VAE(nn.Module):
    def __init__(self, enc, dec):
        super().__init__()
        self.enc = enc
        self.dec = dec
        
    def reparameterize(self, z_mu, z_var):
        if self.training:
            std = torch.exp(0.5*z_var)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(z_mu)
        else:
            return z_mu

    def forward(self, x):
        # encode
        z_mu, z_var = self.enc(x)
        # sample from the distribution having latent parameters z_mu, z_var
        # reparameterize
        x_sample = self.reparameterize(z_mu, z_var)
        # decode
        decoded = self.dec(x_sample)
        return decoded, z_mu, z_var
    
    
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        '''
        Args:
                input_dim: A integer indicating the size of input.
                hidden_dim: A integer indicating the size of hidden dimension.
                z_dim: A integer indicating the latent dimension.
        '''
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, z_dim)
        self.var = nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
        # x is of shape [batch_size, input_dim]
        hidden = functional.relu(self.linear(x))
        # hidden is of shape [batch_size, hidden_dim]
        z_mu = self.mu(hidden)
        # z_mu is of shape [batch_size, latent_dim]
        z_var = self.var(hidden)
        # z_var is of shape [batch_size, latent_dim]
        return z_mu, z_var
    

class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, output_dim):
        '''
        Args:
                z_dim: A integer indicating the latent size.
                hidden_dim: A integer indicating the size of hidden dimension.
                output_dim: A integer indicating the output dimension.
        '''
        super().__init__()
        self.linear = nn.Linear(z_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x is of shape [batch_size, latent_dim]
        hidden = functional.relu(self.linear(x))
        # hidden is of shape [batch_size, hidden_dim]
        predicted = torch.sigmoid(self.out(hidden))
        # predicted is of shape [batch_size, output_dim]
        return predicted
    
    
# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = functional.mse_loss(recon_x, x, reduction='sum')
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def read_text_embeddings(type):
    if type == "train":
        path = TRAIN_PATH
        #max_rows = 1000
    else:
        path = VALID_PATH
        #max_rows = 100
    with open(path, 'r') as f:
        num_cols = len(f.readline().split(','))  # read header
        f.seek(0)
        text_embeddings = np.loadtxt(f, delimiter=",", skiprows=1, usecols=range(1,num_cols), max_rows=1000000, dtype=np.float32)
    
    return text_embeddings


def train(train_set, batch_size, epoch):
    # set the train mode
    model.train()
    # loss of the epoch
    train_loss = 0
    
    train_set_length = len(train_set)
    
    iterator = range(0, train_set_length, batch_size)

    for batch_idx in tqdm(iterator, desc=f"\tTraining : "):

        batch_start = batch_idx
        batch_end = min(batch_start + batch_size, train_set_length)
        
        batch = train_set[batch_start:batch_end]
        batch = batch.float().to(DEVICE)
        
        # update the gradients to zero
        optimizer.zero_grad()
        # forward pass (batch_sample is decode(encode(batch)))
        batch_sample, z_mu, z_var = model(batch)
        # loss 
        loss = loss_function(batch_sample, batch, z_mu, z_var)
        # backward pass
        loss.backward()
        train_loss += loss.item()
        # update the weights
        optimizer.step()

    return train_loss


def test(test_set, batch_size, epoch):
    # set the evaluation mode
    model.eval()
    # test loss for the data
    test_loss = 0

    # we don't need to track the gradients, since we are not updating the parameters during evaluation / testing
    with torch.no_grad():
        
        test_set_length = len(test_set)
    
        iterator = range(0, test_set_length, batch_size)

        for batch_idx in tqdm(iterator, desc=f"\tValidation : "):

            batch_start = batch_idx
            batch_end = min(batch_start + batch_size, test_set_length)

            batch = test_set[batch_start:batch_end]
            batch = batch.float().to(DEVICE)
            
            # reshape the data
            batch = batch.float().to(DEVICE)
            # forward pass
            batch_sample, z_mu, z_var = model(batch)
            # loss 
            loss = loss_function(batch_sample, batch, z_mu, z_var)
            test_loss += loss.item()

    return test_loss