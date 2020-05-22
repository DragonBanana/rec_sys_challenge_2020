from torch import nn
import torch
from torch.utils.data.dataset import Dataset
from transformers import BertForSequenceClassification, AdamW, BertConfig
# from torchviz import make_dot
from transformers.modeling_bert import BertModel
from Utils.Data.Data import get_dataset, get_feature
import numpy as np
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup
import random
import time
from tqdm import tqdm
import pandas as pd
import gc

from Utils.Base.RecommenderBase import RecommenderBase


class FFNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FFNN, self).__init__()
        # Linear function
        self.fc1 = nn.Linear(input_dim, hidden_dim)

        # Non-linearity
        self.sigmoid = nn.Sigmoid()

        # Linear function (readout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Linear function  # LINEAR
        out = self.fc1(x)

        # Non-linearity  # NON-LINEAR
        out = self.sigmoid(out)

        # Linear function (readout)  # LINEAR
        out = self.fc2(out)
        return out


class BertClassifierDoubleInput(nn.Module):

    def __init__(self, input_size_2, hidden_size_2, hidden_dropout_prob=0.1, ):
        super().__init__()

        self.bert = BertModel.from_pretrained("bert-base-multilingual-cased")

        self.dropout = nn.Dropout(hidden_dropout_prob)

        hidden_size_bert = 768
        self.first_layer = nn.Linear(hidden_size_bert + input_size_2, hidden_size_2)

        self.classifier = nn.Linear(hidden_size_2, 1)

    def forward(
            self,
            input_ids=None,
            input_features=None,  # the second input
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None, ):
        bert_outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = bert_outputs[1]

        pooled_output = torch.cat([pooled_output, input_features.float()], dim=1)

        pooled_output = self.first_layer(pooled_output)

        logits = self.classifier(pooled_output)

        outputs = (logits,) + bert_outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.view(-1).float())
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class CustomDataset(Dataset):
    def __init__(self, df_features: pd.DataFrame,
                 df_tokens_reader: pd.io.parsers.TextFileReader,
                 df_label: pd.DataFrame):

        self.df_features = df_features
        self.df_tokens_reader = df_tokens_reader
        self.df_label = df_label
        self.count = -1

    def __len__(self):
        # TODO DEBUG
        print(f"debug len of custom dataset:{len(self.df_features)}")
        return len(self.df_features)

    def __getitem__(self, index):

        # debug
        print(f"debug-> index is:{index}")

        # if true, update the caches, i.e. self.tensors
        if index % self.df_tokens_reader.chunksize == 0:
            self.count += 1

            df_tokens_cache = self.df_tokens_reader.get_chunk()
            df_tokens_cache.columns = ['tokens']
            start = index * self.df_tokens_reader.chunksize
            end = start + self.df_tokens_reader.chunksize
            df_features_cache = self.df_features.iloc[start:end]
            df_label_cache = self.df_label.iloc[start:end]

            text_series = df_tokens_cache['tokens'].map(lambda x: x.split('\t'))
            print(f"first text_series: {text_series}")
            max_len = get_max_len(text_series)
            attention_masks = np.ones((len(text_series), max_len), dtype=np.int8)
            # padding
            for i in range(len(text_series)):
                i_shifted = i + index  # * self.df_tokens_reader.chunksize
                initial_len = len(text_series[i_shifted])
                miss = max_len - initial_len
                text_series[i_shifted] += [0] * miss
                for j in range(initial_len, max_len):
                    attention_masks[i][j] = 0

            # todo we need to optimize this
            list_arr = []
            for feat in df_features_cache.columns:
                list_arr.append(df_features_cache[feat].values)
            feature_mat = np.array(list_arr)
            del list_arr
            gc.collect()

            text_series = text_series.map(lambda l: [int(elem) for elem in l]).map(lambda x: np.array(x))
            print(f"text_series : {text_series}")

            text_np_mat = np.array(list(text_series))
            # print(f"text_np_mat :\n {text_np_mat}")
            # print(f"text_np_mat type : {type(text_np_mat)}")
            # print(f"text_np_mat 0 type : {type(text_np_mat[0])}")

            print(f"text_np_mat 0 : {text_np_mat[0]}")
            text_tensor = torch.tensor(text_np_mat)
            attention_masks = torch.tensor(attention_masks)
            labels = torch.tensor(df_label_cache['tweet_feature_engagement_is_like']
                                  .map(lambda x: 1 if x else 0).values)
            features = torch.tensor(feature_mat.T)
            self.tensors = [text_tensor, attention_masks, features, labels]

        return tuple(tensor[index - self.count * self.df_tokens_reader.chunksize] for tensor in self.tensors)


class CustomDatasetCap(Dataset):
    def __init__(self, df_features: pd.DataFrame,
                 df_tokens_reader: pd.io.parsers.TextFileReader,
                 df_label: pd.DataFrame,
                 cap: int = 128):

        self.df_features = df_features
        self.df_tokens_reader = df_tokens_reader
        self.df_label = df_label
        self.count = -1
        self.cap = cap

    def __len__(self):
        # TODO DEBUG
        print(f"debug len of custom dataset:{len(self.df_features)}")
        return len(self.df_features)

    def __getitem__(self, index):

        # debug
        print(f"debug-> index is:{index}")

        # if true, update the caches, i.e. self.tensors
        if index % self.df_tokens_reader.chunksize == 0:
            sep_tok_id = 102

            self.count += 1

            df_tokens_cache = self.df_tokens_reader.get_chunk()
            df_tokens_cache.columns = ['tokens']
            start = index * self.df_tokens_reader.chunksize
            end = start + self.df_tokens_reader.chunksize
            df_features_cache = self.df_features.iloc[start:end]
            df_label_cache = self.df_label.iloc[start:end]

            text_series = df_tokens_cache['tokens'].map(lambda x: x.split('\t'))
            print(f"first text_series: {text_series}")
            max_len, is_capped = get_max_len_cap(text_series, self.cap)
            attention_masks = np.ones((len(text_series), max_len), dtype=np.int8)
            if is_capped:
                debug_first_branch = False
                debug_second_branch = False

                # remove the additional tokens if exceeds max_len,
                # else: pad
                for i in range(len(text_series)):
                    debug_first_branch = False
                    debug_second_branch = False

                    i_shifted = i + index
                    if len(text_series[i_shifted]) > max_len:
                        debug_first_branch = True
                        # remove the additional tokens
                        while len(text_series[i_shifted]) >= (max_len):
                            text_series[i_shifted].pop()
                        # append the SEP token
                        text_series[i_shifted].append(sep_tok_id)

                    elif len(text_series[i_shifted]) < max_len:  # padding
                        debug_second_branch = True
                        initial_len = len(text_series[i_shifted])
                        miss = max_len - initial_len
                        text_series[i_shifted] += [0] * miss
                        for j in range(initial_len, max_len):
                            attention_masks[i][j] = 0
                    print(
                        f"iteration {i}, debug_first_branch {debug_first_branch} ,debug_second_branch {debug_second_branch}, len: {len(text_series[i_shifted])}")
                    # text_series[i_shifted] = np.array(text_series[i_shifted], dtype=np.int32)
                    # print(f"type of the array: {text_series[i_shifted].dtype}")

            else:  # if the series is not capped, normal padding

                # padding
                for i in range(len(text_series)):
                    i_shifted = i + index  # * self.df_tokens_reader.chunksize
                    initial_len = len(text_series[i_shifted])
                    miss = max_len - initial_len
                    text_series[i_shifted] += [0] * miss
                    for j in range(initial_len, max_len):
                        attention_masks[i][j] = 0
                    # text_series[i_shifted] = np.array(text_series[i_shifted], dtype=np.int32)
                    print(f"iteration {i}, is capped {is_capped}, len: {len(text_series[i_shifted])}")
            # todo we need to optimize this
            list_arr = []
            for feat in df_features_cache.columns:
                list_arr.append(df_features_cache[feat].values)
            feature_mat = np.array(list_arr)
            del list_arr
            gc.collect()

            text_series = text_series.map(lambda l: [int(elem) for elem in l]).map(
                lambda x: np.array(x, dtype=np.int32))

            # print(f"text_series : {text_series}")
            # print(f"text_series type: {type(text_series)}")
            # print(f"text_series to numpy: {text_series.to_numpy()}")

            text_np_mat = np.stack(text_series)
            print(f"text_np_mat :\n {text_np_mat}")
            print(f"text_np_mat shape :\n {text_np_mat.shape}")
            print(f"text_np_mat type : {type(text_np_mat)}")
            print(f"text_np_mat dtype : {text_np_mat.dtype}")
            print(f"text_np_mat 0 type : {type(text_np_mat[0])}")

            print(f"text_np_mat 0 : {text_np_mat[0]}")
            print(f"text_np_mat 0 dtype : {text_np_mat[0].dtype}")
            text_tensor = torch.tensor(text_np_mat, dtype=torch.int64)
            attention_masks = torch.tensor(attention_masks, dtype=torch.int8)
            labels = torch.tensor(df_label_cache['tweet_feature_engagement_is_like']
                                  .map(lambda x: 1 if x else 0).values, dtype=torch.int8)
            features = torch.tensor(feature_mat.T)
            self.tensors = [text_tensor, attention_masks, features, labels]

        return tuple(tensor[index - self.count * self.df_tokens_reader.chunksize] for tensor in self.tensors)


class NNRecNewLoader(RecommenderBase):

    # TODO ES
    def __init__(self, hidden_dropout_prob=0.1, weight_decay=0.0, lr=2e-5, eps=1e-8, num_warmup_steps=0, epochs=4, ):
        super().__init__()
        self.device = None
        self.device = self._find_device()
        self.hidden_dropout_prob = hidden_dropout_prob
        self.weight_decay = weight_decay
        self.lr = lr
        self.eps = eps
        self.num_warmup_steps = num_warmup_steps
        self.epochs = epochs

        self.model = None

    def _find_device(self):
        # If there's a GPU available...
        if torch.cuda.is_available():

            # Tell PyTorch to use the GPU.
            device = torch.device("cuda")

            print('There are %d GPU(s) available.' % torch.cuda.device_count())

            print('We will use the GPU:', torch.cuda.get_device_name(0))

        # If not...
        else:
            print('No GPU available, using the CPU instead.')
            device = torch.device("cpu")

        return device

    # TODO add support for cat features
    def fit(self,
            df_train_features: pd.DataFrame,
            df_train_tokens_reader: pd.io.parsers.TextFileReader,
            df_train_label: pd.DataFrame,
            df_val_features: pd.DataFrame,
            df_val_tokens_reader: pd.io.parsers.TextFileReader,
            df_val_label: pd.DataFrame,
            cat_feature_set: set):
        # Set the seed value all over the place to make this reproducible.
        seed_val = 42

        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        gpu = torch.cuda.is_available()
        if gpu:
            torch.cuda.manual_seed_all(seed_val)

        self.model = BertClassifierDoubleInput(input_size_2=df_train_features.shape[1], hidden_size_2=4)
        self.model.cuda()

        # freeze all bert layers
        # for param in self.model.bert.parameters():
        #     param.requires_grad = False

        # Combine the training inputs into a TensorDataset.
        train_dataset = CustomDatasetCap(df_features=df_train_features, df_tokens_reader=df_train_tokens_reader,
                                         df_label=df_train_label)
        val_dataset = CustomDatasetCap(df_features=df_train_features, df_tokens_reader=df_train_tokens_reader,
                                       df_label=df_train_label)

        train_dataloader, validation_dataloader = create_data_loaders(train_dataset, val_dataset,
                                                                      batch_size=df_train_tokens_reader.chunksize)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]

        # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
        # I believe the 'W' stands for 'Weight Decay fix"
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.lr,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                          eps=self.eps  # args.adam_epsilon  - default is 1e-8.
                          )

        # Total number of training steps is [number of batches] x [number of epochs].
        # (Note that this is not the same as the number of training samples).
        total_steps = len(train_dataloader) * self.epochs

        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,  # Default value in run_glue.py
                                                    num_training_steps=total_steps)

        # We'll store a number of quantities such as training and validation loss,
        # validation accuracy, and timings.
        training_stats = []

        # Measure the total training time for the whole run.
        total_t0 = time.time()

        # For each epoch...
        for epoch_i in range(0, self.epochs):
            # ========================================
            #               Training
            # ========================================

            # Perform one full pass over the training set.

            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, self.epochs))
            print('Training...')
            avg_train_loss, training_time = self.train(self.model, train_dataloader, optimizer, scheduler)

            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on
            # our validation set.

            print("")
            print("Running Validation...")

            avg_val_accuracy, avg_val_loss, validation_time = self.validation(model=self.model,
                                                                              validation_dataloader=validation_dataloader)

            # Record all statistics from this epoch.
            training_stats.append(
                {
                    'epoch': epoch_i + 1,
                    'Training Loss': avg_train_loss,
                    'Valid. Loss': avg_val_loss,
                    'Valid. Accur.': avg_val_accuracy,
                    'Training Time': training_time,
                    'Validation Time': validation_time
                }
            )
        torch.save(self.model.state_dict(), f"./saved_model_epoch{epoch_i + 1}")

        print("")
        print("Training complete!")

        print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

        return training_stats

    def evaluate(self):
        pass

    def get_prediction(self):
        pass

    def load_model(self):
        pass

    def train(self, model, train_dataloader, optimizer, scheduler):

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        # Put the model into training mode. Don't be mislead--the call to
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()

        # For each batch of training data...
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: features
            #   [3]: labels
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_features = batch[2].to(self.device)
            b_labels = batch[3].to(self.device)

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because
            # accumulating the gradients is "convenient while training RNNs".
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()

            # Perform a forward pass (evaluate the model on this training batch).
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # It returns different numbers of parameters depending on what arguments
            # arge given and what flags are set. For our useage here, it returns
            # the loss (because we provided labels) and the "logits"--the model
            # outputs prior to activation.
            loss, logits = model(input_ids=b_input_ids,
                                 input_features=b_features,
                                 token_type_ids=None,
                                 attention_mask=b_input_mask,
                                 labels=b_labels)

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.
            total_train_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))

        return avg_train_loss, training_time

    def validation(self, model, validation_dataloader):

        t0 = time.time()
        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Tracking variables
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using
            # the `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: features
            #   [3]: labels
            b_input_ids = batch[0].to(self.device)
            b_input_mask = batch[1].to(self.device)
            b_features = batch[2].to(self.device)
            b_labels = batch[3].to(self.device)

            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                # token_type_ids is the same as the "segment ids", which
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                # The documentation for this `model` function is here:
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                # Get the "logits" output by the model. The "logits" are the output
                # values prior to applying an activation function like the softmax.
                (loss, logits) = model(input_ids=b_input_ids,
                                       input_features=b_features,
                                       token_type_ids=None,
                                       attention_mask=b_input_mask,
                                       labels=b_labels)

            # Accumulate the validation loss.
            total_eval_loss += loss.item()

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            total_eval_accuracy += flat_accuracy(logits, label_ids)

        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(validation_dataloader)

        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        return avg_val_accuracy, avg_val_loss, validation_time


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    preds = preds.squeeze()
    my_round = lambda x: 1 if x >= 0.5 else 0
    pred_flat = np.fromiter(map(my_round, preds), dtype=np.int).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def get_max_len(sentences):
    max_len = 0
    # For every sentence...
    for sent in sentences:
        # Update the maximum sentence length.
        max_len = max(max_len, len(sent))

    print('Max sentence length: ', max_len)
    return max_len
    # return 322


def get_max_len_cap(sentences, cap: int = 128) -> (int, bool):
    is_capped = False

    max_len = 0
    # For every sentence...
    for sent in sentences:
        # Update the maximum sentence length.
        max_len = max(max_len, len(sent))
        # check if the value is higher than the cap
        if max_len >= cap:
            is_capped = True
            max_len = cap
            break

    print('Max sentence length: ', max_len)
    print('Is capped: ', is_capped)
    return max_len, is_capped


def create_data_loaders(train_dataset, val_dataset, batch_size=3):
    # The DataLoader needs to know our batch size for training, so we specify it
    # here. For fine-tuning BERT on a specific task, the authors recommend a batch
    # size of 16 or 32.

    # Create the DataLoaders for our training and validation sets.
    # We'll take training samples in random order.
    train_dataloader = DataLoader(
        train_dataset,  # The training samples.
        sampler=SequentialSampler(train_dataset),  # Select batches sequentially
        batch_size=batch_size  # Trains with this batch size.
    )

    # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(
        val_dataset,  # The validation samples.
        sampler=SequentialSampler(val_dataset),  # Pull out batches sequentially.
        batch_size=batch_size  # Evaluate with this batch size.
    )

    return train_dataloader, validation_dataloader


def format_time(elapsed):
    import datetime
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

