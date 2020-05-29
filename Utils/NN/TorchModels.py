import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss
from transformers import BertModel, DistilBertModel

from Utils.Eval.Metrics import ComputeMetrics as CoMe


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

    def __init__(self, input_size_2, hidden_size_2, hidden_size3, hidden_dropout_prob=0.1):
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

        pooled_output = nn.ReLU()(pooled_output)

        logits = self.classifier(pooled_output)

        outputs = (logits,) + bert_outputs[2:]  # add hidden states and attention if they are here
        preds = torch.sigmoid(logits)

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.view(-1).float())
            # Declaring the class containing the metrics
            cm = CoMe(preds.detach().cpu().numpy(), labels.detach().cpu().numpy())
            # Evaluating
            prauc = cm.compute_prauc()
            rce = cm.compute_rce()
            # Confusion matrix
            conf = cm.confMatrix()
            # Prediction stats
            max_pred, min_pred, avg = cm.computeStatistics()

            outputs = (loss,) + outputs + (preds, prauc, rce, conf, max_pred, min_pred, avg)


        return outputs  # (loss), logits, (hidden_states), (attentions), prauc, rce, conf, max_pred, min_pred, avg


class DistilBertClassifierDoubleInput(nn.Module):

    def __init__(self, input_size_2, hidden_size_2, hidden_size_3, hidden_dropout_prob=0.1):
        super().__init__()

        self.bert = DistilBertModel.from_pretrained("distilbert-base-multilingual-cased")

        self.dropout = nn.Dropout(hidden_dropout_prob)

        hidden_size_bert = 768
        self.first_layer = nn.Linear(hidden_size_bert + input_size_2, hidden_size_2)
        self.second_layer = nn.Linear(hidden_size_2, hidden_size_3)

        self.classifier = nn.Linear(hidden_size_3, 1)

    def forward(
            self,
            input_ids=None,
            input_features=None,  # the second input
            attention_mask=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None, ):

        distilbert_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = torch.cat([pooled_output, input_features.float()], dim=1)

        pooled_output = self.first_layer(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        pooled_output = self.second_layer(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, dim)

        preds = torch.sigmoid(logits)

        outputs = (logits,) + distilbert_output[1:]

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.view(-1).float())
            # Declaring the class containing the metrics
            cm = CoMe(preds.detach().cpu().numpy(), labels.detach().cpu().numpy())
            # Evaluating
            prauc = cm.compute_prauc()
            rce = cm.compute_rce()
            # Confusion matrix
            conf = cm.confMatrix()
            # Prediction stats
            max_pred, min_pred, avg = cm.computeStatistics()

            outputs = (loss,) + outputs + (preds, prauc, rce, conf, max_pred, min_pred, avg)

        return outputs  # (loss), logits, (hidden_states), (attentions), (preds, prauc, rce, conf, max_pred, min_pred, avg)
