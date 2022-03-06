import torch
from torch import nn
from torch.nn import NLLLoss
import torch.nn.functional as F

from src import config


# def loss_fn(output, target, mask, num_labels):
#     lfn = nn.CrossEntropyLoss()
#     ## Only keep active parts of the loss
#     # we don't need to calculate the loss for the whole sentence,
#     # we just need to calculate the loss where we don't have any padding,
#     # which means the mask is 1
#     active_loss = mask.view(-1) == 1 # check which element of mask is 1
#     active_logits = output.view(-1, num_labels)
#     active_labels = torch.where(
#         active_loss,
#         target.view(-1),
#         # if active_loss is false or 0, then replace it with below value:
#         # lfn.ignore_index is equal to -100
#         # so just saying where it's -100, ignore that index while calculating loss
#         torch.tensor(lfn.ignore_index).type_as(target)
#     )
#     loss = lfn(active_logits, active_labels)
#     return loss
#
# vocab_size = len(vocab_to_id)
# embed_size = config.EMBED_SIZE
# lstm_size = config.LSTM_SIZE
# output_size = config.OUTPUT_SIZE
# lstm_layers = config.LSTM_LAYER
# dropout = config.DROP_RATE
# num_labels = uniq_sentiment



class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, lstm_size, output_size,
                 lstm_layers, dropout, num_labels):
        """
        Initialize the model by setting up the layers.

        Args:
            vocab_size : The vocabulary size.
            embed_size : The embedding layer size.
            lstm_size : The LSTM layer size.
            output_size : The output size.
            lstm_layers : The number of LSTM layers.
            dropout : The dropout probability.
        """
        super(SentimentClassifier, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.lstm_size = lstm_size
        self.output_size = output_size
        self.lstm_layers = lstm_layers
        self.dropout = dropout
        self.num_labels = num_labels

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, lstm_size, lstm_layers,
                            dropout=dropout, batch_first=False)
        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_size, output_size)


    def init_hidden(self, batch_size):
        """Initializes hidden state"""
        # Create two new tensors with sizes (lstm_layers, batch_size, hidden_dim),
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        hidden_state = (weight.new(self.lstm_layers, batch_size, self.lstm_size).zero_(),
                        weight.new(self.lstm_layers, batch_size, self.lstm_size).zero_())

        return hidden_state


    def forward(self, hidden_state, input_ids, attention_mask, labels):
        """
        Args:
            input_ids: The batch of input to the NN.
            hidden_state: The LSTM hidden state.
        Returns:
            logps: log softmax output
            hidden_state: The new hidden state.
        """
        # nn_input: (seq_len, batch_size) here

        # embed_input: (*nn_input, embed_size] -> [seq_len, batch_size, embed_size)
        embed_input = self.embedding(input_ids)

        # lstm_out: (seq_len, batch_size, lstm_size)
        # hidden_state: (lstm_layers, batch_size, lstm_size)
        lstm_out, hidden_state = self.lstm(embed_input, hidden_state)

        # only get the last element of each sequence in a batch, since we only need
        # the last element to predict the sentiment.
        # lstm_out: (batch_size, lstm_size)
        lstm_out = lstm_out[-1, :, :]

        # dropout_out: (batch_size, lstm_size)
        dropout_out = self.dropout_layer(lstm_out)

        # logps: (batch_size, output_size)
        logps = F.log_softmax(self.fc(dropout_out), dim=1)

        # loss = loss_fn(logps, labels, attention_mask, self.num_labels)

        if labels is not None:
            loss_fn = NLLLoss()
            loss = loss_fn(logps, labels)
        else:
            loss = None

        return logps, hidden_state, loss

