import argparse
import random
from pathlib import Path
import json
import numpy as np

import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, n_vocab):
        super(Model, self).__init__()
        
        self.lstm_size = 128
        self.embedding_dim = 128
        self.num_layers = 4

        self.embedding = nn.Embedding(
            num_embeddings=n_vocab,
            embedding_dim=self.embedding_dim,
        )
        self.lstm = nn.LSTM(
            input_size=self.lstm_size,
            hidden_size=self.lstm_size,
            num_layers=self.num_layers,
            dropout=0.2,
        )
        
        # Fully-connected layer
        self.bn = nn.BatchNorm1d(self.lstm_size)
        self.fc = nn.Linear(self.lstm_size, n_vocab)

    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        
        bn = self.bn(output.permute(0, 2, 1))
        logits = self.fc(bn.permute(0, 2, 1))
        return logits, state

    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.lstm_size),
                torch.zeros(self.num_layers, sequence_length, self.lstm_size))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True, help="Path to file to load trained model from", default="")
    ap.add_argument("-l", "--length", required=True, help="Length of generated sequence", default="")
    ap.add_argument("-p", "--prefix", required=True, help="First words of the generated sequence", default="")
    args = ap.parse_args()
    return args


def generate(model_path, text, length):
    json_path = model_path.with_suffix('.json')
    with open(json_path, 'r') as file:
        data = json.loads(file.read())
    index_to_word = data["index_to_word"]
    word_to_index = {}
    for key in index_to_word:
        word_to_index[index_to_word[key]] = int(key)
    
    # Load model
    device = torch.device("cpu")
    model = Model(len(word_to_index))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.to(device)
    model.eval()

    if len(text) == 0:
        words = [random.choice(ds.unique_words)]
    else:
        words = text.split(' ')

    state_h, state_c = model.init_state(len(words))

    for i in range(0, length):
        x = torch.tensor([[word_to_index[w] for w in words[i:]]])
        y_pred, (state_h, state_c) = model(x, (state_h, state_c))

        last_word_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
        word_index = np.random.choice(len(last_word_logits), p=p)
        words.append(index_to_word[str(word_index)])

    return words


if __name__ == '__main__':
    args = parse_args()

    # Load model
    # model = torch.load(args.model, map_location=torch.device('cpu'))
    # print(model)
    # model.eval()

    words = generate(Path(args.model), args.prefix.lower(), int(args.length))
    print(' '.join(words))
    
