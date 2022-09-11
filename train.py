import argparse
import os
import string
import json
from collections import Counter
from pathlib import Path
from sys import stdin

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.optim.lr_scheduler import StepLR


seq_length = 20
num_epoch = 20
batch_size = 8
seed = 1303
train_percent = 80
print_every = 1000


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dir", required=False,
                        help="Input directory to read texts from")
    parser.add_argument("-m", "--model", required=True,
                        help="Path to file to save trained model to")
    return parser.parse_args()


def remove_consecutive_spaces(s):
    return ' '.join(s.split())


def get_uniq_words(words):
    word_counts = Counter(words)
    return sorted(word_counts, key=word_counts.get, reverse=True)


def process_text(text):
    clean_text = "".join(
        [char for char in text if char not in string.punctuation]).lower()
    clean_text = remove_consecutive_spaces(clean_text)
    return clean_text


def load(load_directory):
    train_texts = []

    if load_directory is None:
        print("Enter text. Finish with EOF character. (Ctrl + D)\n")
        text = ""
        for line in stdin:
            text += line
        train_texts.append(process_text(text))

    else:
        for root, _, files in os.walk(load_directory):
            for text_file in files:
                path = os.path.join(root, text_file)
                extension = os.path.splitext(path)[1]

                if extension == ".txt":
                    try:
                        with open(path, encoding='utf-8', mode="r") as f:
                            text = f.read().replace('\n', ' ')
                            train_texts.append(process_text(text))
                    except:
                        # Needed just in case if encoding of the file is strange
                        # (I've seen this in some Kaggle datasets)
                        print(f"error importing path: {path}")

    assert(len(train_texts) != 0)
    text = ' '.join(train_texts)
    print(f'Loaded {len(text)} symbols')
    return text.split(' ')


class Dataset(torch.utils.data.Dataset):
    def __init__(self, load_directory):
        self.words = load(load_directory)
        self.unique_words = get_uniq_words(self.words)

        self.index_to_word = {index: word for index, word in enumerate(self.unique_words)}
        self.word_to_index = {word: index for index, word in enumerate(self.unique_words)}

        self.words_indexes = [self.word_to_index[w] for w in self.words]

    def __len__(self):
        return len(self.words_indexes) - seq_length

    def __getitem__(self, index):
        return (
            torch.tensor(self.words_indexes[index: index + seq_length]),
            torch.tensor(self.words_indexes[index + 1: index + seq_length + 1]),
        )


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


def train(ds_train, ds_test, model):
    model.train()
    print(f"training started. total epoch {num_epoch}\n")

    train_loader = DataLoader(ds_train, batch_size=batch_size)
    test_loader = DataLoader(ds_test, batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = StepLR(opt, step_size=8, gamma=0.0001)

    for epoch in range(num_epoch):
        print(f"Epoch {epoch + 1} started")
        print('-' * 70)

        state_h, state_c = model.init_state(seq_length)
        train_errors = []

        model.train()
        for batch, (x, y) in enumerate(train_loader):
            y_pred, (state_h, state_c) = model(x, (state_h, state_c))
            err = criterion(y_pred.transpose(1, 2), y)

            train_errors.append(err.item())

            state_h = state_h.detach()
            state_c = state_c.detach()

            err.backward()
            opt.step()
            opt.zero_grad()

            if batch % print_every == 0:
                print(f' | train batch {batch}/{len(train_loader)}. train_loss: {err.item()}')

        print('-' * 70)
        test_errors = []

        model.eval()
        with torch.no_grad():
            for batch, (x, y) in enumerate(test_loader):
                y_pred, (state_h, state_c) = model(x, (state_h, state_c))
                err = criterion(y_pred.transpose(1, 2), y)

                test_errors.append(err.item())

                state_h = state_h.detach()
                state_c = state_c.detach()

                if batch % print_every == 0:
                    print(f' | test batch {batch}/{len(test_loader)}. test_loss: {err.item()}')

        scheduler.step()
        train_loss = sum(train_errors) / len(train_errors)
        test_loss = sum(test_errors) / len(test_errors)

        print('-' * 70)
        print(f'Epoch {epoch + 1} ended. train_loss: {train_loss}, test_loss: {test_loss}')
        print()


if __name__ == '__main__':
    args = parse_args()
    model_path = Path(args.model)

    # Load dataset
    ds = Dataset(args.input_dir)

    # Save word_to_index and index_to_word dictionaries to json file for generator's use
    with open(model_path.with_suffix('.json'), encoding='utf-8', mode='w') as file:
        json.dump(ds.index_to_word, file)

    # Train/test split
    train_len = int(len(ds) / 100 * train_percent)
    dataset_train, dataset_test = random_split(ds, [train_len, len(ds) - train_len],
                                               generator=torch.Generator().manual_seed(seed))

    # Train model
    model = Model(len(ds.unique_words))
    train(dataset_train, dataset_test, model)

    # Save model
    torch.save(model.state_dict(), model_path.with_suffix('.pth'))
