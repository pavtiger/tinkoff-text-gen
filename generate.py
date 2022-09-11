import argparse
import random
from pathlib import Path
import json
import numpy as np
import torch

from train import Model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True,
                    help="Path to file to load trained model from", default="")
    parser.add_argument("-l", "--length", required=True,
                    help="Length of generated sequence", default="")
    parser.add_argument("-p", "--prefix", required=False,
                    help="First words of the generated sequence", default="")
    return parser.parse_args()


def generate(model_path, text, length):
    json_path = model_path.with_suffix('.json')
    with open(json_path, 'r') as file:
        data = json.loads(file.read())
    index_to_word = data
    word_to_index = {}
    for key in index_to_word:
        word_to_index[index_to_word[key]] = int(key)

    # Load model
    device = torch.device("cpu")
    model = Model(len(word_to_index))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.to(device)
    model.eval()

    input_words, words = text.split(), []
    for word in input_words:
        if word in word_to_index.keys():
            words.append(word)

    if len(words) == 0:
        words = [random.choice(list(word_to_index.keys()))]

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

    generated_words = generate(Path(args.model), args.prefix.lower(), int(args.length))
    print(' '.join(generated_words))
