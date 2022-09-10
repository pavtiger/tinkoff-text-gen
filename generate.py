import argparse
import random
from pathlib import Path
import json
import torch

from train import Dataset, remove_consecutive_spaces, get_uniq_words, load


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True, help="Path to file to load trained model from", default="")
    ap.add_argument("-l", "--length", required=True, help="Length of generated sequence", default="")
    ap.add_argument("-p", "--prefix", required=True, help="First words of the generated sequence", default="")
    args = vars(ap.parse_args())

    return args


def generate(model, text, length):
    model.eval()
    
    # json_path = path.with_suffix('.json')
    # with open(json_path, 'r') as file:
    #     data = json.loads(file)
    # word_to_index, index_to_word = data["word_to_index"], data["index_to_word"]
    
    # Load model
    # model = Model(ds)
    # model.load_state_dict(torch.load("model.pt"))
    # model.eval()

    if len(text) == 0:
        words = [random.choice(ds.unique_words)]
    else:
        words = text.split(' ')

    state_h, state_c = model.init_state(len(words))

    for i in range(0, length):
        x = torch.tensor([[model.word_to_index[w] for w in words[i:]]])
        y_pred, (state_h, state_c) = model(x.cuda(), (state_h.cuda(), state_c.cuda()))

        last_word_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().cpu().numpy()
        word_index = np.random.choice(len(last_word_logits), p=p)
        words.append(model.index_to_word[word_index])

    return words


if __name__ == '__main__':
    args = parse_args()

    # Load model
    model = torch.load(args["model"])
    model.eval()

    words = generate(model, args["prefix"], args["length"])
    print(' '.join(words))
    
