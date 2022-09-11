# LSTM Text generation 
This is a project to enroll in Tinkoff ML courses. Here is a realisation of LSTM neural network to generate texts.

## Dependencies
```shell
pip install torch --user
```

## Running
### Train
```shell
python train.py --model model.pth --input-dir data/
```
You can also input texts from stdin if you don't specify `input-dir`.
Input will be read until EOF symbol. (`CTRL + D` in Linux and `CTRL + Z and Enter` in Windows)
```shell
python train.py --model model.pth
```
PyTorch model will be saved in `.pth` file along with extra data in `.json` file with the same name.

There is example train data: shortened Dune book in `data/`
### Generation
```shell
python generate.py --model model.pth --length 100 --prefix "word1 word2"
```

There is a pretrained Russian-language model named `model.pth`. It was trained on over 200 000 words from [Kaggle's dataset](https://www.kaggle.com/datasets/d0rj3228/russian-literature) of Russian literature

## Examples
```text
wilderness that his mother returned her to certain houses will feel the eyes paul called usul paul sat up hardeyed into a killing paul grave this he asked im infected by the only safety requires the spacing guild and the minutiae of life of imperial conditioning—supposedly safe enough to menchildren curiosity reduced pauls groin wed best wont waste and yueh a worried frown for such carelessness halleck kicked the hundred pages
```
```text
привет тебе со мною ползла вереница нагруженных скрипящих телег проползала змеей меж людьми подходя к тебе любил 13 марта 1900 валкирия на окружающую среду таким конец всеведущей гордыне прошедший сумрак очи кто любит забыли вы зеленые длинные волосы голос пением нежным мне будет петь потом вылезали из помню камней в полях тогда
```
```text
странны безмолвные страсти и под ярким солнцем год не постный под ярким солнцем взволнован шопотом криком врагов ты почти несвязный бред обнищалой души невинной и стал всему предел и те сколько музыки прошла как недуг и в тумане глаза хозе метнула взгляд и груб человек умирает не откроет могила холодна безмолвствует
```