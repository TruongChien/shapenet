# ShapeGen A Graph Generative models

This repository is the PyTorch implementation of graph generative models. 
- GraphRNN a graph generative model using auto-regressive.
- NetGan and Customer implementation of NetGan.
- Custom GraphRNN ongoing work on improvement.

[Jiaxuan You](https://cs.stanford.edu/~jiaxuan/)\*, [Rex Ying](https://cs.stanford.edu/people/rexy/)\*
, [Xiang Ren](http://www-bcf.usc.edu/~xiangren/), [William L. Hamilton](https://stanford.edu/~wleif/)
, [Jure Leskovec](https://cs.stanford.edu/people/jure/index.html)
, [GraphRNN: Generating Realistic Graphs with Deep Auto-regressive Model](https://arxiv.org/abs/1802.08773) (ICML 2018)

## Installation

The code has been tested over PyTorch latest version 1.10
 - Install PyTorch following the instructions on the [official website](https://pytorch.org/).
 - Check requirement files.
 - 
```bash
conda env create -f environment.yml
conda activate shapegen
conda install pytorch torchvision -c pytorch
```

We can run code in colab, jupiter or standalone app.
```bash
 For Colab you need follow colab notebook.
```
Then install the other dependencies.

```bash
pip install -r requirements.txt
```

First create config.yaml file

By default, trainer will create directory indicated in config.yaml file, each model under result
Both configured in config.yaml Note not all variable syntactically checked.  Validation for config.yaml still in TODO not all done so please use existing example.

Each experiment create set of directories.  Inside each directory we have generated graph 
that used for a train a model.  All model files serialized to pickle files.

## Test run

- Generative notebook mainly to run on colab


```bash
python main.py
```

## Code description

For the GraphRNN model:
`main.py` is the main executable file, and specific arguments are set in `args.py`.
`rnn_generator.py` includes training iterations and calls `model.py` and `data.py`
