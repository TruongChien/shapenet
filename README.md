# ShapeGen A Graph Generative models

![alt text](https://miro.medium.com/max/1875/1*B_CulvZLSmhbQ8L_byrJDA.jpeg)

In this project, I explored mainly two ideas, autoregressive and adversarial models. 
So the core of my project is to apply and explore the generative models for Graph Generation. 
In autoregressive settings and adversarial settings in real-world scenarios, where the graph 
generation is either auto-regressive generation or min-max game. 

[Stanford cs224w](https://web.stanford.edu/class/cs224w/)
[Stanford cs236] (https://deepgenerativemodels.github.io/)
[Jure Leskovec](https://cs.stanford.edu/people/jure/index.html)

- GraphRNN a graph generative model using auto-regressive.
- NetGan and Customer implementation of NetGan.
- Custom GraphRNN ongoing work. (Specifically goal to improve generation and training speed) improvement)

[Jiaxuan You](https://cs.stanford.edu/~jiaxuan/)\*, [Rex Ying](https://cs.stanford.edu/people/rexy/)\*
, [Xiang Ren](http://www-bcf.usc.edu/~xiangren/), [William L. Hamilton](https://stanford.edu/~wleif/)
, [Jure Leskovec](https://cs.stanford.edu/people/jure/index.html)
, [GraphRNN: Generating Realistic Graphs with Deep Auto-regressive Model](https://arxiv.org/abs/1802.08773) (ICML 2018)

[NetGan] (https://arxiv.org/abs/1803.00816) [Aleksandar Bojchevski], [Oleksandr Shchur] [Daniel Zügner],[Stephan Günnemann]

This repository is the PyTorch implementation. 

###

[more details here] (https://medium.com/@spyroot/shapegen-gran-generation-955d5b78e6d8)

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

## Training.

In order to train a network we just need pass trainer specification to a trainer.

Following code delegate to factory class , that essentially returns a dict 
that store model name, sub-models and GraphRNN model. It also returns
a RnnTrainer class that instantiated by same Factory class.

```
  models = model_creator.create_model(verbose=True)
  if trainer_spec.is_train_network():
        decoder = AdjacencyDecoder()
        trainer = model_creator.create_trainer(dataset_loader, models, decoder)
        trainer.train()
```

The main logic of RNN trainer is here
https://github.com/spyroot/shapenet/blob/main/shapegnet/rnn_generator.py
or you can click in folder and expand source tree.

Almost all parameters' trainer reads from specification, it includes 

* Optimizer setting.
* Scheduler.
* All hyperparameteres, and it per each experiment.

During training a RnnTrainer uses.  Current active setting, that describes
when to save, test and log.  

When factory method in create_model, creates the model and respected trainer.
It passed configuration to Rnn Trainer.

```
settings:
  # debug mode
  debug:
    epochs_log:  1000
    start_test:  10
    epochs_test: 10
    epochs_save: 10
  # baseline
  mini:
    # if we need enable early stopping
    early_stopping: True
    epochs_log: 1000
    start_test: 2
    epochs_test: 2
    epochs_save: 2
  # baseline
  baseline:
    early_stopping: True
    epochs_log:  1000
    start_test:  100
    epochs_test: 100
    epochs_save: 100
```

The binding to current active configuration done via 
active_setting: baseline    # indicate what setting to use, so we can switch 

The current data set that we're training
active: 'grid_small'        # dataset set generated.

The current model we use for a experiment.

## Test run

- Generative notebook mainly to run on colab


```bash
python main.py
```

## Code description

For the GraphRNN model:
`main.py` is the main executable file, and specific arguments are set in `args.py`.
`rnn_generator.py` includes training iterations and calls `model.py` and `data.py`
