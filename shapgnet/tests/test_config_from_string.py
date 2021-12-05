import io

from shapgnet.model_config import ModelSpecs

spec = """train: True                 # train or not,  default is True for generation we only need load pre-trained model
active: 'grid_small'        # dataset set generated.
use_model: 'GraphGruRnn'    # model to use , it must be defined in models section.
draw_prediction: True       # at the of training draw.  (TODO here now it will draw last epocs)
load_model: True            # load model or not, and what
load_epoch: 500             # load model.  last epoch
save_model: True            # save model,
regenerate: True            # regenerated,  factor when indicated by epochs_save
active_setting: mini        # indicate what setting to use, so we can switch from debug to production
evaluate: True

early_stopping:
  monitor: loss
  min_delta:
  patience: 100
  mode: max

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
    start_test: 10
    epochs_test: 10
    epochs_save: 50
  # baseline
  baseline:
    early_stopping: True
    epochs_log:  1000
    start_test:  100
    epochs_test: 100
    epochs_save: 100

debug:
  # debug graph generation
  graph_generator: True
  # benchmark dataset loader and sampler, if it true it will return after benchmark
  benchmark_read: False
  # debug model creation
  model_creation: False
  # debug training loops
  train_verbose:  False
  # trace early stopping
  trace_early:    False

training:
  train_ratio: 0.8
  test_ration: 0.8
  validation_ratio: 0.2     # validation ration
  num_workers: 1            # num workers to load data, default 4
  batch_ratio: 32           # num  batches of samples per each epoch, 1 epoch = n batches
  sample_time: 1            # default num sample, note each dataset can overwrite

optimizers:
  node_optimizer:
    eps: 1e-8
    weight_decay: 0
    amsgrad: False
    momentum=0:
    betas: [0.9, 0.999]
    type: Adam
  edge_optimizer:
    eps: 1e-8
    weight_decay: 0
    amsgrad: False
    momentum=0:
    betas: [ 0.9, 0.999 ]
    type: Adam

# lr_schedulers definition
lr_schedulers:
    - type: multistep
      milestones: [ 400, 1000 ]
      name: main_lr_scheduler
    - type: secondary
      milestones: [ 400, 1000 ]
      name: secondary

# Model definition
models:
  # this pure model specific, single model can describe both edges and nodes
  # in case we need use single model for edge and node prediction task ,
  # use keyword single_model: model_name
  GraphGruRnn:
    node_model:
      model: GraphGRU
      optimizer: node_optimizer
      lr_scheduler: main_lr_scheduler
      has_input: True
      has_output: True
    edge_model:
      model: GraphGRU
      optimizer: edge_optimizer
      lr_scheduler: main_lr_scheduler
      input_size: 1
  GraphLstmRnn:
    node_model:
      model: GraphLSTM
      optimizer: node_optimizer
      lr_scheduler: main_lr_scheduler
      has_input: True
      has_output: True
    edge_model:
      model: GraphLSTM
      optimizer: edge_optimizer
      lr_scheduler: main_lr_scheduler
      input_size: 1

plots:
  limit: 100

metrics:
  degree: True
  orbits: True
  clustering: True

trace_prediction_timer: True
trace_training_timer: True
trace_epocs: 1

graph:
   # multiplied (640x10 and 15x32)

  # Generated Grid
  grid:
    # https://networkx.org/documentation/stable/reference/generated/networkx.generators.lattice.grid_2d_graph.html
    epochs: 100
    parameter_shrink: 1
    batch_size: 32
    test_batch_size: 32
    test_total_size: 1000
    num_layers: 4
    lr: 0.003
    milestones: [ 400, 1000 ]
    lr_rate: 0.3
    graph_spec:
      grid_n: [ 10, 20 ]
      grid_m: [ 10, 20 ]
    max_num_node: 0
    max_prev_node: 40
  # just to test code logic
  grid_min:
    epochs: 100
    parameter_shrink: 2
    batch_size: 32
    test_batch_size: 32
    test_total_size: 1000
    num_layers: 4
    lr: 0.003
    milestones: [ 400, 1000 ]
    lr_rate: 0.3
    graph_spec:
      grid_n: [ 2, 5 ]
      grid_m: [ 2, 6 ]
    #    max_num_node: 10
    max_prev_node: 15
  grid_small:
    epochs: 500
    parameter_shrink: 2
    batch_size: 32
    test_batch_size: 32
    test_total_size: 1000
    num_layers: 4
    lr: 0.003
    milestones: [ 400, 1000 ]
    lr_rate: 0.3
    graph_spec:
      grid_n: [ 2, 5 ]
      grid_m: [ 2, 6 ]
#    max_num_node: 10
    max_prev_node: 15
  # Generated Community
  caveman:
    epochs: 20
    parameter_shrink: 1
    batch_size: 32
    test_batch_size: 32
    test_total_size: 1000
    num_layers: 4
    lr: 0.003
    milestones: [ 400, 1000 ]
    lr_rate: 0.3
    graph_spec:
      size_of_cliques: 10
      num_of_cliques_i: [ 2, 3 ]
      num_of_cliques_j: [ 30, 81 ]
      p_edge: 0.8
    # max number num nodes
    max_num_node: 100
    # max nodes
    max_prev_node: 100
  # small caveman community network.
  # check networkx doc for details
  caveman_small:
    epochs: 20
    num_layers: 4
    parameter_shrink: 2
    test_batch_size: 32
    test_total_size: 1000
    batch_size: 32
    milestones: [ 400, 1000 ]
    lr: 0.003
    lr_rate: 0.3
    # graph specs
    graph_spec:
      size_of_cliques: 20
      num_of_cliques_i: [2, 3]
      num_of_cliques_j: [6, 11]
      p_edge: 0.3
      # max number num nodes
    max_num_node: 20
    # max nodes
    max_prev_node: 20

root_dir: "."
log_dir: "logs"
nil_dir: "timing"
graph_dir: "graphs"
results_dir: "results"
timing_dir: "timing"
figures_dir: "figures"
prediction_dir: "prediction"                    # where we save prediction
model_save_dir: "model_save"                    # where we save model
#figures_prediction_dir: "prediction_figures"    #
"""

import os
trainer_spec = ModelSpecs(template_file_name=io.StringIO(spec), verbose=True)
print(trainer_spec.config)