import os
from ..model_config import ModelSpecs

trainer_spec = ModelSpecs()


def fmt_print(left, *argv):
    if len(argv) == 1:
        print(f"{str(left) + ':' :<25} {argv[0]}")
    else:
        print(f"{str(left) + ':' :<25} {argv}")


print("\n-- list of dir:")
fmt_print("Root dir", trainer_spec.root_dir())
fmt_print("Results", trainer_spec.results_dir())
fmt_print("Log", trainer_spec.log_dir())
fmt_print("Model checkpoint", trainer_spec.model_save_dir())
fmt_print("Graph", trainer_spec.graph_dir())
fmt_print("Figure", trainer_spec.figures_dir())
fmt_print("Timing trace", trainer_spec.timing_dir())
fmt_print("Prediction", trainer_spec.prediction_dir())

fmt_print("\n-- list of dirs:")
fmt_print("dir_main", trainer_spec.dir_input)
fmt_print("dir_result", trainer_spec.dir_result)
fmt_print("dir_log", trainer_spec.dir_log)
fmt_print("dir_graph", trainer_spec.dir_graph_save)
fmt_print("dir_figure", trainer_spec.dir_figure)
fmt_print("dir_timing", trainer_spec.dir_timing)
fmt_print("dir_figure", trainer_spec.figure_prediction)

print("")
print("\n-- model used:")
fmt_print("model", trainer_spec.active_model)
fmt_print("graph", trainer_spec.active)

print("\n-- training parameters:")
print("train ratio:\t\t", trainer_spec.train_ratio())
print("test ratio:\t\t\t", trainer_spec.test_ratio())
print("validation ratio:\t", trainer_spec.validation_ratio())
print("do train: \t", trainer_spec.is_train_network())
print("do sample:\t", trainer_spec.is_draw_samples())
print("batch ratio :\t", trainer_spec.batch_ratio())
print("num worker :\t", trainer_spec.num_workers())
print("num worker :\t", trainer_spec.sample_time())

trainer_spec.build_dir()
assert (os.path.isdir("wrong") is False)
assert (os.path.isdir(trainer_spec.dir_result) is True)
assert (os.path.isdir(trainer_spec.model_save_path) is True)
assert (os.path.isdir(trainer_spec.dir_log) is True)
assert (os.path.isdir(trainer_spec.dir_figure) is True)
assert (os.path.isdir(trainer_spec.dir_timing) is True)
assert (os.path.isdir(trainer_spec.figure_prediction) is True)

print("\n-- global parameters:")
fmt_print('Train ratio', trainer_spec.train_ratio())
fmt_print('Test ratio', trainer_spec.test_ratio())
fmt_print('Validation ratio', trainer_spec.validation_ratio())
fmt_print('Train', trainer_spec.is_train_network())
fmt_print('Sample', trainer_spec.is_draw_samples())
fmt_print('Batch ratio', trainer_spec.batch_ratio())
fmt_print('Number samples', trainer_spec.sample_time())
fmt_print('Number of worker', trainer_spec.num_workers())

print("\n-- optimizer parameters:")
fmt_print("Optimizer name", trainer_spec.optimizer_name())
fmt_print("Optimizer type", trainer_spec.optimizer_type())
fmt_print("Weight decay", trainer_spec.weight_decay(), "default: " + str(trainer_spec.weight_decay(default=True)))
fmt_print("Adam eps", trainer_spec.eps())
fmt_print("Adam amsgrad", trainer_spec.amsgrad())
fmt_print("Adam betas", trainer_spec.betas())
fmt_print("SGD momentum", trainer_spec.momentum())
fmt_print("SGD dampening", trainer_spec.dampening())
fmt_print("SGD nesterov", trainer_spec.nesterov())

print("\n-- training epocs parameters:")
fmt_print('Log at', trainer_spec.epochs_log())
fmt_print('Start test at', trainer_spec.start_test())
fmt_print('Epoch test at', trainer_spec.epochs_test())
fmt_print('Epoch save at', trainer_spec.epochs_save())
fmt_print('Trace prediction', trainer_spec.trace_prediction_timer())
fmt_print('Trace training', trainer_spec.trace_training_timer())
fmt_print('Trace epocs', trainer_spec.trace_epocs())

print("\n-- model files templates")
fmt_print('Model file path', trainer_spec.model_node_file_name())
fmt_print('Model file path', trainer_spec.model_edge_file_name())

fmt_print('Default template', trainer_spec.template_file_name())
fmt_print('Default prediction', trainer_spec.prediction_filename())
fmt_print('Default train', trainer_spec.train_filename())
fmt_print('Default test', trainer_spec.test_filename())

fmt_print('Example for test', trainer_spec.test_filename(epoch=22, sample=33))
fmt_print('Example for train', trainer_spec.train_filename(epoch=44, sample=55))
fmt_print('Example for prediction', trainer_spec.prediction_filename(epoch=66, sample=77))

print("\n-- Current model settings:")
fmt_print("Model settings", trainer_spec.model)
fmt_print("Model settings", trainer_spec.lr_schedulers)
fmt_print("Model lr scheduler", trainer_spec.lr_scheduler('main_lr_scheduler'))
fmt_print("Model lr scheduler", trainer_spec.lr_scheduler_type('main_lr_scheduler'))
fmt_print("Model settings", trainer_spec.model['node_model']['optimizer'])
fmt_print("Model settings", trainer_spec.model['edge_model']['optimizer'])