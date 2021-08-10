import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import ml_collections

# helpers
def find_path(path_list, name):
    for i, path in enumerate(path_list):
        if path.find(name) != -1:
            return path

def load_data(path):
    data = []
    with open(path, 'r') as file:
        for info in file.readlines():
            data.append(float(info))
    return data

def cfg_builder(file_path_list, eager_path='eager_acc', eager_data_name="Eager Accuracy", graph_path='graph_acc', graph_data_name="Graph Accuracy", xlabel='Epochs', ylabel='Validation Accuracy'):
    eager_data = load_data(find_path(file_path_list, eager_path))
    graph_data = load_data(find_path(file_path_list, graph_path))
    config = ml_collections.ConfigDict()
    config.xlabel = xlabel
    config.ylabel = ylabel
    config.eager = ml_collections.ConfigDict()
    config.eager.data = eager_data
    config.eager.axis  = np.arange(1, len(eager_data) + 1)
    config.eager.data_name = eager_data_name
    config.graph = ml_collections.ConfigDict()
    config.graph.data = graph_data
    config.graph.axis = np.arange(1, len(graph_data) + 1)
    config.graph.data_name = graph_data_name
    return config

def draw_and_save(config, save_path):
    # setup
    plt.rcParams['figure.dpi'] = 100
    plt.clf()
    # Draw Line Chart
    plt.plot(config.eager.axis, config.eager.data, '-', linewidth=1.5, label=config.eager.data_name)
    plt.plot(config.graph.axis, config.graph.data, '-', linewidth=1.5, label=config.graph.data_name)
    plt.xlabel(config.xlabel, fontproperties='Times New Roman')
    plt.ylabel(config.ylabel, fontproperties='Times New Roman')
    plt.legend(loc='upper right', frameon=True, fontsize=8)
    plt.savefig(save_path)

if __name__ == "__main__":
    assert os.path.exists("results/check_info"), 'you should run "check/check.sh" before drawing graphs'

    # path setup
    file_path = glob.glob("results/check_info/*.txt")

    # data config builder
    acc_cfg = cfg_builder(file_path, eager_path='eager_acc', eager_data_name="Eager Accuracy", graph_path='graph_acc', graph_data_name="Graph Accuracy", xlabel='Epochs', ylabel='Validation Accuracy')
    loss_cfg = cfg_builder(file_path, eager_path='eager_losses', eager_data_name="Eager Losses", graph_path='graph_losses', graph_data_name="Graph Losses",xlabel='Steps', ylabel='Training Losses')
    train_epoch_time_cfg = cfg_builder(file_path, eager_path='eager_train_epoch_time', eager_data_name="Eager Train Time (Epoch)", graph_path='graph_train_epoch_time', graph_data_name="Graph Train Time (Epoch)", xlabel='Epochs', ylabel='Training Time')
    eval_epoch_time_cfg = cfg_builder(file_path, eager_path='eager_eval_epoch_time', eager_data_name="Eager Eval Time (Epoch)",graph_path='graph_eval_epoch_time', graph_data_name="Graph Eval Time (Epoch)", xlabel='Epochs', ylabel='Evaluation Time')
    train_step_time_cfg = cfg_builder(file_path, eager_path='eager_train_step_time', eager_data_name="Eager Train Time (Step)", graph_path='graph_train_step_time', graph_data_name="Graph Train Time (Step)", xlabel='Steps', ylabel='Training Time')

    # draw and save
    os.makedirs("results/picture", exist_ok=True)
    draw_and_save(acc_cfg,  'results/picture/acc.png')
    draw_and_save(loss_cfg, 'results/picture/losses.png')
    draw_and_save(train_epoch_time_cfg, 'results/picture/train_epoch_time.png')
    draw_and_save(eval_epoch_time_cfg, 'results/picture/eval_epoch_time.png')
    draw_and_save(train_step_time_cfg, 'results/picture/train_step_time.png')


