import argparse
import collections
import matplotlib.pyplot
import multitask
import os
import pickle


def parse_args():
    parser = argparse.ArgumentParser(description='Runs a set of models on a holdout set across tasks')
    parser.add_argument('--output_directory', type=str, default=os.path.expanduser('~') + '/experiments/multitask/')
    parser.add_argument('--data_loader', type=str, default='WistubaLibSVMDataLoader')
    parser.add_argument('--train_size_current_task', type=int, default=10)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--extension', type=str, default='png')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if hasattr(multitask.data_loaders, args.data_loader):
        data_loader = getattr(multitask.data_loaders, args.data_loader)
    else:
        raise ValueError('Data loader does not exist:', args.data_loader)

    results_directory = os.path.join(args.output_directory,
                                     data_loader.name,
                                     str(args.train_size_current_task))

    model_task_results = collections.defaultdict(dict)
    for model in os.listdir(results_directory):
        for task in os.listdir(os.path.join(results_directory, model)):
            for seed in os.listdir(os.path.join(results_directory, model, task)):
                results_file = os.path.join(results_directory, model, task, seed, 'measures.pkl')
                with open(results_file, 'rb') as fp:
                    results = pickle.load(fp)
                    model_task_results[model][task] = results

    for measure in ['spearman', 'mse']:
        output_file = os.path.join(args.output_directory, 'offgrid_pertask.%s.%s.%s' % (data_loader.name, measure, args.extension))
        multitask.plot.plot_boxplots(model_task_results, measure, measure, output_file)
