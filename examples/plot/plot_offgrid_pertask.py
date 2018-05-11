import argparse
import collections
import multitask
import os
import pickle

# sshfs jv2657@habanero.columbia.edu:/rigel/home/jv2657/experiments ~/habanero_experiments/
def parse_args():
    parser = argparse.ArgumentParser(description='Runs a set of models on a holdout set across tasks')
    parser.add_argument('--results_directory', type=str, default=os.path.expanduser('~') + '/habanero_experiments/multitask/')
    parser.add_argument('--output_directory', type=str, default=os.path.expanduser('~') + '/experiments/multitask/')
    parser.add_argument('--data_loader', type=str, default='OpenMLLibSVMDataLoader')
    parser.add_argument('--train_size_current_task', type=int, default=10)
    parser.add_argument('--train_size_other_tasks', type=int, default=80)
    parser.add_argument('--num_tasks', type=int, default=50)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--extension', type=str, default='png')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if hasattr(multitask.data_loaders, args.data_loader):
        data_loader = getattr(multitask.data_loaders, args.data_loader)
    else:
        raise ValueError('Data loader does not exist:', args.data_loader)

    models = [
        multitask.models_offgrid.MetaCoregionalizedGPOffgrid(),
        multitask.models_offgrid.MetaCoregionalizedRFOffgrid(),
        multitask.models_offgrid.MetaRandomForestOffgrid(),
        multitask.models_offgrid.MetaSingleOutputGPOffgrid(),
    ]

    results_directory = os.path.join(args.results_directory,
                                     data_loader.name,
                                     str(args.train_size_current_task))

    model_task_results = collections.defaultdict(dict)
    for model in models:
        model_name = model.get_name(args.num_tasks, args.train_size_other_tasks)
        for task in os.listdir(os.path.join(results_directory, model_name)):
            for seed in os.listdir(os.path.join(results_directory, model_name, task)):
                model_file = os.path.join(results_directory, model_name, task, seed, 'model.pkl')
                results_file = os.path.join(results_directory, model_name, task, seed, 'measures.pkl')
                if not os.path.isfile(results_file):
                    raise ValueError('measures.pkl does not exists')
                if not os.path.isfile(model_file):
                    raise ValueError('model.pkl does not exists')

                with open(results_file, 'rb') as fp:
                    results = pickle.load(fp)
                    model_task_results[model_name][task] = results

    for measure in ['spearman', 'mse']:
        output_file = os.path.join(args.output_directory,
                                   'offgrid_pertask.%s.%d.%d.%s.%s' % (data_loader.name,
                                                                       args.train_size_current_task,
                                                                       args.train_size_other_tasks,
                                                                       measure,
                                                                       args.extension))
        multitask.plot.plot_boxplots(model_task_results, measure, measure, output_file)
