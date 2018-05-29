import argparse
import collections
import multitask
import os
import pickle


# sshfs jv2657@habanero.rcs.columbia.edu:/rigel/home/jv2657/experiments ~/habanero_experiments/
def parse_args():
    parser = argparse.ArgumentParser(description='Runs a set of models on a holdout set across tasks')
    parser.add_argument('--results_directory', type=str, default=os.path.expanduser('~') +
                                                                 '/habanero_experiments/multitask')
    parser.add_argument('--output_directory', type=str, default=os.path.expanduser('~') + '/experiments/multitask/')
    parser.add_argument('--data_loader', type=str, default='OpenMLLibSVMDataLoader')
    parser.add_argument('--train_size_current_task', type=int, default=16)
    parser.add_argument('--train_size_other_tasks', type=int, default=80)
    parser.add_argument('--num_tasks', type=int, default=100)
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
        multitask.models_offgrid.MetaStackingGPOffgrid(),
        multitask.models_offgrid.MetaCoregionalizedGPOffgrid(),
        multitask.models_offgrid.MetaCoregionalizedRFOffgrid(),
        multitask.models_offgrid.MetaRandomForestOffgrid(),
        multitask.models_offgrid.MetaSingleOutputGPOffgrid(),
        # multitask.models_offgrid.MetaMultitaskGPGeorgeOffgrid(None, None),
    ]

    results_directory = os.path.join(args.results_directory,
                                     data_loader.name,
                                     str(args.train_size_current_task))

    model_task_results = collections.defaultdict(dict)
    for model in models:
        model_name = model.get_name(args.num_tasks, args.train_size_other_tasks)
        for task in os.listdir(os.path.join(results_directory, model_name)):
            for seed in os.listdir(os.path.join(results_directory, model_name, task)):
                results_file = os.path.join(results_directory, model_name, task, seed, 'measures.pkl')
                if not os.path.isfile(results_file):
                    continue

                with open(results_file, 'rb') as fp:
                    results = pickle.load(fp)
                    model_task_results[model_name][task] = results

    for model in model_task_results.keys():
        print('%s num results: %d' % (model, len(model_task_results[model])))

    measures = ['spearman', 'mse', 'train_time']
    output_file = os.path.join(args.output_directory,
                               'offgrid_pertask.%s.%d.%d.%s' % (data_loader.name,
                                                                args.train_size_current_task,
                                                                args.train_size_other_tasks,
                                                                args.extension))
    multitask.plot.plot_boxplots(model_task_results, measures, output_file)
