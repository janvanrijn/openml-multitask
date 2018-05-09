import argparse
import multitask
import numpy as np
import os
import pickle
import scipy.stats
import sklearn.metrics


def parse_args():
    parser = argparse.ArgumentParser(description='Runs a set of models on a holdout set across tasks')
    parser.add_argument('--output_directory', type=str, default=os.path.expanduser('~') + '/experiments/multitask/multi/')
    parser.add_argument('--data_loader', type=str, default='OpenMLLibSVMDataLoader')
    parser.add_argument('--train_size', type=int, default=30)
    parser.add_argument('--num_tasks', type=int, default=5)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--extension', type=str, default='png')
    parser.add_argument('--use_cache', action='store_true', default=False)
    return parser.parse_args()


def benchmark(args, data_loader):
    np.random.seed(args.random_seed)
    #np.seterr(all='raise')
    res = data_loader.load_data(num_tasks=args.num_tasks)
    tasks_X_values, tasks_y_values, parameter_names, lower_bounds, upper_bounds = res
    num_tasks, num_obs, num_feats = tasks_X_values.shape

    # make train and test sets
    tr_indices = np.random.choice(num_obs, args.train_size, replace=False)
    te_indices = np.array(list(set(range(num_obs)) - set(tr_indices)))

    tasks_X_te = tasks_X_values[:, te_indices, :]
    tasks_X_tr = tasks_X_values[:, tr_indices, :]
    tasks_X_tr = np.reshape(tasks_X_tr, (num_tasks * args.train_size, num_feats))

    tasks_y_te = tasks_y_values[:, te_indices]
    tasks_y_tr = tasks_y_values[:, tr_indices]
    tasks_y_tr = np.reshape(tasks_y_tr, (num_tasks * len(tr_indices), 1))

    print('Train size: %d; test size: %d' % (len(tr_indices), len(te_indices)))
    for idx, column in enumerate(parameter_names):
        print('%s [%f -- %f]' % (column, lower_bounds[idx], upper_bounds[idx]) )

    models = [
        multitask.models_offgrid.MetaMultitaskGPGeorgeOffgrid(lower_bounds, upper_bounds),
        multitask.models_offgrid.MetaCoregionalizedGPOffgrid(),
        multitask.models_offgrid.MetaCoregionalizedRFOffgrid(),
        multitask.models_offgrid.MetaRandomForestOffgrid(),
        multitask.models_offgrid.MetaSingleOutputGPOffgrid()
    ]

    results = dict()
    for model in models:
        # s = random_seed, n = num tasks, tr = train instances per task
        filename = '%s.%s.s%d.n%d.tr%d.pkl' % (data_loader.name, model.name, args.random_seed, num_tasks, args.train_size)
        output_file = os.path.join(args.output_directory, filename)
        if os.path.isfile(output_file) and args.use_cache:
            print(multitask.utils.get_time(), 'Loaded %s from cache' %filename)
            with open(output_file, 'rb') as fp:
                results[model.name] = pickle.load(fp)
            continue
        print(multitask.utils.get_time(), 'Generating %s ' %filename)
        results[model.name] = dict()
        model.fit(tasks_X_tr, tasks_y_tr)

        for idx in range(num_tasks):
            real_scores = tasks_y_te[idx].flatten()

            mean_prediction = model.predict(tasks_X_te[idx])
            if np.unique(mean_prediction).size == 1:
                raise ValueError('Model %s had a constant prediction on task %d' % (model.name, idx))

            spearman = scipy.stats.pearsonr(mean_prediction, real_scores)[0]
            mse = sklearn.metrics.mean_squared_error(real_scores, mean_prediction)
            results[model.name][idx] = {'spearman': spearman, 'mse': mse}

        os.makedirs(args.output_directory, exist_ok=True)
        with open(output_file, 'wb') as fp:
            pickle.dump(results[model.name], fp)
    return results


def run(args):
    if hasattr(multitask.data_loaders, args.data_loader):
        data_loader = getattr(multitask.data_loaders, args.data_loader)
    else:
        raise ValueError('Data loader does not exist:', args.data_loader)

    results = benchmark(args, data_loader)
    for measure in ['spearman', 'mse']:
        # s = random_seed, n = num tasks, tr = train instances per task
        output_file_path = os.path.join(args.output_directory, '%s.s%d.n%d.tr%d.%s.%s' % (data_loader.name,
                                                                                          args.random_seed,
                                                                                          args.num_tasks,
                                                                                          args.train_size,
                                                                                          measure,
                                                                                          args.extension))
        multitask.plot.plot_boxplots(results, measure, measure + ' off grid', output_file_path)


if __name__ == '__main__':
    run(parse_args())
