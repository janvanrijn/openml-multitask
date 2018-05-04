import argparse
import multitask
import numpy as np
import os
import pickle
import scipy.stats
import sklearn.metrics


def parse_args():
    parser = argparse.ArgumentParser(description='Proof of concept of Multi-task GP')
    parser.add_argument('--output_directory', type=str, default='/home/janvanrijn/experiments/multitask/multi/')
    parser.add_argument('--data_loader', type=str, default='OpenMLLibSVMDataLoader')
    parser.add_argument('--train_size', type=int, default=100)
    parser.add_argument('--num_tasks', type=int, default=None)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--extension', type=str, default='png')
    parser.add_argument('--use_cache', action='store_true', default=True)
    return parser.parse_args()


def benchmark(args, data_loader):
    np.random.seed(args.random_seed)
    #np.seterr(all='raise')
    tasks_X_values, tasks_y_values = data_loader.load_data(num_tasks=args.num_tasks)
    num_tasks, num_obs, num_feats = tasks_X_values.shape

    # make train and test sets
    train_indices = np.random.choice(num_obs, args.train_size, replace=False)
    test_indices= np.array(list(set(range(num_obs)) - set(train_indices)))

    task_X_train = tasks_X_values[:, train_indices, :]
    task_X_test = tasks_X_values[:, test_indices, :]
    task_y_train = tasks_y_values[:, train_indices]
    task_y_test = tasks_y_values[:, test_indices]

    print('Train size: %d; test size: %d' % (len(train_indices), len(test_indices)))

    models = [
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
        model.fit(task_X_train, task_y_train)

        for idx in range(num_tasks):
            real_scores = task_y_test[idx].flatten()
            mean_prediction = model.predict(task_X_test, idx)
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
