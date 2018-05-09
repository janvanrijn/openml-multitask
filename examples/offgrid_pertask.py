import argparse
import multitask
import numpy as np
import os
import pickle
import scipy.stats
import sklearn.metrics


def parse_args():
    parser = argparse.ArgumentParser(description='Runs a set of models on a holdout set across tasks')
    parser.add_argument('--output_directory', type=str, default=os.path.expanduser('~') + '/experiments/multitask/')
    parser.add_argument('--data_loader', type=str, default='WistubaLibSVMDataLoader')
    parser.add_argument('--train_size_current_task', type=int, default=10)
    parser.add_argument('--train_size_other_tasks', type=int, default=100)
    parser.add_argument('--num_tasks', type=int, default=25)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--use_cache', action='store_true', default=True)
    return parser.parse_args()


def run_model_on_task(args, data_loader, model, task_idx, X_tr, y_tr, X_te):
    np.random.seed(args.random_seed)

    model_fullname = model.get_name(args.num_tasks, args.train_size_other_tasks)
    cache_directory = os.path.join(args.output_directory,
                                   data_loader.name,
                                   str(args.train_size_current_task),
                                   model_fullname,
                                   str(task_idx),
                                   str(args.random_seed))
    os.makedirs(cache_directory, exist_ok=True)
    output_file = os.path.join(cache_directory, 'model.pkl')
    if not os.path.isfile(output_file) or not args.use_cache:
        print(multitask.utils.get_time(), 'Started training %s ' % model_fullname)
        model.fit(X_tr, y_tr)
        with open(output_file, 'wb') as fp:
            pickle.dump(model, fp)
        print(multitask.utils.get_time(), 'Finished training %s ' % model_fullname)
    else:
        print(multitask.utils.get_time(), 'Loaded %s from cache' % model_fullname)

    with open(output_file, 'rb') as fp:
        current_model = pickle.load(fp)

    return current_model.predict(X_te)


def run_on_task(args, task_idx):
    np.random.seed(args.random_seed)
    if hasattr(multitask.data_loaders, args.data_loader):
        data_loader = getattr(multitask.data_loaders, args.data_loader)
    else:
        raise ValueError('Data loader does not exist:', args.data_loader)

    res = data_loader.load_data(num_tasks=args.num_tasks, per_task_limit=args.train_size_other_tasks)
    tasks_X_values, tasks_y_values, parameter_names, lower_bounds, upper_bounds = res
    num_tasks, num_obs, num_feats = tasks_X_values.shape

    # make train and test sets
    tr_self_indices = np.random.choice(num_obs, args.train_size_current_task, replace=False)
    te_self_indices = np.array(list(set(range(num_obs)) - set(tr_self_indices)))

    # reshape the array after deleting the test task
    X_tr = np.reshape(np.delete(tasks_X_values, task_idx, 0), ((num_tasks - 1) * num_obs, num_feats))
    y_tr = np.reshape(np.delete(tasks_y_values, task_idx, 0), ((num_tasks - 1) * num_obs, 1))

    # now add the train instances of the test task
    X_tr = np.concatenate((X_tr, tasks_X_values[task_idx, tr_self_indices]))
    y_tr = np.concatenate((y_tr, tasks_y_values[task_idx, tr_self_indices].reshape(-1, 1)))

    assert X_tr.shape == (args.train_size_other_tasks * (args.num_tasks - 1) + args.train_size_current_task, num_feats)
    assert y_tr.shape == (args.train_size_other_tasks * (args.num_tasks - 1) + args.train_size_current_task, 1)

    X_te = tasks_X_values[task_idx, te_self_indices]
    y_te = tasks_y_values[task_idx, te_self_indices]

    models = [
        multitask.models_offgrid.MetaCoregionalizedGPOffgrid(),
        multitask.models_offgrid.MetaCoregionalizedRFOffgrid(),
        multitask.models_offgrid.MetaRandomForestOffgrid(),
        multitask.models_offgrid.MetaSingleOutputGPOffgrid(),
        multitask.models_offgrid.MetaMultitaskGPGeorgeOffgrid(lower_bounds, upper_bounds),
    ]

    for model in models:
        model_fullname = model.get_name(args.num_tasks, args.train_size_other_tasks)
        cache_directory = os.path.join(args.output_directory,
                                       data_loader.name,
                                       str(args.train_size_current_task),
                                       model_fullname,
                                       str(task_idx),
                                       str(args.random_seed))
        results_file = os.path.join(cache_directory, 'measures.pkl')

        if os.path.isfile(results_file):
            with open(results_file, 'rb') as fp:
                print(multitask.utils.get_time(), 'Loaded results from cache')
                results = pickle.load(fp)
        else:
            print(multitask.utils.get_time(), 'Generating results')
            real_scores = y_te.flatten()
            predictions = run_model_on_task(args, data_loader, model, task_idx, X_tr, y_tr, X_te)
            spearman = scipy.stats.spearmanr(real_scores, predictions)[0]
            mse = sklearn.metrics.mean_squared_error(real_scores, predictions)
            results = {'spearman': spearman, 'mse': mse}
            with open(results_file, 'wb') as fp:
                pickle.dump(results, fp)

        print(multitask.utils.get_time(), results)


def run(args):
    for task_idx in range(args.num_tasks):
        run_on_task(parse_args(), task_idx)


if __name__ == '__main__':
    run(parse_args())
