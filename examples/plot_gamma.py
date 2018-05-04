import argparse
import collections
import matplotlib.pyplot as plt
import multitask
import numpy as np
import os
import scipy.stats


def parse_args():
    parser = argparse.ArgumentParser(description='Proof of concept of Multi-task GP')
    parser.add_argument('--output_directory', type=str, default='/home/janvanrijn/experiments/multitask/coregionalized/')
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--extension', type=str, default='png')
    parser.add_argument('--use_cache', action='store_true', default=True)
    return parser.parse_args()


def plot_original_performance(tasks_X_values, tasks_y_values, output_file):
    marks = [',', '+', '.', 'o', '*']
    fig, axes = plt.subplots(1, 1)
    axes.set_title('Parameter Sweep')
    axes.set_ylabel('Predictive Accuracy')
    axes.set_xlabel('Gamma (log-values, normalized to [-1, 1])')
    for reg in range(5):
        axes.scatter(tasks_X_values[reg, :, 0], tasks_y_values[reg], marker=marks[reg], label='task%d' % reg)

    plt.savefig(output_file)
    plt.close()


def remove_range_and_fit(tasks_X_values, tasks_y_values, train_size, output_directory, extension):
    num_tasks, num_obs, num_feats = tasks_X_values.shape
    assert num_feats == 2  # gamma and task idx

    test_size = num_obs - train_size
    tasks_X_tr = np.zeros((num_tasks, train_size, num_feats), dtype=float)
    tasks_y_tr = np.zeros((num_tasks, train_size), dtype=float)
    tasks_X_te = np.zeros((num_tasks, test_size, num_feats), dtype=float)
    tasks_y_te = np.zeros((num_tasks, test_size), dtype=float)

    for idx in range(num_tasks):
        start_idx = np.random.randint(num_obs)
        tr_indices = np.remainder(np.arange(start_idx, start_idx + train_size), num_obs)
        te_indices = np.remainder(np.arange(start_idx - test_size, start_idx), num_obs)
        assert(set(tr_indices).union(set(te_indices)) == set(range(num_obs)))
        tasks_X_tr[idx] = tasks_X_values[idx, tr_indices]
        tasks_y_tr[idx] = tasks_y_values[idx, tr_indices]
        tasks_X_te[idx] = tasks_X_values[idx, te_indices]
        tasks_y_te[idx] = tasks_y_values[idx, te_indices]

    models = [multitask.models_offgrid.MetaCoregionalizedGPOffgrid(),
              multitask.models_offgrid.MetaSingleOutputGPOffgrid()]

    correlations = collections.defaultdict(list)

    for model in models:
        print(multitask.utils.get_time(), 'Start fitting', model.name)
        model.fit(tasks_X_tr, tasks_y_tr)
        print(multitask.utils.get_time(), 'Done fitting', model.name)

    output_directory = os.path.join(output_directory, 'tr%s' % train_size)
    os.makedirs(output_directory, exist_ok=True)

    for region in range(num_tasks):
        output_file = os.path.join(output_directory, 'task-%d.%s' %(region, extension))
        fig = plt.figure(figsize=(16, 6))

        # Set common labels
        fig.text(0.5, 0.005, 'Gamma (log-values, normalized to [-1, 1])', ha='center', va='center')
        fig.text(0.005, 0.5, 'Predictive Accuracy', ha='center', va='center', rotation='vertical')

        ax = [fig.add_subplot(121), fig.add_subplot(122)]

        for model_idx, model in enumerate(models):
            real_scores = tasks_y_te[idx].flatten()
            mean_prediction = model.predict(tasks_X_te, region)
            spearman = scipy.stats.pearsonr(mean_prediction, real_scores)[0]
            correlations[model.name].append(spearman)
            
            model.plot(region, ax[model_idx])

            ax[model_idx].set_title('%s [%0.2f]' %(model.name, spearman))
            ax[model_idx].set_xlim(-1, 1)
            ax[model_idx].set_ylim(0, 1)

            ax[model_idx].plot(tasks_X_tr[region, :, 0], tasks_y_tr[region], 'o')
            ax[model_idx].plot(tasks_X_te[region, :, 0], tasks_y_te[region], 'o')
            ax[model_idx].legend_.remove()

        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()

    for model, scores in correlations.items():
        print(model, np.mean(scores), 'spearman correlation')


def run(args):
    np.random.seed(args.random_seed)
    #np.seterr(all='raise')

    tasks_X_values, tasks_y_values = multitask.data_loaders.WistubaLibSVMDataLoader.load_data_rbf_fixed_complexity()
    os.makedirs(args.output_directory, exist_ok=True)
    output_file = os.path.join(args.output_directory, 'parameter-sweep.%s' % args.extension)

    plot_original_performance(tasks_X_values, tasks_y_values, output_file)
    remove_range_and_fit(tasks_X_values, tasks_y_values, 9, args.output_directory, args.extension)


if __name__ == '__main__':
    run(parse_args())
