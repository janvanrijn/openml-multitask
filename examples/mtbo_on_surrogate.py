import argparse
import os
import json
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)

from robo.fmin import mtbo

from hpolib.benchmarks.ml.surrogate_svm import SurrogateSVM


def parse_args():
    parser = argparse.ArgumentParser(description='Runs MTBO benchmark. Code provided by Aaron Klein')
    parser.add_argument('--surrogate_directory', type=str, default='../data/svm_on_mnist_grid')
    parser.add_argument('--output_directory', type=str, default='/home/janvanrijn/experiments/multitask/mtbo')

    parser.add_argument('--run_id', type=int, default=1)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--dataset_divider', type=int, default=16)

    return parser.parse_args()


def run(args):
    run_id = args.run_id # whatever it means
    seed = args.random_state
    auxillay_dataset = args.dataset_divider

    rng = np.random.RandomState(seed)

    f = SurrogateSVM(path=args.surrogate_directory, rng=rng)
    num_iterations = 80

    os.makedirs(args.output_directory, exist_ok=True)

    def objective(x, task):
        if task == 0:
            dataset_fraction = float(1/auxillay_dataset)
        elif task == 1:
            dataset_fraction = 1

        res = f.objective_function(x, dataset_fraction=dataset_fraction)
        return res["function_value"], res["cost"]

    info = f.get_meta_information()
    bounds = np.array(info['bounds'])
    results = mtbo(objective_function=objective,
                   lower=bounds[:, 0], upper=bounds[:, 1],
                   n_init=5, num_iterations=num_iterations, n_hypers=50,
                   rng=rng, output_path=args.output_directory, inc_estimation="last_seen")

    results["run_id"] = run_id
    results['X'] = results['X'].tolist()
    results['y'] = results['y'].tolist()
    results['c'] = results['c'].tolist()

    test_error = []
    current_inc = None
    current_inc_val = None
    cum_cost = 0

    for i, inc in enumerate(results["incumbents"]):
        print(inc)
        if current_inc == inc:
            test_error.append(current_inc_val)
        else:
            y = f.objective_function_test(inc)["function_value"]
            test_error.append(y)

            current_inc = inc
            current_inc_val = y
        print(current_inc_val)

        # Compute the time it would have taken to evaluate this configuration
        c = results["c"][i]
        cum_cost += c

        # Estimate the runtime as the optimization overhead + estimated cost
        results["runtime"][i] += cum_cost
        results["test_error"] = test_error

        with open(os.path.join(args.output_directory, 'results_%d.json' % run_id), 'w') as fh:
            json.dump(results, fh)


if __name__ == '__main__':
    run(parse_args())