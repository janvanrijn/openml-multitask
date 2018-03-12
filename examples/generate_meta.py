import argparse
import arff
import copy
import math
import numpy as np
import openml
import openmlcontrib

from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from openmlcontrib.preprocessing.imputation import ConditionalImputer

def parse_args():
    parser = argparse.ArgumentParser(description='Generate data for openml-pimp project')
    parser.add_argument('--parameter', type=str, default='gamma', help='the parameter to vary')
    parser.add_argument('--min', type=float, default=2**-15, help='min of range')
    parser.add_argument('--max', type=float, default=2**3, help='max of range')
    parser.add_argument('--log_base', type=float, default=2, help='base of log scale')
    parser.add_argument('--steps', type=int, default=19, help='number of steps to generate the dataset with')
    parser.add_argument('--tasks', type=int, nargs='+', default=[3, 11, 12, 14, 15, ], help='defines the tasks')
    parser.add_argument('--output_file', type=str, default='multi_task.svm')

    return parser.parse_args()


def construct_pipeline(indices=None):
    steps = [('imputation', ConditionalImputer(strategy='median',
                                               fill_empty=0,
                                               categorical_features=indices,
                                               strategy_nominal='most_frequent')),
             ('hotencoding', OneHotEncoder(handle_unknown='ignore',
                                           categorical_features=indices)),
             ('scaling', StandardScaler(with_mean=False)),
             ('variencethreshold', VarianceThreshold()),
             ('classifier', SVC(kernel='rbf', random_state=1))]
    return Pipeline(steps=steps)


if __name__ == '__main__':
    args = parse_args()

    if args.log_base is not None:
        required_values = np.logspace(math.log(args.min, args.log_base),
                                      math.log(args.max, args.log_base),
                                      num=args.steps, base=args.log_base)
    else:
        raise NotImplementedError()

    pipeline = construct_pipeline()
    flow = openml.flows.sklearn_to_flow(pipeline)
    flow_id = openml.flows.flow_exists(flow.name, flow.external_version)

    all_setups = openml.setups.list_setups(flow=flow_id)
    all_evaluations = openml.evaluations.list_evaluations('predictive_accuracy', flow=[flow_id], task=args.tasks)

    results = dict()

    for task_id in args.tasks:
        task = openml.tasks.get_task(task_id)
        data_name = task.get_dataset().name
        data_qualities = task.get_dataset().qualities
        print("Obtained task %d (%s); %s attributes; %s observations" %
              (task_id, data_name, data_qualities['NumberOfFeatures'], data_qualities['NumberOfInstances']))

        indices = task.get_dataset().get_features_by_type('nominal', [task.target_name])
        pipeline.set_params(hotencoding__categorical_features=indices, imputation__categorical_features=indices)

        task_evaluations = openmlcontrib.misc.filter_listing(all_evaluations, 'task_id', [task_id], dict_representation=False)
        task_setups = copy.deepcopy(all_setups)
        for param, value in pipeline.get_params().items():
            if param in [i[0] for i in pipeline.steps] or param in ['steps']: #TODO: steps
                continue
            param_name = param.split('__')[-1]
            if param_name == args.parameter or param_name in ['dtype']: #TODO: fix dtype
                continue
            # print('before', param_name, len(task_setups))
            if isinstance(value, int) or isinstance(value, float):
                task_setups = openmlcontrib.setups.filter_setup_list(task_setups, param_name, min=value, max=value)
            else:
                task_setups = openmlcontrib.setups.filter_setup_list(task_setups, param_name, allowed_values=[value])
            # print('after', param_name, len(task_setups))

        for value in required_values:
            params_dict = {'classifier__' + args.parameter: value}
            pipeline.set_params(**params_dict)
            current_setups = openmlcontrib.setups.filter_setup_list(task_setups, args.parameter, value, value)
            current_evaluations = openmlcontrib.misc.filter_listing(task_evaluations, 'setup_id', list(current_setups.keys()), dict_representation=False)
            #
            # if len(current_evaluations) == 0:
            #     print('Couldn\'t find value %f: %d setups %d runs' % (value, len(current_setups), len(current_evaluations)))
            #     run = openml.runs.run_model_on_task(task, pipeline)
            #     run.publish()
            #     print(run.run_id)

            if len(current_evaluations) > 0:
                arb_run_id = next(iter(current_evaluations))
                if value not in results:
                    results[value] = dict()
                results[value][task_id] = current_evaluations[arb_run_id].value

    # TO ARFF
    param_name = args.parameter
    if args.log_base is not None:
        param_name += '_log'
    attributes = list((param_name, 'NUMERIC'))
    for task_id in args.tasks:
        attributes.append(('score_at_task_' + str(task_id), 'NUMERIC'))

    data = list()
    for param_val, task_score in results.items():
        param_val_scaled = param_val
        if args.log_base is not None:
            param_val_scaled = math.log(param_val, args.log_base)
        current = [param_val_scaled]
        for task_id in args.tasks:
            current.append(results[param_val][task_id])
        data.append(current)

    relation = 'multitask-svm'
    description = 'Generated using openml-multitask repository'
    dataset = {'relation': relation, 'attributes': attributes, 'data': data, 'description' : description}
    with open(args.filename, 'w') as fp:
        fp.write(arff.dumps(dataset))
