import matplotlib.pyplot as plt


def plot_boxplots(results_dict, measures, output_file):
    fig, ax = plt.subplots(len(measures), 1, figsize=(8,6*len(measures)))

    for idx, measure in enumerate(measures):
        labels = []
        series = []

        for model in results_dict:
            current_series = list()
            for task in results_dict[model]:
                current_series.append(results_dict[model][task][measure])

            labels.append(model)
            series.append(current_series)

        # basic plot
        ax[idx].boxplot(series)
        ax[idx].set_xticklabels(labels, rotation=45, ha='right')
        ax[idx].set_title(measure)

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
