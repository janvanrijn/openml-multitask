import matplotlib.pyplot as plt


def plot_boxplots(results_dict, measure, title, output_file):
    fig, ax = plt.subplots(1, 1)

    labels = []
    series = []
    for model in results_dict:
        current_series = list()
        for task in results_dict[model]:
            current_series.append(results_dict[model][task][measure])

        labels.append(model)
        series.append(current_series)

    # basic plot
    ax.boxplot(series)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
