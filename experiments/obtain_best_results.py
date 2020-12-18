import pandas as pd
import numpy as np
import os


def obtain_best_results():
    datasets = os.listdir('../results/')
    base = pd.read_csv('../results/' + datasets[0] + '/results.csv', error_bad_lines=False, sep=';',
                       index_col='Unnamed: 0')
    columns = base.columns.values
    models = base['MODEL'].unique()
    result_best_models = pd.DataFrame(columns=columns)

    for dataset in datasets:
        results_dataset = pd.read_csv('../results/' + dataset + '/results.csv', error_bad_lines=False, sep=';',
                                      index_col='Unnamed: 0')
        for model in models:
            minimum = np.min(results_dataset.loc[results_dataset['MODEL'] == model, ['wape']]).values[0]
            best_model = results_dataset.loc[(results_dataset['wape'] == minimum) & (results_dataset['MODEL'] == model),
                         :]
            result_best_models = result_best_models.append(best_model)

    if not os.path.exists('../results/best_results/'):
        os.mkdir('../results/best_results/')

    result_best_models.to_csv('../results/best_results/best_results.csv', sep=';')


def main():
    obtain_best_results()


if __name__ == '__main__':
    main()
