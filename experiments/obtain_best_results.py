import pandas as pd
import numpy as np
import os
import shutil


def save_best_results():
    """ It makes a csv with the best model for each dataset in wape terms """
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

    if not os.path.exists('../results_best/'):
        os.mkdir('../results_best/')

    result_best_models.to_csv('../results_best/results_best.csv', sep=';')
    return result_best_models


def obtain_paths_predictions(result_best_models):
    """It obtains the path of the file .npy, which contains the prediction of the model"""
    paths = []
    models = result_best_models.reset_index()
    for i in range(len(models)):
        model = models.loc[i, :]

        path = model['DATASET'] + '/' + str(model['NORMALIZATION']) + '/' + str(
            model['PAST_HISTORY_FACTOR']) + '/' + str(model['EPOCHS']) + '/' + str(model['BATCH_SIZE']) + '/' + str(
            model['LEARNING_RATE']) + '/' + model['MODEL'] + '/' + str(model['MODEL_INDEX']) + '.npy'

        paths.append(path)
    return paths


def save_best_predictions(paths):
    """It saves the predictions in the dir ./results_best"""
    dir_res = '../results/'
    dir_dest = '../results_best/'
    for path in paths:
        dataset = path.split('/')[0].strip()
        modelo = path.split('/')[-2].strip()
        nombre = dataset + '_' + modelo + '.npy'

        if not os.path.exists(dir_dest + dataset):
            os.mkdir(dir_dest + dataset)

        shutil.copyfile(dir_res + path, dir_dest + '/' + dataset + '/' + nombre)
    print('[INFO] Results of the best models saved in ./results_best')


def obtain_best_results():
    result_best_models = save_best_results()
    paths = obtain_paths_predictions(result_best_models)
    save_best_predictions(paths)


if __name__ == '__main__':
    obtain_best_results()
