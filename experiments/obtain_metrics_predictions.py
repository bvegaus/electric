import openpyxl
import os
import pandas as pd
import numpy as np
from metrics import METRICS


def get_models(datasets):
    """It obtains the models used into the experiments"""
    dataframe = pd.read_csv('../results/' + datasets[0] + '/results.csv', sep=';')
    models = dataframe['MODEL'].unique()
    return models.tolist()


def get_best_prediction(results, metric, model, dataset):
    """It calculates the best prediction of one model in one dataset"""
    model_rows = results.loc[results['MODEL'] == model, :]
    for index, row in model_rows.iterrows():
        path_y_test_denorm = '../data/' + dataset + '/' + np.str(row['NORMALIZATION']) + '/' + \
                             np.str(row['PAST_HISTORY_FACTOR']) + '/'

        path_preds = '../results/' + dataset + '/' + np.str(row['NORMALIZATION']) + '/' + np.str(
            row['PAST_HISTORY_FACTOR']) + '/' + np.str(row['EPOCHS']) + '/' + np.str(row['BATCH_SIZE']) + '/' + np.str(
            row['LEARNING_RATE']) + '/' + model + '/' + np.str(row['MODEL_INDEX']) + '.npy'

        y_test_denorm = np.load(path_y_test_denorm + 'y_test_denorm.np.npy').flatten()
        preds = np.load(path_preds).flatten()

        value = METRICS[metric](y_test_denorm, preds)

    return value


def create_excel():
    """It create the excel where the results are going to be saved"""
    if not os.path.exists('../results_best/'):
        os.mkdir('../results_best/')

    excel = pd.ExcelWriter('../results_best/metrics_by_predictions.xlsx', engine='openpyxl')
    excel.book = openpyxl.Workbook()
    return excel


def calculate_metrics(datasets, models, metrics, excel):
    """It calculate the metrics, of each model in each dataset, and save them into the excel"""
    columns_names = ['dataset'] + models

    for metric in metrics:
        res = pd.DataFrame(columns=columns_names).set_index('dataset')
        for dataset in datasets:

            results = pd.read_csv('../results/' + dataset + '/results.csv', sep=';', index_col='Unnamed: 0')
            row = []

            for model in models:
                value = get_best_prediction(results, metric, model, dataset)
                row.append(value)

            res.loc[dataset, :] = row

        res.to_excel(excel, sheet_name=metric)

    return excel


def save_excel(excel):
    """It saves the excel with the information"""
    default_sheet = excel.book[excel.book.sheetnames[0]]
    excel.book.remove(default_sheet)
    excel.save()
    excel.close()


def get_metrics():
    """Calculate the best values for a metrics of each model of each dataset, and saves the results into the sheets
    of an excel"""
    metrics = ['mse', 'rmse', 'mae', 'wape', 'mase']
    datasets = os.listdir('../results/')
    models = get_models(datasets)

    excel = create_excel()
    excel = calculate_metrics(datasets, models, metrics, excel)
    save_excel(excel)

    print('[INFO] Values of the metrics by predictions saved into "./results_best/metrics_by_predictions.xlsx"')


if __name__ == '__main__':
    get_metrics()
