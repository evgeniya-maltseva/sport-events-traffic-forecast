"""Evaluation script for measuring mean squared error."""
import sys
import json
import logging
import pathlib
import pickle
import tarfile

import numpy as np
import pandas as pd
import xgboost

from sklearn.metrics import mean_absolute_error, median_absolute_error, mean_squared_error

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def mean_absolute_percentage_error(y_true, y_pred):
    mask = y_true != 0
    return np.nanmean(np.abs((y_true - y_pred)/y_true)[mask])

def symmetric_mape(y_true, y_pred):
    mask = y_true > 0
    y_true, y_pred = y_true[mask], y_pred[mask]
    return np.nanmean(np.abs(y_pred - y_true)/ ((y_true + y_pred)/2))


def median_absolute_percentage_error(y_true, y_pred):
    return np.nanmedian(np.abs((y_true - y_pred)/y_true))


if __name__ == "__main__":
    model_group_name = sys.argv[1]
    approval_status = sys.argv[2]
    execution_date_str = sys.argv[3]
    region = sys.argv[4]
    algorithm_name = sys.argv[5]
    service = sys.argv[6]
    environment = sys.argv[7]

    logger.debug("Starting evaluation.")
    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path="")

    logger.debug("Loading xgboost model.")
    model = pickle.load(open("xgboost-model", "rb"))

    logger.debug("Reading test data.")
    test_path = "/opt/ml/processing/val/val.csv"
    df = pd.read_csv(test_path, header=None)
    logger.info(df.shape)

    logger.debug("Reading test data.")
    # y_test = df.iloc[:, 0].to_numpy()
    y_test = df.iloc[:, 0].values
    # logger.info(len(y_test))
    # logger.info(y_test)
    # df.drop(df.columns[0], axis=1, inplace=True)
    X_test = xgboost.DMatrix(df.iloc[:, 2:].values)

    logger.info("Performing predictions against test data.")
    predictions = model.predict(X_test)
    # logger.info(len(predictions))
    # logger.info(predictions)

    logger.debug("Calculating mean squared error.")
    # mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    med_ae = median_absolute_error(y_test, predictions)
    mape = mean_absolute_percentage_error(y_test, predictions)
    med_ape = median_absolute_percentage_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    smape = symmetric_mape(y_test, predictions)
    std = np.std(y_test - predictions)
    metric_std = 0

    regression_metrics = {
        # "mse": {
        #     "value": mse,
        #     "standard_deviation": std
        # },
        "mae": {
            "value": float(format(mae, '.10f')),
            "standard_deviation": metric_std
        },
        "med_ae": {
            "value": float(format(med_ae, '.10f')),
            "standard_deviation": metric_std
        },
        "mape": {
            "value": float(format(mape, '.10f')),
            "standard_deviation": metric_std
        },
        "smape": {
            "value": float(format(smape, '.10f')),
            "standard_deviation": metric_std
        },
        "med_ape": {
            "value": float(format(med_ape, '.10f')),
            "standard_deviation": metric_std
        },
        "mse": {
            "value": float(format(mse, '.10f')),
            "standard_deviation": metric_std
        },
        "std": {
            "value": float(format(std, '.10f')),
            "standard_deviation": metric_std
        },
    }

    report_dict = {
        "regression_metrics": regression_metrics,
    }

    output_dir = "/opt/ml/processing/evaluation"
    metrics_dir = "/opt/ml/processing/metrics"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(metrics_dir).mkdir(parents=True, exist_ok=True)

    with open(f"{metrics_dir}/{algorithm_name}.json", "w") as f:
        f.write(json.dumps({
            "regression_metrics": regression_metrics,
            'region': region,
            'service': service,
            'environment': environment,
            'model_group_name': model_group_name,
            'algorithm_name': algorithm_name
        }))

    # logger.info("Writing out evaluation report with mse: %f", mse)
    logger.info("Writing out evaluation report")
    logger.info(f"Mean Absolute Error: {mae}")
    logger.info(f"Median Absolute Error: {med_ae}")
    logger.info(f"Mean Absolute Percentage Error: {mape}")
    logger.info(f"Median Absolute Percentage Error: {med_ape}")
    logger.info(f"Symmetric Mean Absolute Percentage Error: {smape}")
    logger.info(f"Mean Squared Error: {mse}")
    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
