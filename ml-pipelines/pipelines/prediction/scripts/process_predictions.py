import argparse
import pandas as pd
import os
import sys
import json
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

DATETIME_COL_NAME = "ds"
PREDICT_EXTENSION = 'out'
METADATA_EXTENSION = 'csv'
OUTPUT_DATETIME_COL_NAME = DATETIME_COL_NAME


def get_input_filename(input_path, extension='csv'):
    input_filenames = [f for f in os.listdir(input_path)
                       if os.path.isfile(os.path.join(input_path, f)) and f.endswith(extension)]
    if len(input_filenames) == 0:
        logger.error(f"No {extension} files found in {input_path}")
        return None
    elif len(input_filenames) > 1:
        logger.warning(f"Expected only one {extension} file in {input_path}, found many")
        logger.warning(f"Will return latest {extension} filename in {input_path}")
        input_filenames.sort(key=lambda x: os.stat(os.path.join(input_path, x)).st_mtime)
    input_filename = input_filenames[0]
    logger.info(f"Returning {input_filename} in input path: {input_path}")
    return input_filename


def save_processed_data(data, output_path, output_filename,
                        data_message="Saving output data to local output path", header=False, exclude_cols=None):
    output_data = data.copy()
    if (not isinstance(output_data, pd.DataFrame)) or output_data.empty:
        logger.error("Output data is not pandas dataframe or empty")
        return
    logger.info(data_message)
    logger.info(f"Data shape: {output_data.shape}")
    logger.info(output_data.head(2))
    try:
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        logger.info(f"Local path: {output_path}")
        if exclude_cols:
            output_data.drop(columns=exclude_cols, inplace=True, errors='ignore')
        output_data.to_csv(os.path.join(output_path, output_filename), index=False, header=header)
        logger.info("Data saved successfully")
    except Exception as ex:
        logger.error(ex)
        logger.error("Error saving output data")


if __name__ == '__main__':
    logger.info("Starting prediction postprocessing")
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata-columns', type=str)
    parser.add_argument("--region", type=str)
    args = parser.parse_args()

    metadata_cols = args.metadata_columns.split(',')
    if len(metadata_cols) == 0:
        logger.error("No metadata selected")
        sys.exit(1)
    logger.info(f"Metadata: {metadata_cols}")
    region = args.region
    logger.debug(f"Model region: {region}")
    base_dir = "/opt/ml/processing"
    input_predictions_dir = "input_predictions"
    input_timestamps_dir = "input_predictions_ts"

    input_predictions_path = os.path.join(base_dir, input_predictions_dir)
    input_predictions_filename = get_input_filename(input_predictions_path, extension=PREDICT_EXTENSION)
    input_timestamps_path = os.path.join(base_dir, input_timestamps_dir)
    input_timestamps_filename = get_input_filename(input_timestamps_path, extension=METADATA_EXTENSION)

    output_dir = "output_predictions"
    output_filename = input_predictions_filename.replace(f".{PREDICT_EXTENSION}", "")
    if not output_filename.endswith(".csv"):
        output_filename = output_filename + ".csv"

    try:
        predictions = pd.read_csv(os.path.join(input_predictions_path,
                                               input_predictions_filename), names=['yhat'])
        if predictions['yhat'].dtypes == 'float64': 
            logger.info("Type of predictions is float")
        elif predictions['yhat'].dtypes == 'object':
            predictions.yhat = predictions.yhat.apply(lambda x: json.loads(x)['score'])
            logger.info("Type of predictions is object")
        else:
            logger.info("Type of predictions is unknown")
    except Exception as e:
        logger.error(f"Could not ingest predictions file")
        logger.exception(e)
        sys.exit(1)

    try:
        timestamps = pd.read_csv(os.path.join(input_timestamps_path,
                                              input_timestamps_filename), header=0)
        if isinstance(timestamps, pd.Series):
            timestamps = timestamps.to_frame()
    except Exception as e:
        logger.error(f"Could not ingest prediction timestamp file")
        logger.exception(e)
        sys.exit(1)

    assert len(predictions) == len(timestamps), "Predictions size is unequal to timestamps size"

    for col in metadata_cols:
        if col in timestamps.columns:
            predictions[col] = timestamps[col]
        else:
            logger.error(f"Metadata column {col} not found!")

    output_path = os.path.join(base_dir, output_dir)
    save_processed_data(predictions, output_path, output_filename,
                        data_message="Saving predictions with metadata", header=True)
