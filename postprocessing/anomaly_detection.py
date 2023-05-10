import os
import argparse
import boto3
import s3fs
import json
import subprocess
import sys
import requests
import logging
from datetime import datetime
import numpy as np
import pandas as pd
from time import sleep
import sagemaker
from sagemaker.estimator import Estimator

logging.getLogger('tad').setLevel(logging.DEBUG)
from tad import anomaly_detect_ts

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

s3 = s3fs.S3FileSystem()
exec_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
exec_datetime_dt = datetime.now()
exec_date = exec_datetime_dt.date().strftime("%Y-%m-%d")


def get_session_and_client(region, default_bucket):
    """Gets the sagemaker session and client based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    sagemaker_session = sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )
    return sagemaker_session, sagemaker_client


def save_to_s3(data, path, **kwargs):
    with s3.open(path, 'w') as f:
        data.to_csv(f, **kwargs)


def parse_arguments():
    parser = argparse.ArgumentParser(description='anomaly detection')
    parser.add_argument(
        "--ingest-date",
        help="Ingest date in format Y-m-d (default: current day)",
        required=False,
        default=datetime.now().strftime("%Y-%m-%d")
    ),
    parser.add_argument(
        "--service",
        help="Service name",
    ),
    parser.add_argument(
        "--region",
        help="Aws region name",
    ),
    parser.add_argument(
        "--sagemaker-project-aws-region",
        help="Aws region name for sagemaker",
    ),
    parser.add_argument(
        "--s3-bucket",
        help="s3 bucket name",
    )
    parser.add_argument(
        "--aggregated-metrics-location",
        help="Location of aggregated metrics on S3",
    )
    parser.add_argument(
        "--aggregated-predictions-location",
        help="Location to aggregated predictions",
    )
    parser.add_argument(
        "--anomalies-location",
        help="Location of anomalies on S3",
    )
    parser.add_argument(
        "--clean-metric-location",
        help="Location of cleansed metric",
    )
    parser.add_argument(
        "--temp-bucket",
        help="Temporary s3 bucket name",
    )
    parser.add_argument(
        "--slack-url",
        help="Url to send slack notification",
    )
    parser.add_argument(
        "--sagemaker-execution-role",
        help="Sagemaker execution role",
    )
    parser.add_argument(
        "--kms-key-id",
        help="Sagemaker encryption key",
    )

    return parser.parse_args()


def dataset_from_s3(datapath, converters=None, dtype=None, **kwargs):
    if converters is None:
        converters = {}
    if not datapath.endswith('.csv'):
        fs = s3fs.core.S3FileSystem()
        datapath = fs.ls(datapath)[-1]
    with s3.open(datapath) as f:
        df = pd.read_csv(f, converters=converters, dtype=dtype, **kwargs)
    return df


def send_message(icon: str, title: str, message: str, slack_url: str):
    slack_data = {
        "username": 'anomalies-alert',
        "icon_emoji": icon,
        "attachments": [
            {
                "fields": [
                    {
                        "title": title,
                        "value": message,
                        "short": "false",
                    }
                ]
            }
        ]
    }

    byte_length = str(sys.getsizeof(slack_data))
    headers = {'Content-Type': "application/json", 'Content-Length': byte_length}
    response = requests.post(slack_url, data=json.dumps(slack_data), headers=headers)

    if response.status_code != 200:
        raise Exception(response.status_code, response.text)


def anomaly_to_message(anomaly: pd.Series) -> str:
    return f"from {anomaly['start_time']} till {anomaly['end_time']} {anomaly['annotation']}"


def notify_about_anomalies(anomaly_df: pd.DataFrame, service_name: str, region: str, processing_date: str,
                           slack_url: str):
    if anomaly_df.start_time.dtype != 'object':
        for c in ['start_time', 'end_time']:
            anomaly_df[c] = anomaly_df[c].dt.tz_localize(None).astype(str)
    mask_start = anomaly_df.start_time.apply(lambda x: x.split(' ')[0] == processing_date)
    mask_end = anomaly_df.end_time.apply(lambda x: x.split(' ')[0] == processing_date)
    current_anomalies = anomaly_df[mask_start | mask_end]

    title = f"Anomaly report for {service_name} {region} on {processing_date}"
    if not current_anomalies.empty:
        message = "The following anomalous periods were detected:\n" + \
                  "\n".join(anomaly_to_message(anomaly) for _, anomaly in current_anomalies.iterrows())
        send_message(icon=":cloudy:",
                     title=title,
                     message=message,
                     slack_url=slack_url)
    else:
        logger.info(f"No anomalies on {processing_date} detected!")


def create_anomalies_periods(data, anomaly_column='anomaly', anomalies_df=None):
    if anomalies_df is not None:
        anomalies_df.index.name = 'datetime'
        data = data.drop('datetime', axis=1).merge(anomalies_df, how='left', on='datetime')
        data_with_anom = data[~data[anomaly_column].isnull()]
        output_columns = ['start_time', 'end_time']
    else:
        data_with_anom = data[data[anomaly_column] == True]
        output_columns = ['start_time', 'end_time', 'annotation', 'algorithm']

    data_with_anom.loc[:, 'datetime'] = data_with_anom.index
    data_with_anom.loc[:, 'anomaly_index'] = data_with_anom.sort_index()['datetime'].diff().gt(
        np.timedelta64(30, 'm')).cumsum()

    data_with_anom.loc[:, 'diff_start'] = data_with_anom.anomaly_index.diff()
    data_with_anom['diff_start'].fillna(1, inplace=True)

    data_with_anom.loc[:, 'diff_end'] = data_with_anom.anomaly_index.diff(-1)
    data_with_anom['diff_end'].fillna(-1, inplace=True)

    data_with_anom.loc[:, 'start_time'] = data_with_anom['datetime'].where(data_with_anom['diff_start'] == 1)
    data_with_anom.loc[:, 'end_time'] = data_with_anom['datetime'].where(data_with_anom['diff_end'] == -1)

    data_with_anom.loc[:, 'start_time'] = data_with_anom['start_time'].ffill()
    data_with_anom.loc[:, 'end_time'] = data_with_anom['end_time'].bfill()

    anomalies_df = data_with_anom.drop_duplicates(subset=['start_time', 'end_time'])[output_columns]
    anomalies_df = anomalies_df.reset_index(drop=True)

    return anomalies_df


class APMAnomalyDetection:
    def __init__(
            self,
            ingest_date,
            service,
            session,
            sm_client,
            model_package_group_name,
            role,
            temp_bucket,
            train_period_start,
            train_period_end,
            columns_to_drop):

        self.ingest_date = ingest_date
        self.service = service
        self.session = session
        self.sm_client = sm_client
        self.model_package_group_name = model_package_group_name
        self.role = role
        self.temp_bucket = temp_bucket
        self.train_period_start = train_period_start
        self.train_period_end = train_period_end
        self.columns_to_drop = columns_to_drop

    def transform(self, traffic, predicted, cpu_util, error_rate, running_ecs=None, response_time=None):
        # Concatenate all datasets
        if response_time is not None:
            dfs = [traffic, predicted, cpu_util, error_rate, running_ecs, response_time]
        else:
            dfs = [traffic, predicted, cpu_util, error_rate, running_ecs]
        data = pd.concat(
            [df.assign(datetime=pd.to_datetime(df['datetime'], utc=True)).set_index('datetime') for df in dfs]
            , axis=1)

        # Limit data from min of cpu_util date to max of traffic date
        data = data.loc[cpu_util.datetime.min():traffic.datetime.max()].dropna()
        data.index = pd.Series(data.index).dt.tz_localize(None)

        return data

    def split(self, data):

        train_data = data.loc[self.train_period_start:self.train_period_end]

        return train_data

    @staticmethod
    def add_lags(df, n_lags=10):
        df = df.copy()
        for col in df.columns:
            df[f'{col}_diff_from_median'] = df[col] - df[col].median()
            df[f'{col}_more_quant_999'] = (df[col] > df[col].quantile(0.999)).astype('int')
            df[f'{col}_log'] = np.log(df[col] + 1)
            for l in range(1, n_lags + 1):
                df[f'{col}_lag_{l}'] = df[col].shift(l)
        return df

    def feature_engineering(self, data):

        data = self.add_lags(data).dropna()

        data['hour'] = data.index.hour
        data['weekday'] = data.index.dayofweek

        return data

    def fit(self, train_dataset):

        training_image_uri = sagemaker.image_uris.retrieve(
            framework="randomcutforest",
            region=sagemaker_project_aws_region,
            py_version="py3",
            instance_type="ml.m4.xlarge"
        )
        model_path = f"s3://{bucket_name}/airflow2/{BASE_JOB_PREFIX}/model/{MODEL_NAME}"

        rcf = Estimator(
            image_uri=training_image_uri,
            instance_type="ml.m4.2xlarge",
            instance_count=1,
            output_path=model_path,
            base_job_name=self.model_package_group_name,
            sagemaker_session=self.session,
            role=self.role,
            output_kms_key=kms_key_id,
            disable_profiler=True
        )

        rcf.set_hyperparameters(
            feature_dim=train_dataset.shape[1],
            num_samples_per_tree=512,
            num_trees=70,
        )
        train_location = f"s3://{self.temp_bucket}/rcf/input-{exec_datetime}"
        logger.info(f"Feature dim: {train_dataset.shape[1]}")
        train_dataset.to_csv(os.path.join(train_location, "train.csv"), index=False, header=None)
        train_data = sagemaker.inputs.TrainingInput(
            s3_data=train_location, content_type='text/csv;label_size=0', distribution='ShardedByS3Key'
        )

        rcf.fit({'train': train_data})
        return rcf

    def register_model(self, model):

        # Specify the model source
        model_url = f"{model.output_path}/{model.latest_training_job.job_name}/output/model.tar.gz"
        training_image_uri = model.training_image_uri()
        modelpackage_inference_specification = {
            "InferenceSpecification": {
                "Containers": [
                    {
                        "Image": training_image_uri,
                        "ModelDataUrl": model_url
                    }
                ],
                "SupportedContentTypes": ["text/csv"],
                "SupportedResponseMIMETypes": ["text/csv"],
            }
        }

        # Alternatively, you can specify the model source like this:
        # modelpackage_inference_specification["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]=model_url

        create_model_package_input_dict = {
            "ModelPackageGroupName": self.model_package_group_name,
            "ModelPackageDescription": "Model to detect anomalies in service metrics",
            "ModelApprovalStatus": "Approved"
        }
        create_model_package_input_dict.update(modelpackage_inference_specification)

        model_groups_info = self.sm_client.list_model_package_groups(NameContains="rcf", MaxResults=100)[
            'ModelPackageGroupSummaryList']
        model_groups = [item.get('ModelPackageGroupName') for item in model_groups_info]

        if self.model_package_group_name not in model_groups:
            logger.info("Creating new model package group first")
            model_package_group_input_dict = {
                "ModelPackageGroupName": self.model_package_group_name,
                "ModelPackageGroupDescription": "Model package group to detect anomalies in service metrics"
            }

            create_model_package_group_response = self.sm_client.create_model_package_group(
                **model_package_group_input_dict)
            logger.info(
                'ModelPackageGroup Arn : {}'.format(create_model_package_group_response['ModelPackageGroupArn']))
            sleep(10)

        create_model_package_response = self.sm_client.create_model_package(**create_model_package_input_dict)
        model_package_arn = create_model_package_response["ModelPackageArn"]
        logger.info('ModelPackage Version ARN : {}'.format(model_package_arn))

    def predict(self, model, prediction_data, columns_to_select, quantiles=None):

        # Upload prediction data to s3

        if quantiles is None:
            quantiles = [0.99]
        prediction_data_path = f"s3://{self.temp_bucket}/rcf-{exec_datetime}/input/temp_prediction_data.csv"
        save_to_s3(prediction_data, prediction_data_path, header=False, index=False)

        # Create transformer
        rcf_transformer = model.transformer(
            instance_count=1,
            instance_type="ml.m4.2xlarge",
            strategy="MultiRecord",
            assemble_with="Line",
            output_path=f"s3://{self.temp_bucket}/rcf-{exec_datetime}/output_predictions/",
        )
        # Transform - predict
        rcf_transformer.transform(prediction_data_path, content_type="text/csv", split_type="Line")
        rcf_transformer.wait()

        # Read predicted scores from s3
        predictions = pd.read_csv(rcf_transformer.output_path + 'temp_prediction_data.csv.out', header=None)

        prediction_data.loc[:, 'score'] = predictions[0].apply(lambda x: json.loads(x)['score']).values

        data_with_scores = prediction_data[columns_to_select + ['score']].copy()

        for q in quantiles:
            data_with_scores[f'anomaly_{q}'] = data_with_scores.score >= data_with_scores.score.quantile(q)

        return data_with_scores

    @staticmethod
    def thresholding_algo(y, lag, threshold, influence):
        signals = np.zeros(len(y))
        filteredY = np.array(y)
        avgFilter = [0] * len(y)
        stdFilter = [0] * len(y)
        avgFilter[lag - 1] = np.mean(y[0:lag])
        stdFilter[lag - 1] = np.std(y[0:lag])
        for i in range(lag, len(y)):
            if abs(y[i] - avgFilter[i - 1]) > threshold * stdFilter[i - 1]:
                if y[i] > avgFilter[i - 1]:
                    signals[i] = 1
                else:
                    signals[i] = -1

                filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i - 1]
                avgFilter[i] = np.mean(filteredY[(i - lag + 1):i + 1])
                stdFilter[i] = np.std(filteredY[(i - lag + 1):i + 1])
            else:
                signals[i] = 0
                filteredY[i] = y[i]
                avgFilter[i] = np.mean(filteredY[(i - lag + 1):i + 1])
                stdFilter[i] = np.std(filteredY[(i - lag + 1):i + 1])

        return dict(signals=np.asarray(signals),
                    avgFilter=np.asarray(avgFilter),
                    stdFilter=np.asarray(stdFilter))

    def add_annotation(self, data):

        df = data.copy()
        metrics = ['error_rate', 'cpu_util']
        signals = ['signal_er', 'signal_cu']

        d_error_rate = APMAnomalyDetection.thresholding_algo(y=df.error_rate.values, lag=24 * 12, threshold=5,
                                                             influence=0.2)
        df['signal_er'] = d_error_rate['signals']
        d_cpu_util = APMAnomalyDetection.thresholding_algo(y=df.cpu_util.values, lag=24 * 2, threshold=4, influence=0.2)
        df['signal_cu'] = d_cpu_util['signals']
        if self.service == 'fitt':
            d_response_time = APMAnomalyDetection.thresholding_algo(y=df.response_time.values, lag=24 * 2, threshold=6,
                                                                    influence=0.2)
            df['signal_rt'] = d_response_time['signals']
            metrics = metrics + ['response_time']
            signals = signals + ['signal_rt']

        factors = lambda x: [metric for metric, signal in zip(metrics, signals) if df.loc[x, signal] in [-1, 1]]

        df['datetime'] = df.index
        df['algorithm'] = 'Random Cut Forest'
        if 'anomaly' in df.columns:
            df['annotation'] = np.where(df['annotation'].isnull(), df['datetime'].apply(lambda x: factors(x)),
                                        df['annotation'])
            df['algorithm'] = df['algorithm'].mask(~df.anomaly.isnull(), 'Twitter Anomaly Detection')
        else:
            df['annotation'] = df['datetime'].apply(lambda x: factors(x))

        return df

    def drop_baseless_anomalies(self, df, quantiles=None):

        if quantiles is None:
            quantiles = [0.95, 0.99]
        for q in quantiles:
            df.loc[df['annotation'].apply(len) == 0, f'anomaly_{q}'] = False
        return df

    @staticmethod
    def detect_metrics_twitter_anomalies(data, max_anoms_dict):
        anomalies_df = pd.DataFrame()
        for metric in max_anoms_dict.keys():
            metric_df = data[[metric]]
            anomalies_tw = anomaly_detect_ts(metric_df[metric], max_anoms_dict[metric], direction='pos',
                                             alpha=0.01, threshold='med_max', verbose=True, plot=False)
            anomalies_tw = pd.DataFrame(anomalies_tw['anoms'], columns=['anomaly'])
            logger.info(f'Twitter anomalies df shape for metric {metric}: {anomalies_tw.shape}')
            if not anomalies_tw.empty:
                anomalies_tw.loc[:, 'annotation'] = metric
                anomalies_tw.loc[:, 'annotation'] = anomalies_tw['annotation'].apply(lambda x: [x])
                anomalies_df = pd.concat([anomalies_df, anomalies_tw])

        if not anomalies_df.empty:
            anomalies_df = anomalies_df.groupby(anomalies_df.index).agg({
                'anomaly': sum,
                'annotation': sum
            })
            return anomalies_df
        else:
            return None

    @staticmethod
    def add_twitter_anomalies(results, twitter_anomalies, anomaly_column='anomaly_0.99'):

        rcf_results = results.copy()
        twitter_anomalies.index.name = 'datetime'
        anomalies_combined = rcf_results.merge(twitter_anomalies, how='left', on='datetime')
        anomalies_combined[anomaly_column] = anomalies_combined[anomaly_column].mask(
            ~anomalies_combined.anomaly.isnull(), True)

        return anomalies_combined

    def detect_metrics_anomalies(self, data_raw, columns_to_select, max_anoms_dict, anomaly_column='anomaly_0.99'):

        data = data_raw.drop(columns=ad.columns_to_drop)
        data = ad.feature_engineering(data)

        train_data = ad.split(data)

        rcf = ad.fit(train_data)

        self.register_model(rcf)

        prediction_start = np.datetime64(ad.ingest_date, 'D') - PREDICTION_PERIOD
        prediction_data = data.loc[prediction_start:]

        logger.info(f'___Starting prediction on data from {prediction_start}')
        results = ad.predict(model=rcf, columns_to_select=columns_to_select, prediction_data=prediction_data)

        twitter_anomalies = APMAnomalyDetection.detect_metrics_twitter_anomalies(results, max_anoms_dict)
        if twitter_anomalies is not None:
            anomalies_combined = APMAnomalyDetection.add_twitter_anomalies(results, twitter_anomalies,
                                                                           anomaly_column=anomaly_column)
        else:
            anomalies_combined = results.copy()

        results_ann = ad.add_annotation(anomalies_combined)
        logger.info('Annotations added')

        results_ann_clean = ad.drop_baseless_anomalies(results_ann)

        final_result = create_anomalies_periods(results_ann_clean, anomaly_column=anomaly_column)
        final_result.loc[:, 'annotation'] = final_result['annotation'].apply(lambda x: 'based on ' + ', '.join(x))

        return final_result


# FUNCTION FOR ANOMALY DETECTION IN PREDICTIONS 

def detect_sport_twitter_anomalies(data, max_anoms_dict):
    sports_anom_periods = pd.DataFrame()
    for sport in max_anoms_dict.keys():
        df = data[['datetime', sport]]
        df.index = df.datetime
        anomalies = anomaly_detect_ts(df[sport], max_anoms=max_anoms_dict[sport], direction='pos',
                                      alpha=0.01, threshold='med_max', verbose=True, plot=False)

        anomalies = pd.DataFrame(anomalies['anoms'], columns=['anomaly'])

        if not anomalies.empty:
            anom_periods = create_anomalies_periods(df, anomalies_df=anomalies)
            anom_periods['annotation'] = f'based on the history of {sport} predictions'
            anom_periods['algorithm'] = 'Twitter Anomaly Detection'

            sports_anom_periods = pd.concat([sports_anom_periods, anom_periods], ignore_index=True)
    return sports_anom_periods


def delete_extra_model(sm_client, key_word, after_date):
    models = sm_client.list_models(
        NameContains=key_word, CreationTimeAfter=after_date, MaxResults=100)['Models']
    for model in models:
        model_name = model["ModelName"]
        logger.info(f"Deleting {model_name}")
        # Delete model
        sm_client.delete_model(ModelName=model_name)


if __name__ == "__main__":

    arguments = parse_arguments()
    ingest_date: str = arguments.ingest_date
    region = arguments.region
    service = arguments.service
    bucket_name = arguments.s3_bucket
    slack_url = arguments.slack_url
    sm_execution_role = arguments.sagemaker_execution_role
    sagemaker_project_aws_region = arguments.sagemaker_project_aws_region
    kms_key_id = arguments.kms_key_id
    temp_bucket = arguments.temp_bucket
    aggregated_metrics_location = arguments.aggregated_metrics_location
    aggregated_predictions_location = arguments.aggregated_predictions_location
    clean_metric_location = arguments.clean_metric_location
    anomalies_location = arguments.anomalies_location

    sagemaker_session, sagemaker_client = get_session_and_client(region=sagemaker_project_aws_region,
                                                                 default_bucket=bucket_name)

    region_underscored = region.replace("-", "_")

    BASE_JOB_PREFIX = "aiop-tf"
    MODEL_NAME = "rcf"
    # TEMP_BUCKET = "aiop-temporary-sandbox-us-east-1-051291311226"

    model_package_group_name = '-'.join(
        [
            BASE_JOB_PREFIX,
            service,
            region,
            MODEL_NAME
        ]
    )

    TRAIN_PERIOD_START = np.datetime64(ingest_date, 'D') - 90
    TRAIN_PERIOD_END = ingest_date
    PREDICTION_PERIOD = 61

    COLUMNS_TO_DROP = ['traffic', 'traffic_predicted', 'running_ecs']
    COLUMNS_TO_SELECT = ['error_rate', 'cpu_util']

    MAX_ANOM_DICT_METRICS = {
        'error_rate': 0.0001,
        'cpu_util': 0.0001
    }
    if service == 'fitt':
        MAX_ANOM_DICT_METRICS['response_time'] = 0.0001

    service_region = f"{service}/{region}"
    TRAFFIC_PATH = f's3://{bucket_name}/{clean_metric_location}/{service_region}/new_relic_application_summary_throughput/{ingest_date}/cleansed.csv'
    PREDICTED_PATH = f's3://{bucket_name}/{aggregated_predictions_location}/{service_region}/{ingest_date}/predictions_full.csv'
    CPU_UTIL_PATH = f's3://{bucket_name}/{aggregated_metrics_location}/{service_region}/cpuutilization/{ingest_date}'
    ERROR_RATE_PATH = f's3://{bucket_name}/{aggregated_metrics_location}/{service_region}/error_rate/{ingest_date}'
    RESPONSE_TIME_PATH = f's3://{bucket_name}/{aggregated_metrics_location}/{service_region}/response_time/{ingest_date}'
    RUNNING_ECS_PATH = f's3://{bucket_name}/{aggregated_metrics_location}/{service_region}/running_ecs_tasks/{ingest_date}'

    traffic = dataset_from_s3(TRAFFIC_PATH).rename(columns={'count': 'traffic'})
    logger.info('Actual traffic data has been read')
    predicted = dataset_from_s3(PREDICTED_PATH, parse_dates=['ds']).rename(
        columns={'ds': 'datetime', 'total': 'traffic_predicted'})
    logger.info('Predicted traffic data has been read')
    cpu_util = dataset_from_s3(CPU_UTIL_PATH, usecols=['datetime', 'count']).rename(columns={'count': 'cpu_util'})
    logger.info('CPU utilization data has been read')
    error_rate = dataset_from_s3(ERROR_RATE_PATH, usecols=['datetime', 'count']).rename(columns={'count': 'error_rate'})
    logger.info('Error rate data has been read')
    running_ecs = dataset_from_s3(RUNNING_ECS_PATH, usecols=['datetime', 'count']).rename(
        columns={'count': 'running_ecs'})
    logger.info('Running ECS data has been read')

    # RCF
    ad = APMAnomalyDetection(
        ingest_date,
        service,
        session=sagemaker_session,
        sm_client=sagemaker_client,
        role=sm_execution_role,
        model_package_group_name=model_package_group_name,
        temp_bucket=temp_bucket,
        train_period_start=TRAIN_PERIOD_START,
        train_period_end=TRAIN_PERIOD_END,
        columns_to_drop=COLUMNS_TO_DROP
    )
    if service == 'fitt':
        response_time = dataset_from_s3(RESPONSE_TIME_PATH, usecols=['datetime', 'count']).rename(
            columns={'count': 'response_time'})
        logger.info('Response time data has been read')
        COLUMNS_TO_SELECT = COLUMNS_TO_SELECT + ['response_time']

        data_raw = ad.transform(traffic, predicted[['datetime', 'traffic_predicted']], cpu_util, error_rate,
                                running_ecs, response_time)

    else:
        data_raw = ad.transform(traffic, predicted[['datetime', 'traffic_predicted']], cpu_util, error_rate,
                                running_ecs)

    metrics_anom_periods = ad.detect_metrics_anomalies(data_raw, max_anoms_dict=MAX_ANOM_DICT_METRICS,
                                                       columns_to_select=COLUMNS_TO_SELECT)
    logger.info('Anomalies detection based on service metrics by RCF and Twitter completed')

    # Twitter
    train_start_date = np.datetime64(ingest_date, 'D') - 60

    max_anoms_dict_sport = {
        'baseball': 0.001,
        'ufc': 0.001,
        'basketball': 0.001,
        'hockey': 0.0005,
        'football': 0.001,
    }

    prediction_data = predicted.loc[predicted.datetime > train_start_date]

    sports_anom_periods = detect_sport_twitter_anomalies(prediction_data, max_anoms_dict_sport)
    logger.info('Anomalies detection based on sports predictions by Twitter AD completed')

    all_results = pd.concat([metrics_anom_periods, sports_anom_periods], ignore_index=True)

    if not all_results.empty:
        logger.info(all_results)

        all_results.to_csv("./anomalies.csv", index_label="ds")
        subprocess.call(["aws", "s3", "cp", "./anomalies.csv",
                         f"s3://{bucket_name}/{anomalies_location}/{service}/{region}/{ingest_date}/"])

        # NOTIFICATION TO SLACK
        notify_about_anomalies(
            anomaly_df=all_results,
            service_name=service,
            region=region,
            processing_date=ingest_date,
            slack_url=slack_url)
    else:
        logger.info('Anomalous periods were not found')

    delete_extra_model(sagemaker_client, exec_date, exec_datetime_dt)

