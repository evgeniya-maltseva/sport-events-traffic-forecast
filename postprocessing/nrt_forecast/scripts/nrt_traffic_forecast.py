import sys
import argparse
import dataclasses
import datetime
import logging
import subprocess
from functools import reduce
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import math
import numpy as np
import pandas as pd
import s3fs
from datadog import initialize, api
from elasticsearch import Elasticsearch, RequestsHttpConnection, helpers
from pandas import Timestamp
from scipy.special import inv_boxcox
from scipy.stats import boxcox
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.seasonal import STL, seasonal_decompose

DATE_PATTERN = "%Y-%m-%d"
DATETIME_PATTERN = "%Y-%m-%dT%H:%M:%S"
DATETIME_PATTERN_FOR_ES = "%Y-%m-%d %H:%M:%S"
DATETIME_PATTERN_WITH_Z = "%Y-%m-%dT%H:%M:%S.000Z"

DATE_PATTERN_WITHOUT_DAY = "%Y-%m"

DS_COL_NAME = "ds"
DATETIME_COL_NAME = "datetime"
COUNT_COL_NAME = "count"
N_TASKS_COL_NAME = "n_tasks"
TIMESTAMP_COL_NAME = "timestamp"
SERIES_START_COL_NAME = "series_start"
ENVIRONMENT_COL_NAME = "environment"
REGION_COL_NAME = "region"
SERVICE_COL_NAME = "service"
START_DATE_COL_NAME = "start_date"

AIOPS_SERVICE_COL_NAME = "aiops_service"
METADATA_COL_NAME = "metadata"
START_DS_COL_NAME = "start_ds"
END_DS_COL_NAME = "end_ds"

# variables for CommonTaskPreparation
START_COL_NAME = "start"
END_COL_NAME = "end"
END_DATE_COL_NAME = "end_date"
INTERVAL_COL_NAME = "interval"
INTERVAL_ID_COL_NAME = "interval_id"
MAX_TASKS_TS_COL_NAME = "max_tasks_timestamp"
MAX_TASKS_COL_NAME = "max_tasks"
HAS_EXTRA_COL_NAME = "has_extra"
DATE_COL_NAME = "date"
EVENTS_COL_NAME = "events"
SPORT_COL_NAME = "sport"
LEAGUE_COL_NAME = "league"
PREDICTION_DATE_COL_NAME = "prediction_date"
ORIGINAL_PREDICTION_DATE_COL_NAME = "original_prediction_date"

TRAFFIC_GRANULARITY = "5min"
TRAFFIC_GRANULARITY_TIMEDELTA = pd.to_timedelta(TRAFFIC_GRANULARITY)
TWO_HOURS_TIMEDELTA = pd.to_timedelta("2h")
ONE_HOUR_TIMEDELTA = pd.to_timedelta("1h")
ONE_HOUR_POINTS = int(pd.to_timedelta("1h") / TRAFFIC_GRANULARITY_TIMEDELTA)
ONE_DAY_POINTS = int(pd.to_timedelta("1d") / TRAFFIC_GRANULARITY_TIMEDELTA)
ONE_WEEK_POINTS = int(pd.to_timedelta("1w") / TRAFFIC_GRANULARITY_TIMEDELTA)

YEAR_MONTH_PATTERN = "%Y-%m"
INTERVAL_ID_COL_NAME = "interval_id"
RECOMMENDATION_TYPE_COL_NAME = "recommendation_type"

s3 = s3fs.S3FileSystem()

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def parse_arguments(args=None):
    parser = argparse.ArgumentParser(description='NRT forecast for traffic')
    parser.add_argument(
        "--prediction-datetime",
        help="Datetime for making forecast in format",
        required=False,
        default=datetime.now().strftime(DATETIME_PATTERN),
    )
    parser.add_argument(
        "--query",
        help="DataDog query"
    )
    parser.add_argument(
        "--metric-name",
        help="DataDog metric name"
    )
    parser.add_argument(
        "--data-bucket",
        help="Target bucket",
    )
    parser.add_argument(
        "--clean-metric-location",
        help="Location of cleansed metric",
    )
    parser.add_argument(
        "--extra-loaded-metric-location",
        help="Location of extra loaded with NRT metrics",
    )
    parser.add_argument(
        "--nrt-forecast-location",
        help="Location of NRT forecast",
    )
    parser.add_argument(
        "--aggregated-predictions-location",
        help="Location to aggregated predictions",
    )
    parser.add_argument(
        "--task-predictions-location",
        help="Location to aggregated predictions",
    )
    parser.add_argument(
        "--adjusted-task-predictions-location",
        help="Location to aggregated predictions",
    )
    parser.add_argument(
        "--task-scaling-recommendations-folder-pattern",
        help="Templated location of recommendations",
    )
    parser.add_argument(
        "--adjusted-task-scaling-recommendations-folder-pattern",
        help="Templated location of adjusted recommendations",
    )
    parser.add_argument(
        "--datadog-api-key",
        help="Datadog API key",
    )
    parser.add_argument(
        "--datadog-app-key",
        help="Datadog app key",
    )
    parser.add_argument(
        "--service-name",
        help="Service name",
    )
    parser.add_argument(
        "--region",
        help="Region",
    )
    parser.add_argument(
        "--environment",
        help="Environment",
    )
    parser.add_argument(
        "--history-days-limit",
        help="N days history to train NRT forecast",
        type=int
    )
    parser.add_argument(
        "--es-hosts",
        help="ES hosts separated with comma",
        required=False,
        type=lambda s: s.split(",")
    )
    parser.add_argument(
        "--polyfit-degree",
        help="Degree for polyfit in NRT forecast",
        type=int
    )
    parser.add_argument(
        "--weights-tau",
        help="Weights tau in NRT forecast",
        type=float
    )
    parser.add_argument(
        "--means-diff-percent-threshold-up",
        help="Threshold for diff in nrt forecast and prediction means to act when nrt forecast is higher",
        type=int
    )
    parser.add_argument(
        "--means-diff-percent-threshold-down",
        help="Threshold for diff in nrt forecast and prediction means to act when nrt forecast is lower",
        type=int
    )
    parser.add_argument(
        "--change-up-threshold",
        help="Threshold for diff in n_tasks to make changes if n_tasks are more than before",
        type=float
    )
    parser.add_argument(
        "--change-down-threshold",
        help="Threshold for diff in n_tasks to make changes if n_tasks are fewer than before",
        type=float
    )
    parser.add_argument(
        "--slope-threshold",
        help="Threshold for nrt forecast slope to make changes",
        type=int
    )
    parser.add_argument(
        "--horizon",
        help="Time horizon for forecast in W, D, h, m, S, ms, us, ns"
    )
    parser.add_argument(
        "--aws-service",
        help="AWS service name"
    )
    parser.add_argument(
        "--services-info-path",
        help="Path to json with service additional info"
    )

    return parser.parse_args(args)


def upload_to_s3(local_filename: str,
                 target_s3_folder_path: str):
    target_path = target_s3_folder_path if target_s3_folder_path.endswith("/") else target_s3_folder_path + "/"
    logger.info(f"Uploading '{local_filename}' to '{target_path}'")
    subprocess.check_call([
        "aws", "s3", "cp",
        f"./{local_filename}",
        target_path])


# TODO: shared code with dynamic_task_scaling_recommendations.py
def prepare_s3_folder_path(path: str) -> str:
    path_with_prefix = path if path.startswith("s3://") else f"s3://{path}"
    path_with_slash = path_with_prefix if path_with_prefix.endswith("/") else f"{path_with_prefix}/"
    return path_with_slash


def upload_to_s3(local_filename: str,
                 target_s3_folder_path: str):
    prepared_path = prepare_s3_folder_path(target_s3_folder_path)
    logger.info(f"Uploading '{local_filename}' to '{prepared_path}'")
    subprocess.check_call([
        "aws", "s3", "cp",
        f"./{local_filename}",
        prepared_path])


def write_json_to_s3(df: pd.DataFrame,
                     s3_folder_path: str,
                     filename: str = "task_recommendations.json"):
    df.to_json(path_or_buf=f"./{filename}",
               orient='records',
               date_format='iso')
    upload_to_s3(filename, s3_folder_path)


# TODO: shared code with dynamic_task_scaling_recommendations.py and task_scaling_recommendations.py
class CommonTaskPreparation:
    columns_to_merge_service_info: List[str]

    def __init__(self,
                 process_date: datetime,
                 bucket: str,
                 sports_list: List[str],
                 services_info_path: str,
                 calendar_pattern: str,
                 prediction_period: str = "1W",
                 min_time_between_intervals: str = "1h",
                 ):
        self.process_date = process_date
        self.formatted_date = process_date.strftime(DATE_PATTERN)
        self.bucket = bucket
        self.sports = sports_list
        self.calendar_pattern = calendar_pattern.replace("//", "/")

        self.full_service_info_path = f"s3://{self.bucket}/{services_info_path}"
        self.service_info_df = pd.read_json(self.full_service_info_path)
        self.service_info_df["account_id"] = self.service_info_df["account_id"].astype(str)

        # to filter out recommendations from yesterday
        self.prediction_period_start: datetime = process_date - pd.to_timedelta("5m")
        self.prediction_period_end: datetime = self.prediction_period_start + pd.to_timedelta(prediction_period)
        self.min_time_between_intervals = pd.to_timedelta(min_time_between_intervals)

    @staticmethod
    def generate_id(recommendations: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    def add_meta_info_column(self,
                             recommendations: pd.DataFrame,
                             prediction_date: datetime) -> pd.DataFrame:
        return recommendations \
            .assign(**{METADATA_COL_NAME: [{PREDICTION_DATE_COL_NAME: prediction_date}] * len(recommendations)})

    def read_calendar(self) -> pd.DataFrame:
        path = self.calendar_pattern.format(date=self.process_date)
        return pd.read_csv(f"s3://{self.bucket}/{path}")

    def aggregate_calendar_by_sport(self,
                                    calendar: pd.DataFrame) -> Dict:
        calendar[START_COL_NAME] = pd.to_datetime(calendar["date_utc"])
        calendar[END_COL_NAME] = pd.to_datetime(calendar["end_with_avg_duration"])

        calendar[SPORT_COL_NAME] = np.where(calendar[LEAGUE_COL_NAME] == "ufc", "ufc", calendar[SPORT_COL_NAME])

        filtered: pd.DataFrame = calendar.loc[(calendar[START_COL_NAME] <= self.prediction_period_end) &
                                              (calendar[END_COL_NAME] >= self.prediction_period_start)]

        filtered[EVENTS_COL_NAME] = filtered[[START_COL_NAME, END_COL_NAME, "id", "name", LEAGUE_COL_NAME]] \
            .apply(lambda x: x.to_dict(), axis=1)
        return filtered \
            .groupby(SPORT_COL_NAME) \
            .agg({EVENTS_COL_NAME: list}) \
            .to_dict("index")

    def calculate_sport_weights_in_extra_traffic(self,
                                                 record: Dict,
                                                 calendar: Dict):
        extra_traffic = {sport: record[sport] for sport in self.sports}
        total_extra_traffic = sum(extra_traffic.values())

        record["sport_weight_in_extra_traffic"] = {
            sport: {
                "weight": traffic / total_extra_traffic,
                EVENTS_COL_NAME: [event for event in calendar.get(sport, {}).get(EVENTS_COL_NAME, [])
                                  if (event[START_COL_NAME] <= record[END_DS_COL_NAME])
                                  and (event[END_COL_NAME] >= record[START_DS_COL_NAME])]
            }
            for sport, traffic in extra_traffic.items()}
        return record

    def add_service_info(self,
                         recommendations: pd.DataFrame) -> pd.DataFrame:
        return recommendations.merge(self.service_info_df, how="left", on=self.columns_to_merge_service_info)

    def find_intervals_and_corresponding_tasks(self,
                                               traffic: pd.DataFrame,
                                               tasks: pd.DataFrame,
                                               aggregated_calendar: Dict) -> pd.DataFrame:
        traffic[DS_COL_NAME] = pd.to_datetime(traffic[DS_COL_NAME])
        tasks[DS_COL_NAME] = pd.to_datetime(tasks[DS_COL_NAME])

        traffic[HAS_EXTRA_COL_NAME] = reduce(lambda a, b: a | b,
                                             ((traffic[col_name] > 0) for col_name in self.sports))

        filtered_traffic = traffic.loc[
            traffic[HAS_EXTRA_COL_NAME] &
            (traffic[DS_COL_NAME] >= self.prediction_period_start) &
            (traffic[DS_COL_NAME] <= self.prediction_period_end)]
        filtered_tasks = tasks[[N_TASKS_COL_NAME, DS_COL_NAME]].loc[
            (tasks[DS_COL_NAME] >= self.prediction_period_start) &
            (tasks[DS_COL_NAME] <= self.prediction_period_end)]

        filtered_traffic[INTERVAL_COL_NAME] = filtered_traffic[DS_COL_NAME] \
            .diff() \
            .gt(self.min_time_between_intervals) \
            .cumsum()

        merged = filtered_traffic.merge(right=filtered_tasks, on=DS_COL_NAME)
        merged[MAX_TASKS_COL_NAME] = merged.groupby([INTERVAL_COL_NAME])[N_TASKS_COL_NAME].transform(max)
        merged[MAX_TASKS_TS_COL_NAME] = merged.loc[merged[N_TASKS_COL_NAME] == merged[MAX_TASKS_COL_NAME], DS_COL_NAME]

        aggregated = merged \
            .groupby(INTERVAL_COL_NAME) \
            .agg({DS_COL_NAME: ["min", "max"],
                  N_TASKS_COL_NAME: ["max"],
                  MAX_TASKS_TS_COL_NAME: ["max"],
                  **{sport: ["sum"] for sport in self.sports}
                  })

        aggregated.columns = aggregated.columns.to_flat_index()
        renamed: pd.DataFrame = aggregated.rename(
            columns={(DS_COL_NAME, "min"): START_DS_COL_NAME,
                     (DS_COL_NAME, "max"): END_DS_COL_NAME,
                     (N_TASKS_COL_NAME, "max"): N_TASKS_COL_NAME,
                     (MAX_TASKS_TS_COL_NAME, "max"): MAX_TASKS_TS_COL_NAME,
                     **{(sport, "sum"): sport for sport in self.sports}
                     })

        return renamed \
            .apply(func=self.calculate_sport_weights_in_extra_traffic, axis=1, args=(aggregated_calendar,)) \
            .loc[(renamed[START_DS_COL_NAME] != self.prediction_period_start) &
                 (renamed[END_DS_COL_NAME] != self.prediction_period_end)] \
            .drop(self.sports, axis=1)


class NRTTaskPreparation(CommonTaskPreparation):
    columns_to_merge_service_info: List[str] = [SERVICE_COL_NAME, REGION_COL_NAME]

    def __init__(self,
                 process_date: datetime,
                 bucket: str,
                 aiops_service: str,
                 aws_service: str,
                 region: str,
                 services_info_path: str,
                 task_scaling_recommendations_folder_pattern: str,
                 start_ds: datetime,
                 end_ds: datetime,
                 n_tasks: int,
                 ):
        # TODO: sports_list and calendar_pattern are not needed, should be changed in the future
        super().__init__(process_date=process_date, bucket=bucket, services_info_path=services_info_path,
                         sports_list=[], calendar_pattern='')
        self.aiops_service = aiops_service
        self.aws_service = aws_service
        self.region = region
        self.task_scaling_recommendations_folder_pattern = task_scaling_recommendations_folder_pattern \
            .replace("//", "/")
        self.start_ds = start_ds
        self.end_ds = end_ds
        self.n_tasks = n_tasks

    @staticmethod
    def generate_id(recommendations: pd.DataFrame) -> pd.DataFrame:
        with_start_date = recommendations \
            .assign(**{START_DATE_COL_NAME: recommendations[START_DS_COL_NAME].dt.strftime(DATE_PATTERN)})

        with_start_date[INTERVAL_ID_COL_NAME] = with_start_date[START_DATE_COL_NAME] + "_" + \
                                                with_start_date[AIOPS_SERVICE_COL_NAME] + "_" + \
                                                with_start_date[REGION_COL_NAME] + "_" + \
                                                "nrt_generated"

        return with_start_date.drop(START_DATE_COL_NAME, axis=1)

    def get_nrt_recommendation_df(self) -> pd.DataFrame:
        df = pd.DataFrame({
            'start_ds': [self.start_ds],
            'end_ds': [self.end_ds],
            'n_tasks': [str(self.n_tasks)],
            MAX_TASKS_TS_COL_NAME: [self.start_ds],
            'sport_weight_in_extra_traffic': [{}]
        })
        return df

    def prepare_task_recommendations(self):
        main_df: pd.DataFrame = self.get_nrt_recommendation_df()
        with_service_region: pd.DataFrame = main_df.assign(**{REGION_COL_NAME: self.region,
                                                              SERVICE_COL_NAME: self.aws_service,
                                                              AIOPS_SERVICE_COL_NAME: self.aiops_service})
        with_service_info: pd.DataFrame = self.add_service_info(with_service_region)
        with_metadata: pd.DataFrame = self.add_meta_info_column(recommendations=with_service_info,
                                                                prediction_date=self.process_date + timedelta(days=1))
        with_interval_id: pd.DataFrame = self.generate_id(with_metadata)

        return with_interval_id


class NRTForecast:

    def __init__(self,
                 polyfit_degree: int,
                 weights_tau: float,
                 input_data_col: str,
                 horizon_str: str = "2h"):
        self.polyfit_degree = polyfit_degree
        self.weights_tau = weights_tau
        self.input_data_col = input_data_col
        self.horizon = pd.to_timedelta(horizon_str)
        self.horizon_points = int(self.horizon / TRAFFIC_GRANULARITY_TIMEDELTA)
        self.lamb = None

    def delete_anomalies(self, data):
        logger.info("Deleting anomalies")
        data = data.asfreq(TRAFFIC_GRANULARITY)

        stl = STL(data[self.input_data_col],
                  period=ONE_DAY_POINTS)
        result = stl.fit()

        _, _, resid = result.seasonal, result.trend, result.resid

        resid_mu = resid.mean()
        resid_dev = resid.std()

        lower = resid_mu - 4 * resid_dev
        upper = resid_mu + 4 * resid_dev

        anomalies = data[self.input_data_col][(resid < lower) | (resid > upper)]
        mask = data.index.isin(anomalies.index)
        data[f'{self.input_data_col}_cleaned'] = data[self.input_data_col]
        data.loc[mask, f'{self.input_data_col}_cleaned'] = None

        return data

    def transform(self,
                  df: pd.DataFrame,
                  window_size: int = ONE_HOUR_POINTS) -> pd.DataFrame:
        logger.info("Preparing data for forecast")
        data = df.copy()

        data.index = data[DATETIME_COL_NAME].dt.tz_localize(None)
        data.drop('datetime', axis=1, inplace=True)

        df_cleaned = self.delete_anomalies(data)

        df_cleaned[f'{self.input_data_col}_interp'] = df_cleaned[f'{self.input_data_col}_cleaned'].interpolate()
        df_cleaned[f'{self.input_data_col}_interp'] = df_cleaned[f'{self.input_data_col}_interp'].clip(lower=1)

        df_cleaned[f'{self.input_data_col}_smoothed_max'] = df_cleaned[f'{self.input_data_col}_interp'].rolling(
            window=window_size, min_periods=1, center=True).max()

        df_cleaned[f'{self.input_data_col}_box_cox'], self.lamb = boxcox(
            df_cleaned[f'{self.input_data_col}_smoothed_max'])

        df_cleaned = df_cleaned.dropna(subset=[f'{self.input_data_col}_box_cox'])

        return df_cleaned

    def seasonality_forecast(self, seasonal_component, lags):

        model = AutoReg(seasonal_component, lags=lags, trend='c', seasonal=False, old_names=False)
        model_fit = model.fit()

        predictions = model_fit.predict(start=len(seasonal_component),
                                        end=len(seasonal_component) + self.horizon_points,
                                        dynamic=False)

        return predictions

    def prepare_trend_features(self, trend):
        train_idx = len(trend)

        X = [i for i in range(0, train_idx + self.horizon_points)]
        y = trend.values
        X_train, _ = X[:train_idx], X[train_idx:]

        coef = np.polyfit(X_train, y, self.polyfit_degree)
        curve = np.polyval(coef, X)

        return curve

    @staticmethod
    def calculate_weights(X, tau=0.02):

        weights = np.ones([len(X)])
        for i in range(len(X)):
            diff = X[-1] - X[i]
            weights[i] = np.exp(np.power(diff, 2) / (-2 * tau ** 2))
        return weights

    def trend_forecast(self, trend, degree):
        # model trend with a polynomial model

        X = self.prepare_trend_features(trend)
        y = trend.values
        split_index = len(X) - self.horizon_points
        X_train, X_test = X[:split_index], X[split_index:]

        if self.weights_tau == -1:
            weights = None
        else:
            weights = NRTForecast.calculate_weights(X_train, self.weights_tau)

        coef = np.polyfit(X_train, y, degree, w=weights)
        curve = np.polyval(coef, X_test)

        max_idx = trend.index.max()
        start_fcst_date = max_idx + TRAFFIC_GRANULARITY_TIMEDELTA
        end_fcst_date = max_idx + self.horizon
        indeces = pd.date_range(start_fcst_date, end_fcst_date, freq=TRAFFIC_GRANULARITY)
        trend_forecast = pd.DataFrame(curve, index=indeces, columns=['trend_forecast'])
        trend_forecast.index.name = 'datetime'

        return trend_forecast

    def residuals_forecast(self, residuals, span=12):
        # Model parameters
        alpha = 2 / (span + 1)

        # resid_smoothed = residuals.ewm(span=24, adjust=False).mean()

        model = SimpleExpSmoothing(residuals)
        results = model.fit(smoothing_level=alpha, optimized=False)
        residuals1_predictions = results.forecast(steps=self.horizon_points)

        return residuals1_predictions

    def compose_forecast(self, predictions_s1, predictions_s7, trend_fcst, residuals1_predictions,
                         residuals2_predictions):

        s1 = pd.DataFrame(predictions_s1, columns=['s1'])
        s1.index.name = 'datetime'
        s7 = pd.DataFrame(predictions_s7, columns=['s7'])
        s7.index.name = 'datetime'
        seasonal_df = s1.merge(s7, on='datetime', how='inner')

        composed = seasonal_df.merge(trend_fcst['trend_forecast'], on='datetime', how='inner')

        composed['preds_sum'] = composed['s1'] + composed['s7'] + composed['trend_forecast']

        composed['preds_sum_resid1'] = composed['preds_sum'] + residuals1_predictions.values
        composed['preds_sum_resid2'] = composed['preds_sum_resid1'] + residuals2_predictions.values
        composed['preds_inv'] = inv_boxcox(composed.preds_sum_resid2, self.lamb)

        n_nans_preds = composed['preds_sum_resid2'].isnull().sum()
        n_nans_inv_preds = composed['preds_inv'].isnull().sum()
        if n_nans_preds == 0 and n_nans_inv_preds > 0:
            sys.exit(
                "The invert boxcox transformation caused NaNs in NRT forecast, and this forecast will not be used for scaling recommendation adjustment.")

        return composed

    def make_forecast(self,
                      input_data: pd.DataFrame) -> pd.DataFrame:
        logger.info("Making forecast for throughput")
        train_data: pd.DataFrame = self.transform(input_data)

        result_bc = seasonal_decompose(x=train_data[f'{self.input_data_col}_box_cox'],
                                       model='additive',
                                       period=ONE_DAY_POINTS,
                                       extrapolate_trend='freq')
        result_trend_decomp = seasonal_decompose(x=result_bc.trend.dropna(),
                                                 model='additive',
                                                 period=ONE_WEEK_POINTS,
                                                 extrapolate_trend='freq')

        seasonal_component = result_bc.seasonal.dropna()
        trend_seasonal_component = result_trend_decomp.seasonal.dropna()
        trend = result_trend_decomp.trend.dropna()
        residuals1 = result_bc.resid.dropna()
        residuals2 = result_trend_decomp.resid.dropna()

        predictions_s1 = self.seasonality_forecast(seasonal_component, lags=ONE_DAY_POINTS)
        predictions_s7 = self.seasonality_forecast(trend_seasonal_component, lags=ONE_WEEK_POINTS)

        trend_fcst = self.trend_forecast(trend, degree=2)

        residuals1_predictions = self.residuals_forecast(residuals1, span=12)
        residuals2_predictions = self.residuals_forecast(residuals2, span=12)

        forecast = self.compose_forecast(
            predictions_s1,
            predictions_s7,
            trend_fcst,
            residuals1_predictions,
            residuals2_predictions
        )

        return forecast[["preds_inv"]].rename(columns={"preds_inv": COUNT_COL_NAME})


class ThroughputIngester:
    def __init__(self,
                 prediction_datetime: datetime,
                 metric_name: str,
                 query: str,
                 data_bucket: str,
                 clean_metric_location: str,
                 extra_loaded_metric_location: str,
                 nrt_forecast_location: str,
                 api_key: str,
                 app_key: str,
                 service: str,
                 region: str,
                 history_days_limit: int,
                 max_days_for_extra_loading: int = 3
                 ):
        self.query = query
        self.data_bucket = data_bucket
        self.history_days_limit = history_days_limit
        self.cleansed_metric_path = f"{clean_metric_location}/{service}/{region}/{metric_name}/"
        self.extra_loaded_metrics_path = f"{extra_loaded_metric_location}/{service}/{region}/{metric_name}/"
        self.nrt_forecast_path = f"{nrt_forecast_location}/{service}/{region}/{metric_name}/"
        self.prediction_datetime = prediction_datetime
        self.max_days_for_extra_loading = max_days_for_extra_loading
        initialize(api_key=api_key, app_key=app_key)

    def retrieve_data_from_datadog_api(self,
                                       left_edge: datetime,
                                       right_edge: datetime) -> Dict:
        logger.info("Retrieving data from Datadog:\n"
                    f"Query: {self.query}\n"
                    f"{left_edge} - {right_edge}")
        results = api.Metric.query(
            start=left_edge.timestamp(),
            end=right_edge.timestamp(),
            query=self.query
        )

        if results.get("status") == "error":
            raise Exception(f"Query failed with the following response {results.get('error')}")
        return results

    def load_extra_throughput(self,
                              left_edge: datetime,
                              right_edge: datetime) -> pd.DataFrame:
        logger.info(f"Getting data from Datadog: {left_edge} - {right_edge}")
        records: List[Dict] = []

        cur_left = left_edge
        while cur_left < right_edge:
            next_left = cur_left + timedelta(days=1)
            cur_right = min(next_left, right_edge)
            response: Dict = self.retrieve_data_from_datadog_api(cur_left, cur_right)
            records.extend(response["series"][0]["pointlist"])
            cur_left = next_left

        extra = pd.DataFrame.from_records(records, columns=[DATETIME_COL_NAME, COUNT_COL_NAME])
        extra[DATETIME_COL_NAME] = pd \
            .to_datetime(extra[DATETIME_COL_NAME] / 1000, unit="s", utc=True) \
            .dt.tz_convert(None)
        return extra

    def put_extra_loaded_throughput_to_s3(self,
                                          throughput: pd.DataFrame) -> None:
        left_edge = throughput[DATETIME_COL_NAME].min()
        right_edge = throughput[DATETIME_COL_NAME].max()

        file_name = f"{left_edge.strftime(DATETIME_PATTERN)}_{right_edge.strftime(DATETIME_PATTERN)}.csv"

        throughput.to_csv(f"./{file_name}", index=False)
        output_path = f"s3://{self.data_bucket}/{self.extra_loaded_metrics_path}{left_edge.strftime(DATE_PATTERN)}/"

        logger.info(f"Putting extra loaded file to s3")
        upload_to_s3(file_name, output_path)

    def get_present_throughput(self) -> pd.DataFrame:
        logger.info(f"Looking for last {self.history_days_limit} days throughput "
                    f"data in '{self.data_bucket}/{self.cleansed_metric_path}'")
        files: List[str] = s3.glob(f"{self.data_bucket}/{self.cleansed_metric_path}*/cleansed.csv")
        last = max(file for file in files if file.split("/")[-2] <= self.prediction_datetime.strftime(DATE_PATTERN))
        throughput = pd.read_csv("s3://" + last)
        throughput[DATETIME_COL_NAME] = pd.to_datetime(throughput[DATETIME_COL_NAME]).dt.tz_convert(None)
        return throughput[
            throughput[DATETIME_COL_NAME] >= self.prediction_datetime - timedelta(days=self.history_days_limit)]

    def get_extra_loaded_earlier_throughput(self,
                                            left_edge: datetime,
                                            right_edge: datetime) -> Optional[pd.DataFrame]:
        days_to_check = [(right_edge - timedelta(days=i)).strftime(DATE_PATTERN)
                         for i in range(self.max_days_for_extra_loading * 2 + 1)]

        logger.info(f"Checking {days_to_check} days for previously extra loaded throughput "
                    f"between '{left_edge}' and '{right_edge}' "
                    f"in '{self.data_bucket}/{self.extra_loaded_metrics_path}'")
        all_files = [file
                     for day in days_to_check
                     for file in s3.ls(f"{self.data_bucket}/{self.extra_loaded_metrics_path}{day}/")
                     if day >= (left_edge - timedelta(days=self.max_days_for_extra_loading)).strftime(DATE_PATTERN)
                     and s3.isfile(file)]
        parsed_filenames: List[Tuple[str, datetime, datetime]] = [self.parse_extra_loaded_filename(file)
                                                                  for file in all_files]

        filtered_dataframes = [pd.read_csv("s3://" + f) for (f, left, right) in parsed_filenames
                               if ((left_edge <= right <= right_edge) or
                                   (left_edge <= left <= right_edge))]
        if len(filtered_dataframes) > 0:
            concat = pd.concat(filtered_dataframes)
            concat[DATETIME_COL_NAME] = pd.to_datetime(concat[DATETIME_COL_NAME])
            res = concat[(left_edge < concat[DATETIME_COL_NAME]) & (concat[DATETIME_COL_NAME] <= right_edge)] \
                .drop_duplicates(subset=DATETIME_COL_NAME)
            logger.info(f"Found {len(res)} records in previously loaded extra throughput")
            return res
        else:
            logger.info("Didn't find any extra loaded throughput")
            return None

    @staticmethod
    def parse_extra_loaded_filename(path: str) -> Tuple[str, datetime, datetime]:
        file_name_parts = path.split("/")[-1].replace(".csv", "").split("_")
        return path, datetime.strptime(file_name_parts[0], DATETIME_PATTERN), datetime.strptime(file_name_parts[1],
                                                                                                DATETIME_PATTERN)

    def get_latest_throughput(self) -> pd.DataFrame:
        latest_data: List[pd.DataFrame] = []
        logger.info(f"PREDICTION_DATETIME: {self.prediction_datetime}")

        present_throughput: pd.DataFrame = self.get_present_throughput()
        logger.info(f"Present throughput:\n" + str(present_throughput))
        latest_data.append(present_throughput)
        max_present_throughput = present_throughput[DATETIME_COL_NAME].max()
        if (self.prediction_datetime - max_present_throughput) > timedelta(days=self.max_days_for_extra_loading):
            raise Exception(f"Last data date '{max_present_throughput}' is more than {self.max_days_for_extra_loading} "
                            f"days earlier than prediction datetime '{self.prediction_datetime}'!")

        left_edge = max_present_throughput
        extra_loaded_earlier_throughput: Optional[pd.DataFrame] = self.get_extra_loaded_earlier_throughput(
            left_edge=left_edge.to_pydatetime(),
            right_edge=self.prediction_datetime)
        if extra_loaded_earlier_throughput is not None:
            logger.info(f"Extra loaded earlier throughput:\n" + str(extra_loaded_earlier_throughput))
            latest_data.append(extra_loaded_earlier_throughput)
            left_edge = extra_loaded_earlier_throughput[DATETIME_COL_NAME].max()

        left_edge_to_load: datetime = left_edge.to_pydatetime() + timedelta(minutes=5)
        if left_edge_to_load < self.prediction_datetime:
            extra_loaded: pd.DataFrame = self.load_extra_throughput(
                left_edge=left_edge_to_load,
                right_edge=self.prediction_datetime)
            logger.info(f"Extra loaded now throughput:\n" + str(extra_loaded))
            self.put_extra_loaded_throughput_to_s3(extra_loaded)
            latest_data.append(extra_loaded)

        return pd.concat(latest_data)

    def upload_nrt_forecast_to_s3(self,
                                  forecast: pd.DataFrame):
        filename = f"{self.prediction_datetime.strftime(DATETIME_PATTERN)}.csv"
        forecast.to_csv(f"./{filename}")
        output_path = f"s3://{self.data_bucket}/{self.nrt_forecast_path}{self.prediction_datetime.strftime(DATE_PATTERN)}"
        logger.info(f"Uploading NRT forecast to s3")
        upload_to_s3(filename, output_path)


@dataclass
class ScalingRecommendation:
    start_ds: Timestamp
    end_ds: Timestamp
    interval_id: str
    n_tasks: int
    max_tasks_timestamp: Timestamp
    aiops_service: str
    service: str
    region: str
    account_id: str
    cluster_name: str
    metadata: Dict
    sport_weight_in_extra_traffic: Dict
    original_start_ds: Optional[Timestamp] = None
    original_end_ds: Optional[Timestamp] = None
    original_n_tasks: Optional[int] = None

    @classmethod
    def from_dict(cls,
                  data: Dict):
        return cls(start_ds=pd.to_datetime(data[START_DS_COL_NAME]).tz_localize(None),
                   end_ds=pd.to_datetime(data[END_DS_COL_NAME]).tz_localize(None),
                   interval_id=data[INTERVAL_ID_COL_NAME],
                   n_tasks=data[N_TASKS_COL_NAME],
                   max_tasks_timestamp=pd.to_datetime(data[MAX_TASKS_TS_COL_NAME]).tz_localize(None),
                   service=data["service"],
                   aiops_service=data["aiops_service"],
                   region=data["region"],
                   account_id=data["account_id"],
                   cluster_name=data["cluster_name"],
                   metadata=data["metadata"],
                   sport_weight_in_extra_traffic=data["sport_weight_in_extra_traffic"],
                   original_start_ds=cls.parse_optional_timestamp(data.get("original_start_ds", None)),
                   original_end_ds=cls.parse_optional_timestamp(data.get("original_end_ds", None)),
                   original_n_tasks=data.get("original_n_tasks", None))

    @staticmethod
    def parse_optional_timestamp(str_ts: str):
        if str_ts:
            return pd.to_datetime(str_ts).tz_localize(None)
        else:
            return None

    @staticmethod
    def add_prediction_date_to_metadata(existing_metadata: Dict,
                                        prediction_date: datetime) -> Dict:
        update_original = {} if ORIGINAL_PREDICTION_DATE_COL_NAME in existing_metadata else {
            ORIGINAL_PREDICTION_DATE_COL_NAME: existing_metadata[PREDICTION_DATE_COL_NAME]}

        return {**existing_metadata,
                PREDICTION_DATE_COL_NAME: prediction_date,
                **update_original}

    def is_intersected(self,
                       left_edge: Timestamp,
                       right_edge: Timestamp) -> bool:
        return (self.start_ds <= right_edge) and (self.end_ds >= left_edge)

    def is_close_after(self,
                       right_edge: Timestamp,
                       max_interval_between: timedelta) -> bool:
        return timedelta() <= (self.start_ds - right_edge) <= max_interval_between

    def to_dict_for_publishing(self) -> Dict:
        d = self.__dict__
        dt_fields = [START_DS_COL_NAME, END_DS_COL_NAME, "original_start_ds", "original_end_ds", MAX_TASKS_TS_COL_NAME]
        for field in dt_fields:
            if d.get(field):
                d[field] = d[field].strftime(DATETIME_PATTERN_FOR_ES)
        return d


# TODO: shared code with task_scaling_recommendations_loader.py
class CommonTaskScalingRecommendationLoader:
    recommendation_type: str
    start_date_source_col_name: str = "start_ds"

    def __init__(self,
                 index_prefix: str,
                 timeseries_index_prefix: str,
                 environment: str,
                 es_hosts: Optional[List[str]] = None
                 ):
        if es_hosts is None:
            es_hosts = ['vpc-aiop-sandbox-tr5aggpi5eesn7vbqmjlspcdmi.us-east-1.es.amazonaws.com']
        self.index_prefix = index_prefix
        self.timeseries_index_prefix = timeseries_index_prefix
        self.common_props = {
            ENVIRONMENT_COL_NAME: environment
        }
        self.es = Elasticsearch(hosts=es_hosts, scheme="https",
                                port=443, use_ssl=True, verify_certs=True,
                                connection_class=RequestsHttpConnection, timeout=30,
                                max_retries=10, retry_on_timeout=True)

    def read_recommendations(self,
                             process_date: datetime) -> pd.DataFrame:
        raise NotImplementedError()

    def prepare_recommendations(self,
                                recommendations: pd.DataFrame) -> pd.DataFrame:
        recommendations = recommendations[recommendations['account_id'].notnull()]
        recommendations["account_id"] = recommendations["account_id"].astype(str).str.split(".").str[0]

        recommendations[START_DATE_COL_NAME] = pd.to_datetime(
            recommendations[self.start_date_source_col_name]).dt.strftime(DATE_PATTERN)
        recommendations[START_DS_COL_NAME] = pd.to_datetime(recommendations[START_DS_COL_NAME]).dt.strftime(
            DATETIME_PATTERN_FOR_ES)
        recommendations[END_DS_COL_NAME] = pd.to_datetime(recommendations[END_DS_COL_NAME]).dt.strftime(
            DATETIME_PATTERN_FOR_ES)
        recommendations[MAX_TASKS_TS_COL_NAME] = pd.to_datetime(recommendations[MAX_TASKS_TS_COL_NAME]).dt.strftime(
            DATETIME_PATTERN_FOR_ES)
        return recommendations.assign(**self.common_props)

    def publish_to_index(self,
                         index_name: str,
                         recommendations: List[Dict]):
        logger.info(f"Publishing {len(recommendations)} records to es '{index_name}'")
        success_number, errors = helpers.bulk(client=self.es, actions=recommendations, index=index_name,
                                              chunk_size=100, max_retries=5, initial_backoff=10)
        if errors:
            raise Exception("During load following errors occurred:\n" + "\n".join(errors))

    def publish_to_plain_index(self,
                               recommendations: pd.DataFrame,
                               year_month: str):
        index_name = f"{self.index_prefix}-{year_month}"
        records: List[Dict] = [{**x,
                                "_id": f"{x[INTERVAL_ID_COL_NAME]}_{self.recommendation_type}_{x[ENVIRONMENT_COL_NAME]}"}
                               for x in recommendations.to_dict(orient='records')]
        self.publish_to_index(index_name, records)

    def clean_time_series_index(self,
                                update_intervals: List[str]):
        index_pattern = f"{self.timeseries_index_prefix}-*-*"
        delete_query = {
            "bool": {
                "filter": [
                    {
                        "terms": {INTERVAL_ID_COL_NAME: update_intervals}
                    },
                    {
                        "term": {RECOMMENDATION_TYPE_COL_NAME: self.recommendation_type}
                    }
                ]
            }
        }
        self.es.delete_by_query(index=index_pattern,
                                body={"query": delete_query})

    def publish_to_time_series_index(self,
                                     recommendations: pd.DataFrame,
                                     year_month: str):
        index_name = f"{self.timeseries_index_prefix}-{year_month}"

        update_intervals: List[str] = list(recommendations[INTERVAL_ID_COL_NAME].unique())
        without_events = recommendations \
            .rename(columns={N_TASKS_COL_NAME: "count"}) \
            .drop(labels="sport_weight_in_extra_traffic", axis=1)

        without_events[TIMESTAMP_COL_NAME] = without_events \
            .apply(lambda row: pd.date_range(row[START_DS_COL_NAME], row['end_ds'], freq='5T'), axis=1)

        records: List[Dict] = without_events \
            .explode(column=TIMESTAMP_COL_NAME, ignore_index=True) \
            .to_dict(orient='records')

        records_with_ids: List[Dict] = [
            {**x,
             TIMESTAMP_COL_NAME: x[TIMESTAMP_COL_NAME].strftime(DATETIME_PATTERN_FOR_ES),
             "_id": f"{x[INTERVAL_ID_COL_NAME]}_{self.recommendation_type}_{x[TIMESTAMP_COL_NAME].strftime(DATETIME_PATTERN)}_{x[ENVIRONMENT_COL_NAME]}",
             RECOMMENDATION_TYPE_COL_NAME: self.recommendation_type}
            for x in records]

        self.clean_time_series_index(update_intervals)
        self.publish_to_index(index_name, records_with_ids)

    @staticmethod
    def divide_by_year_months(recommendations: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        unique_dates = recommendations[START_DATE_COL_NAME].unique()
        unique_year_months = {d.rsplit("-", 1)[0]
                              for d in unique_dates}
        return {ym: recommendations[recommendations[START_DATE_COL_NAME].str.startswith(ym)]
                for ym in unique_year_months}

    def publish_recommendations(self,
                                process_date: datetime):
        recommendations: pd.DataFrame = self.read_recommendations(process_date)
        prepared_recommendations: pd.DataFrame = self.prepare_recommendations(recommendations)

        for year_month, recs in self.divide_by_year_months(prepared_recommendations).items():
            logger.info(f"Updating year-month {year_month}")
            self.publish_to_plain_index(recs, year_month)
            self.publish_to_time_series_index(recs, year_month)


class ElasticPublisher(CommonTaskScalingRecommendationLoader):
    nrt_forecast_index_prefix = "aiop-v2-nrt-trafficforecast"
    recommendation_type: str = "adjusted"

    def __init__(self,
                 processing_date: datetime,
                 service_name: str,
                 region: str,
                 environment: str,
                 es_hosts: Optional[List[str]] = None):
        super().__init__(index_prefix="aiops-adjusted-task-scaling-recommendations",
                         timeseries_index_prefix="aiops-timeseries-task-scaling-recommendations",
                         environment=environment,
                         es_hosts=es_hosts)
        self.processing_date = processing_date
        self.service_name = service_name
        self.region = region
        self.environment = environment

    @staticmethod
    def add_id(d: Dict):
        d['_id'] = d[TIMESTAMP_COL_NAME] + d[SERIES_START_COL_NAME] + d[ENVIRONMENT_COL_NAME] + d[REGION_COL_NAME] + \
                   d[SERVICE_COL_NAME]
        return d

    def publish_nrt_forecast(self,
                             data: pd.DataFrame):
        index_name: str = f"{self.nrt_forecast_index_prefix}-{self.processing_date.strftime(DATE_PATTERN_WITHOUT_DAY)}"

        logger.info(f"Uploading forecast to es '{index_name}'")

        data[ENVIRONMENT_COL_NAME] = self.environment
        data[REGION_COL_NAME] = self.region
        data[SERVICE_COL_NAME] = self.service_name
        data[TIMESTAMP_COL_NAME] = data.index.strftime(DATETIME_PATTERN_FOR_ES).astype(str)
        data[SERIES_START_COL_NAME] = data[TIMESTAMP_COL_NAME].min()
        records = [self.add_id(x) for x in data.to_dict("records")]

        success_number, errors = helpers.bulk(self.es, records, index=index_name)

        if errors:
            raise Exception("During load following errors occurred:\n" + "\n".join(errors))

    def publish_adjusted_scaling_recommendation(self,
                                                recommendation: ScalingRecommendation):
        record: Dict = recommendation.to_dict_for_publishing()
        recommendations: pd.DataFrame = self.prepare_recommendations(pd.DataFrame.from_records([record]))

        for year_month, recs in self.divide_by_year_months(recommendations).items():
            logger.info(f"Updating year-month {year_month}")
            self.publish_to_plain_index(recs, year_month)
            self.publish_to_time_series_index(recs, year_month)


class RecommendationAdjuster:

    def __init__(self,
                 prediction_datetime: datetime,
                 data_bucket: str,
                 aggregated_predictions_location: str,
                 task_predictions_location: str,
                 adjusted_task_predictions_location: str,
                 task_scaling_recommendations_folder_pattern: str,
                 adjusted_task_scaling_recommendations_folder_pattern: str,
                 service: str,
                 region: str,
                 means_diff_percent_threshold_up: int = 25,
                 means_diff_percent_threshold_down: int = -30,
                 change_up_threshold: float = 0.3,
                 change_down_threshold: float = -0.4,
                 slope_threshold: int = 5,
                 decision_making_period_length: str = "30m"):
        self.data_bucket = data_bucket
        self.prediction_datetime = prediction_datetime
        self.means_diff_percent_threshold_up = means_diff_percent_threshold_up
        self.means_diff_percent_threshold_down = means_diff_percent_threshold_down
        self.change_up_threshold = change_up_threshold
        self.change_down_threshold = change_down_threshold
        self.slope_threshold = slope_threshold
        self.service = service
        self.region = region
        self.aggregated_predictions_path = f"{aggregated_predictions_location}/{service}/{region}/"
        self.tasks_predictions_path = f"{task_predictions_location}/{service}/{region}/"
        self.nrt_tasks_prediction_path = f"{adjusted_task_predictions_location}/service={service}/region={region}/"
        self.scaling_recommendations_path_pattern = task_scaling_recommendations_folder_pattern
        self.adjusted_scaling_recommendations_path_pattern = adjusted_task_scaling_recommendations_folder_pattern
        self.datetime_after_decision_making_period = prediction_datetime + pd.to_timedelta(
            decision_making_period_length)

    def get_throughput_predictions(self,
                                   left_edge: datetime,
                                   right_edge: datetime) -> pd.DataFrame:
        files: List[str] = s3.glob(f"{self.data_bucket}/{self.aggregated_predictions_path}*/predictions_full.csv")
        last = max(file for file in files if file.split("/")[-2] <= self.prediction_datetime.strftime(DATE_PATTERN))
        logger.info(f"Loading last found version of throughput prediction from '{last}'")
        predictions = pd.read_csv("s3://" + last)
        predictions[DATETIME_COL_NAME] = pd.to_datetime(predictions[DS_COL_NAME])
        predictions[COUNT_COL_NAME] = predictions["total"]

        return predictions[[COUNT_COL_NAME, DATETIME_COL_NAME]][
            (predictions[DATETIME_COL_NAME] >= left_edge) & (predictions[DATETIME_COL_NAME] <= right_edge)]

    def get_tasks_predictions(self,
                              left_edge: datetime,
                              right_edge: datetime) -> pd.DataFrame:
        files: List[str] = s3.glob(f"{self.data_bucket}/{self.tasks_predictions_path}*/tasks_predictions.csv")
        last = max(file for file in files if file.split("/")[-2] <= self.prediction_datetime.strftime(DATE_PATTERN))
        logger.info(f"Loading last found version of task predictions from '{last}'")

        predictions = pd.read_csv("s3://" + last)
        predictions[DATETIME_COL_NAME] = pd.to_datetime(predictions[DS_COL_NAME])

        return predictions[[N_TASKS_COL_NAME, DATETIME_COL_NAME]][
            (predictions[DATETIME_COL_NAME] >= left_edge) & (predictions[DATETIME_COL_NAME] <= right_edge)]

    def get_last_scaling_recommendations(self,
                                         relative_path: str) -> List[ScalingRecommendation]:
        formatted_folder = relative_path.format(
            region=self.region,
            service=self.service,
            date="*"
        )
        glob_path = f"{self.data_bucket}/{formatted_folder}/*.json"
        files: List[str] = s3.glob(glob_path)
        # last = max((file for file in files), default=None)
        three_days_before_pred_date = (self.prediction_datetime - pd.Timedelta('3d')).strftime(DATE_PATTERN)
        last = max((file for file in files if three_days_before_pred_date < file.split("/")[-4].split('=')[
            1] <= self.prediction_datetime.strftime(DATE_PATTERN)), default=None)
        if last:
            logger.info(f"Loading last found original scaling recommendations from '{last}'")
            recommendations: pd.DataFrame = pd.read_json("s3://" + last)
            return [ScalingRecommendation.from_dict(d) for d in recommendations.to_dict(orient='records')]
        else:
            logger.info(f"There is no original scaling recommendations in {glob_path}")
            return []

    def put_adjusted_scaling_recommendation_to_s3(self,
                                                  recommendation: ScalingRecommendation):
        filename = f"{self.prediction_datetime.strftime(DATETIME_PATTERN)}.json"
        df = pd.DataFrame.from_records([recommendation.__dict__])
        df.to_json(path_or_buf=f"./{filename}",
                   orient='records',
                   date_format='iso')

        logger.info("Uploading adjusted scaling recommendation to s3")
        formatted_folder = self.adjusted_scaling_recommendations_path_pattern.format(
            region=self.region,
            service=self.service,
            date=self.prediction_datetime.strftime(DATE_PATTERN)
        )
        output_path = f"s3://{self.data_bucket}/{formatted_folder}/"
        upload_to_s3(filename, output_path)

    def put_adjusted_tasks_prediction_to_s3(self,
                                            task_prediction: pd.DataFrame):
        filename = f"{self.prediction_datetime.strftime(DATETIME_PATTERN)}.csv"
        task_prediction.to_csv(f"./{filename}")
        output_path = f"s3://{self.data_bucket}/{self.nrt_tasks_prediction_path}{self.prediction_datetime.strftime(DATE_PATTERN)}/"
        logger.info(f"Uploading adjusted tasks prediction to s3")
        upload_to_s3(filename, output_path)

    def get_latest_adjusted_tasks_predictions(self) -> Optional[pd.DataFrame]:
        glob_path = f"{self.data_bucket}/{self.nrt_tasks_prediction_path}*/*.csv"
        files: List[str] = s3.glob(glob_path)
        last = max((file for file in files if file.split("/")[-2] <= self.prediction_datetime.strftime(DATE_PATTERN)),
                   default=None)
        if last:
            logger.info(f"Loading last found version of nrt adjusted task predictions from '{last}'")
            tasks_prediction = pd.read_csv("s3://" + last)
            tasks_prediction.index = pd.to_datetime(tasks_prediction[DATETIME_COL_NAME])
            tasks_prediction.drop(DATETIME_COL_NAME, axis=1, inplace=True)
            return tasks_prediction
        else:
            logger.info(f"There is no previously nrt adjusted tasks prediction in {glob_path}")
            return None

    @staticmethod
    def get_means_diff_percent(prediction: pd.DataFrame,
                               nrt_forecast: pd.DataFrame) -> int:
        max_prediction = prediction[COUNT_COL_NAME].max()
        max_nrt = nrt_forecast[COUNT_COL_NAME].max()
        diff = round(100 * (max_nrt - max_prediction) / max_prediction)
        return diff

    @staticmethod
    def trend_slope_in_degrees(nrt_forecast: pd.DataFrame) -> int:
        y1 = nrt_forecast.loc[nrt_forecast.first_valid_index(), COUNT_COL_NAME]
        y2 = nrt_forecast.loc[nrt_forecast.last_valid_index(), COUNT_COL_NAME]
        y1_hat = y1 / y2
        y2_hat = 1
        radians = (y2_hat - y1_hat)
        degrees = math.degrees(radians)
        return degrees

    @staticmethod
    def recommendation_artifacts(recommendation: ScalingRecommendation):
        if recommendation is not None:
            rec_dict = {
                'start_ds': recommendation.start_ds,
                'end_ds': recommendation.end_ds,
                'n_tasks': recommendation.n_tasks,
                'orig_start_ds': recommendation.original_start_ds,
                'orig_end_ds': recommendation.original_end_ds,
                'orig_n_tasks': recommendation.original_n_tasks,
                'max_tasks_timestamp': recommendation.max_tasks_timestamp
            }
            return rec_dict
        else:
            return None

    def get_adjusted_scaling_recommendation(self,
                                            nrt_forecast: pd.DataFrame,
                                            latest_throughput: pd.DataFrame) -> Optional[ScalingRecommendation]:
        cut_nrt_forecast = nrt_forecast[nrt_forecast.index >= self.datetime_after_decision_making_period]
        adjusting_time = cut_nrt_forecast.index.min()  # diff timestamp
        forecast_end_time = cut_nrt_forecast.index.max()  # cut_off_forecast_date

        latest_throughput.index = latest_throughput[DATETIME_COL_NAME]
        latest_throughput = latest_throughput.sort_index()
        traffic_prediction_extended: pd.DataFrame = self.get_throughput_predictions(
            left_edge=adjusting_time,
            right_edge=forecast_end_time)
        traffic_prediction = traffic_prediction_extended[
            traffic_prediction_extended[DATETIME_COL_NAME] >= adjusting_time]
        task_prediction: pd.DataFrame = self.get_tasks_predictions(
            left_edge=adjusting_time,
            right_edge=forecast_end_time)

        traffic_prediction_extended.index = traffic_prediction_extended[DATETIME_COL_NAME]
        task_prediction.index = task_prediction[DATETIME_COL_NAME]

        last_recommendations: List[ScalingRecommendation] = self.get_last_scaling_recommendations(
            self.scaling_recommendations_path_pattern)
        try:
            last_adjusted_recommendations: List[ScalingRecommendation] = self.get_last_scaling_recommendations(
                self.adjusted_scaling_recommendations_path_pattern)
        except FileNotFoundError:
            logger.info('last_adjusted_recommendations were not found')
            last_adjusted_recommendations = []

        # nrt_adjusted_tasks_prediction = self.get_latest_adjusted_tasks_predictions()

        logger.info(f"Comparing NRT forecast and traffic prediction on interval {adjusting_time} - {forecast_end_time}")
        means_diff_percent: int = self.get_means_diff_percent(traffic_prediction, cut_nrt_forecast)

        affected_recommendation = next(
            (rec for rec in last_recommendations
             if rec.is_intersected(left_edge=adjusting_time, right_edge=forecast_end_time)
             or rec.is_close_after(right_edge=forecast_end_time, max_interval_between=timedelta(hours=15))), None)
        adjusted_affected_recommendation = next(
            (rec for rec in last_adjusted_recommendations
             if rec.is_intersected(left_edge=adjusting_time, right_edge=forecast_end_time)), None)

        n_tasks_max = task_prediction.loc[adjusting_time: forecast_end_time, N_TASKS_COL_NAME].max()
        logger.info(f'Max of predicted n_tasks for nrt forecast period: {n_tasks_max}')

        if adjusted_affected_recommendation:
            new_n_tasks = adjusted_affected_recommendation.n_tasks
            logger.info('Previous n_tasks is from adjusted_affected_recommendation')
            max_tasks_end = adjusted_affected_recommendation.max_tasks_timestamp
            logger.info(f'Max_tasks end: {max_tasks_end}')
        elif affected_recommendation:
            new_n_tasks = affected_recommendation.n_tasks
            logger.info('Previous n_tasks is from affected_recommendation')
            max_tasks_end = affected_recommendation.max_tasks_timestamp
            logger.info(f'Max_tasks end: {max_tasks_end}')
        else:
            #             new_n_tasks = task_prediction.loc[adjusting_time, N_TASKS_COL_NAME]
            new_n_tasks = n_tasks_max
            logger.info('Previous n_tasks is from task_prediction')
            max_tasks_end = adjusting_time  # to not stop adjustment if there is no recommendation

        # Find previous recommendation start 
        start_tasks_change_timestamp = adjusting_time
        if affected_recommendation is not None:
            recommendation_start_time = affected_recommendation.start_ds
            one_hour_before_recommendation_start = recommendation_start_time - ONE_HOUR_TIMEDELTA

            if (adjusting_time >= one_hour_before_recommendation_start) & (adjusting_time < recommendation_start_time):
                logger.info(
                    'A scaling event is expected within 1 hour. Therefore, adjusting if needed will be postponed till start of scaling event')
                start_tasks_change_timestamp = recommendation_start_time

        # Calculate nrt_forecast trend slope
        slope = self.trend_slope_in_degrees(cut_nrt_forecast)
        logger.info(f'NRT forecast slope: {round(slope)}')

        new_adjusted_recommendations: Optional[ScalingRecommendation] = None

        if (means_diff_percent > self.means_diff_percent_threshold_up) | (
                means_diff_percent < self.means_diff_percent_threshold_down):
            logger.info(
                f"Means diff {means_diff_percent}% is higher than threshold {self.means_diff_percent_threshold_up}% "
                f"or lower than threshold {self.means_diff_percent_threshold_down}%, "
                f"need to adjust recommendations")

            previous_new_n_tasks = new_n_tasks
            logger.info(f'Previous new n tasks: {previous_new_n_tasks}')
            new_n_tasks = round(n_tasks_max * (100 + means_diff_percent) / 100)
            logger.info(f'Adjusted n_tasks by diff: {new_n_tasks}')
            n_tasks_change = (new_n_tasks - previous_new_n_tasks) / previous_new_n_tasks
            logger.info(f'N_tasks change %: {round(n_tasks_change * 100)}')

            slope_pos_small_change_less_than_th_up = slope < self.slope_threshold and 0 <= n_tasks_change < self.change_up_threshold
            slope_neg_small_change_less_than_th_down = adjuster.change_down_threshold < n_tasks_change <= 0
            max_tasks_not_reached_but_want_downscale = adjusting_time < max_tasks_end and means_diff_percent < adjuster.means_diff_percent_threshold_down

            if slope_pos_small_change_less_than_th_up or slope_neg_small_change_less_than_th_down or max_tasks_not_reached_but_want_downscale:
                logger.info(
                    "New n_tasks will not be changed because the tasks change is insignificant in comparison with previous adjustment")
                new_n_tasks = previous_new_n_tasks

            else:
                if adjusted_affected_recommendation:
                    logger.info(f"Found intersecting already adjusted recommendation, "
                                f"adjusting it again \n{self.recommendation_artifacts(adjusted_affected_recommendation)}")
                    new_adjusted_recommendations = dataclasses.replace(
                        adjusted_affected_recommendation,
                        start_ds=min(start_tasks_change_timestamp,
                                     adjusted_affected_recommendation.original_start_ds),
                        end_ds=max(adjusted_affected_recommendation.end_ds,
                                   adjusted_affected_recommendation.original_end_ds,
                                   forecast_end_time),
                        n_tasks=new_n_tasks,
                        metadata=ScalingRecommendation.add_prediction_date_to_metadata(
                            existing_metadata=adjusted_affected_recommendation.metadata,
                            prediction_date=self.prediction_datetime
                        )
                    )
                elif affected_recommendation:
                    logger.info(f"There is not intersecting adjusted recommendation, "
                                f"but close original recommendation were found, "
                                f"adjusting it \n{self.recommendation_artifacts(affected_recommendation)}")
                    new_adjusted_recommendations = dataclasses.replace(
                        affected_recommendation,
                        start_ds=min(start_tasks_change_timestamp,
                                     affected_recommendation.start_ds),
                        end_ds=max(affected_recommendation.end_ds,
                                   forecast_end_time),
                        n_tasks=new_n_tasks,
                        original_start_ds=affected_recommendation.start_ds,
                        original_end_ds=affected_recommendation.end_ds,
                        original_n_tasks=affected_recommendation.n_tasks,
                        metadata=ScalingRecommendation.add_prediction_date_to_metadata(
                            existing_metadata=affected_recommendation.metadata,
                            prediction_date=self.prediction_datetime
                        )
                    )

                else:
                    msg = "There is no recommendation for nearest 15 hours"
                    logger.info(msg)
                    if new_n_tasks <= n_tasks_max:
                        logger.info(
                            f'NRT forecast suggests decreasing the number of tasks between scaling recommendations')
                    else:
                        logger.info(f"There is the traffic increase that wasn't explained by events, "
                                    f"new scaling recommendation will be created based on NRT forecast")
                        task_preparation = NRTTaskPreparation(
                            process_date=datetime.strptime(parsed.prediction_datetime, DATETIME_PATTERN),
                            bucket=parsed.data_bucket,
                            aiops_service=parsed.service_name,
                            aws_service=parsed.aws_service,
                            region=parsed.region,
                            services_info_path=parsed.services_info_path,
                            task_scaling_recommendations_folder_pattern=parsed.adjusted_task_scaling_recommendations_folder_pattern,
                            start_ds=adjusting_time,
                            end_ds=forecast_end_time,
                            n_tasks=new_n_tasks
                        )

                        nrt_task_recommendation: pd.DataFrame = task_preparation.prepare_task_recommendations()
                        new_adjusted_recommendations = ScalingRecommendation.from_dict(nrt_task_recommendation.to_dict(orient='records')[0])
                    

        else:
            logger.info(
                f"Means diff {means_diff_percent}% is higher than threshold {self.means_diff_percent_threshold_down}% "
                f"and lower than threshold {self.means_diff_percent_threshold_up}%, "
                f"everything is fine")

        logger.info(f"Previous tasks prediction dataset: \n\t{task_prediction}")

        logger.info(f"Adjusted affected: \n\t{self.recommendation_artifacts(adjusted_affected_recommendation)}")
        logger.info(f"Affected: \n\t{self.recommendation_artifacts(affected_recommendation)}")

        return new_adjusted_recommendations


if __name__ == '__main__':
    parsed = parse_arguments()
    prediction_datetime_arg: datetime = datetime.strptime(parsed.prediction_datetime, DATETIME_PATTERN)

    ingester = ThroughputIngester(
        prediction_datetime=prediction_datetime_arg,
        query=parsed.query,
        metric_name=parsed.metric_name,
        data_bucket=parsed.data_bucket,
        clean_metric_location=parsed.clean_metric_location,
        extra_loaded_metric_location=parsed.extra_loaded_metric_location,
        nrt_forecast_location=parsed.nrt_forecast_location,
        api_key=parsed.datadog_api_key,
        app_key=parsed.datadog_app_key,
        service=parsed.service_name,
        region=parsed.region,
        history_days_limit=parsed.history_days_limit
    )
    forecaster = NRTForecast(
        polyfit_degree=parsed.polyfit_degree,
        weights_tau=parsed.weights_tau,
        horizon_str=parsed.horizon,
        input_data_col=COUNT_COL_NAME
    )
    publisher = ElasticPublisher(
        processing_date=prediction_datetime_arg,
        service_name=parsed.service_name,
        region=parsed.region,
        environment=parsed.environment,
        es_hosts=parsed.es_hosts
    )
    adjuster = RecommendationAdjuster(
        prediction_datetime=prediction_datetime_arg,
        data_bucket=parsed.data_bucket,
        aggregated_predictions_location=parsed.aggregated_predictions_location,
        task_predictions_location=parsed.task_predictions_location,
        adjusted_task_predictions_location=parsed.adjusted_task_predictions_location,
        task_scaling_recommendations_folder_pattern=parsed.task_scaling_recommendations_folder_pattern,
        adjusted_task_scaling_recommendations_folder_pattern=parsed.adjusted_task_scaling_recommendations_folder_pattern,
        service=parsed.service_name,
        region=parsed.region,
        means_diff_percent_threshold_up=parsed.means_diff_percent_threshold_up,
        means_diff_percent_threshold_down=parsed.means_diff_percent_threshold_down,
        change_up_threshold=parsed.change_up_threshold,
        change_down_threshold=parsed.change_down_threshold,
        slope_threshold=parsed.slope_threshold,
    )

    latest_throughput: pd.DataFrame = ingester.get_latest_throughput()

    traffic_nrt_forecast: pd.DataFrame = forecaster.make_forecast(input_data=latest_throughput)
    ingester.upload_nrt_forecast_to_s3(traffic_nrt_forecast)
    publisher.publish_nrt_forecast(traffic_nrt_forecast)

    adjusted_scaling_recommendation: Optional[ScalingRecommendation] = adjuster.get_adjusted_scaling_recommendation(
        traffic_nrt_forecast, latest_throughput)

    if adjusted_scaling_recommendation:
        logger.info(f"There is new adjusted scaling recommendation: \n\t{adjusted_scaling_recommendation}")
        adjuster.put_adjusted_scaling_recommendation_to_s3(adjusted_scaling_recommendation)
        publisher.publish_adjusted_scaling_recommendation(adjusted_scaling_recommendation)
