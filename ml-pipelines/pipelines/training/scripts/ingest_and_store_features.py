"""Data preprocessing - feature generation"""
import argparse
from typing import List
import glob
import json
import logging
import os
import re
import sys
import unicodedata
from datetime import datetime, timedelta
from time import sleep

import boto3
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import sagemaker
from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.session import Session

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

SEED = 42
RANK_TYPE = 'league_rank'

minutes_scale = list(range(0, 61, 5))

NUM_WEEKS = 1

COMPETITIONS_COL_NAME = "competitions"
DATE_UTC_COL_NAME = "date_utc"

data_types_dict = {
    'float64': 'Fractional',
    'int64': 'Integral',
    'string': 'String'
}

def combine_from_parquets(base_dir, data_dir, fields=None, year='*',
                          sport='*', league='*', client=None):
    paths = os.path.join(base_dir, data_dir, f'year={year}',
                         f'sport_partition={sport}',
                         f'league_partition={league}',
                         '*parquet')
    # logger.info(paths)
    all_parquet_files = glob.glob(paths)
    # logger.info(all_parquet_files)
    df = pq.ParquetDataset(all_parquet_files).read_pandas().to_pandas().sort_values(by='starttime')
    if client is not None:
        df = df[df['client_app'] == client]
    df['sport_league'] = df['sport'] + '/' + df['league']
    df['year'] = year

    if fields is not None:
        return df[fields]
    else:
        return df


def combine_from_csv(base_dir, data_dir, subdir_pattern='*/*.csv', search_latest=False, converters={}, dtype=None):
    files = glob.glob(os.path.join(base_dir, data_dir, subdir_pattern))
    if len(files) == 0:
        print("No csv files found")
        return None
    # logger.info(f"Files found: {files}")
    if search_latest:
        files.sort(key=lambda x: os.stat(x).st_mtime, reverse=True)
        files = [files[0]]
    print(f"Combining dataframe from: {files}")
    data = pd.concat([pd.read_csv(f, converters=converters, dtype=dtype) for f in files], ignore_index=True)
    return data


def extract_time_parts(data, ds_name='starttime', drop_ds=False):
    data['year'] = pd.DatetimeIndex(data[ds_name]).year
    data['month'] = pd.DatetimeIndex(data[ds_name]).month
    data['day'] = pd.DatetimeIndex(data[ds_name]).day
    data['hour'] = pd.DatetimeIndex(data[ds_name]).hour
    data['minute'] = pd.DatetimeIndex(data[ds_name]).minute
    if drop_ds:
        data.drop(ds_name, axis=1, inplace=True)
    return data


def prepare_event_traffic(all_watch_traffic, sport_name):
    event_watch_traffic = all_watch_traffic[(all_watch_traffic['sport'] == sport_name)][
        ['id', 'starttime', 'count']].rename(columns={'id': 'event_id', 'count': 'event_count'})
    all_watch_traffic_during_event = pd.merge(event_watch_traffic, all_watch_traffic, how='inner',
                                              on=['starttime']).drop(['sport', 'league'], axis=1).rename(
        columns={'id': 'coevent_id', 'count': 'coevent_count'})
    event_traffic_vs_coevent = all_watch_traffic_during_event.groupby(['event_id', 'starttime', 'event_count']).agg(
        {'coevent_count': 'sum'}).reset_index()
    event_traffic_vs_coevent = extract_time_parts(event_traffic_vs_coevent)
    return event_traffic_vs_coevent


def calculate_influence(event):
    return event['event_count'] / event['coevent_count']


def get_to_service_scale(event):
    return event['service_count'] * event['event_influence']


def get_to_service_extra_traffic_scale(event):
    return max([(event['service_count'] - event['baseline_count']), 0]) * event['event_influence']


def prepare_service_traffic_vs_events_traffic(service_traffic, service_baseline, event_traffic_vs_coevent):
    service_traffic_vs_baseline = pd.merge(service_traffic,
                                           service_baseline,
                                           how='inner', on=['year', 'month', 'day', 'hour', 'minute'])
    service_traffic_vs_events_traffic = pd.merge(event_traffic_vs_coevent,
                                                 service_traffic_vs_baseline,
                                                 how='inner',
                                                 on=['year', 'month', 'day', 'hour', 'minute'])
    service_traffic_vs_events_traffic = service_traffic_vs_events_traffic.rename(columns={'starttime': 'event_time'})

    with pd.option_context('display.max_rows', 10, 'display.max_columns', None):  # more options can be specified also
        print(service_traffic_vs_events_traffic)

    service_traffic_vs_events_traffic['event_influence'] = service_traffic_vs_events_traffic.apply(calculate_influence,
                                                                                                   axis=1)
    service_traffic_vs_events_traffic['event_count_on_service_scale'] = service_traffic_vs_events_traffic.apply(
        get_to_service_scale, axis=1)
    service_traffic_vs_events_traffic[
        'event_count_on_service_extra_traffic_scale'] = service_traffic_vs_events_traffic.apply(
        get_to_service_extra_traffic_scale, axis=1)
    service_traffic_vs_events_traffic['event_count_on_service_extra_traffic_scale_smoothing_avg'] = \
        service_traffic_vs_events_traffic['event_count_on_service_extra_traffic_scale'].rolling(window=6).mean()
    return service_traffic_vs_events_traffic


def extract_date_features(data, add_day=True, add_hour=True, date_name='ds'):
    if add_day:
        data['day'] = data[date_name].dt.dayofweek
    if add_hour:
        data['hour'] = data[date_name].dt.hour
    return data


def extract_competition_features(cur_sport_calendar: pd.DataFrame, convert_to_int=True):
    exploded = cur_sport_calendar \
        .explode(COMPETITIONS_COL_NAME) \
        .reset_index(drop=True)

    with_extracted_columns = exploded \
        .join(pd.json_normalize(exploded[COMPETITIONS_COL_NAME], max_level=0))

    with_extracted_columns['competition_cl'] = with_extracted_columns[COMPETITIONS_COL_NAME]

    if convert_to_int:
        with_extracted_columns['division_competition'] = with_extracted_columns['division_competition'].astype(int)
        with_extracted_columns['conference_competition'] = with_extracted_columns['conference_competition'].astype(int)
        with_extracted_columns['on_watch_espn'] = with_extracted_columns['on_watch_espn'].astype(int)

    return with_extracted_columns.drop(COMPETITIONS_COL_NAME, axis=1)


def encode_competitors(df, competitor_columns, competitors=None, prefix='team_'):
    df.fillna('Unknown_competitor', inplace=True)
    if competitors:
        competitor_list = competitors
    else:
        competitor_list = pd.unique(df[competitor_columns].values.ravel('K'))
    print(f"Number of competitors: {len(competitor_list)}")
    indicator_cols = []
    for comp in competitor_list:
        indicator_col = prefix + comp
        df[indicator_col] = 0
        for col in competitor_columns:
            df[indicator_col] += (df[col] == comp).astype(int)
        indicator_cols.append(indicator_col)
    return indicator_cols


def extract_venue(cur_sport_calendar):
    cur_sport_calendar["venue"] = cur_sport_calendar["venue"].fillna(value="venue_unknown")
    return cur_sport_calendar


def generate_media_features(data):
    data['media'].fillna('media_unknown', inplace=True)
    data['media'] = data['media'].apply(lambda r: r.replace('|', ',').split(','))
    data['media'] = data['media'].apply(tuple)
    return data


def normalize_string(string):
    out_string = unicodedata.normalize("NFD", string)
    out_string = re.sub("[\u0300-\u036f]", "", out_string)
    return out_string


def clear_by_threshold(data, target_col, timestamp_col, threshold):
    if threshold <= 0:
        return data
    peak_points = data.loc[data[target_col] >= threshold, timestamp_col]
    print(peak_points)
    for peak_point in peak_points:
        # Could be simplified with indices
        # Provided that data is sorted by timestamp_col
        print(peak_point)
        sel_data_at_peak = data[timestamp_col] == peak_point
        sel_data_before_peak = data[timestamp_col] < peak_point
        data.loc[sel_data_at_peak, target_col] = data.loc[sel_data_before_peak, target_col].median()
    return data


def find_target_count(df):
    df['target'] = df.where(df['service_count'] == df['service_count'].max())[TARGET_METRIC]
    return df


def prepare_model_data(service_traffic_vs_events_traffic, cur_sport_calendar,
                       TARGET_METRIC, TIMESTAMP_COL=DATE_UTC_COL_NAME, prediction_date=None,
                       predict_horizon=None, airings=None,
                       filter_by_airing=False, agg_fun='max', threshold=3E4):
    cur_sport_calendar_exploded_by_filler = cur_sport_calendar.explode('filler')
    service_traffic_vs_events_traffic_agg = pd.merge(service_traffic_vs_events_traffic,
                                                     cur_sport_calendar_exploded_by_filler, how='inner',
                                                     left_on=['event_id', 'event_time'], right_on=['id', 'filler'])
    
    # service_traffic_vs_events_traffic_agg = service_traffic_vs_events_traffic_agg.groupby(['id']).apply(lambda x: find_target_count(x))
    # service_traffic_vs_events_traffic_agg = service_traffic_vs_events_traffic_agg.groupby(['id']).agg(
    #     {'target': agg_fun, 'service_count': agg_fun}).rename(columns={'target':TARGET_METRIC}).reset_index()

    service_traffic_vs_events_traffic_agg = service_traffic_vs_events_traffic_agg.groupby(['id']).agg(
        {TARGET_METRIC: agg_fun, 'service_count': agg_fun}).reset_index()
    
    service_traffic_vs_events_traffic_agg = pd.merge(service_traffic_vs_events_traffic_agg, cur_sport_calendar, on=['id'])
    service_traffic_vs_events_traffic_agg.sort_values(TIMESTAMP_COL, inplace=True)
    service_traffic_vs_events_traffic_agg['id'] = service_traffic_vs_events_traffic_agg['id'].astype(int)
    service_traffic_vs_events_traffic_agg = clear_by_threshold(service_traffic_vs_events_traffic_agg,
                                                               TARGET_METRIC, TIMESTAMP_COL,
                                                               threshold)
    if predict_horizon is not None and prediction_date is not None:
        logger.info(cur_sport_calendar.shape)
        max_event_date_in_horizon = prediction_date + pd.Timedelta(days=predict_horizon)
        logger.info('Max date for predicting taking into account the horizon %s' % max_event_date_in_horizon)
        future_events = cur_sport_calendar[(cur_sport_calendar[DATE_UTC_COL_NAME] >= prediction_date) & (
                cur_sport_calendar[DATE_UTC_COL_NAME] < max_event_date_in_horizon)]
        logger.info('Max date in future events %s' % future_events[DATE_UTC_COL_NAME].max())
        if filter_by_airing:
            print(f'future_events before filter by airings: {future_events.shape}')
            airings['game_id'] = airings.game_id.astype(float).astype(int).astype(str)
            future_events = future_events[future_events['id'].isin(airings['game_id'].astype('str').values)]
            print(f'future_events after filter by airings: {future_events.shape}')

        future_events[TARGET_METRIC] = -1
        future_events['service_count'] = -1
        future_events.sort_values(DATE_UTC_COL_NAME, inplace=True)
        future_events['id'] = future_events['id'].astype(int)
        return service_traffic_vs_events_traffic_agg, future_events
    return service_traffic_vs_events_traffic_agg


def select_model_features(data, features, TARGET_METRIC, date_col=DATE_UTC_COL_NAME):
    model_data = data[features + ['id']]
    if TARGET_METRIC in data:
        model_data['y'] = data[TARGET_METRIC]
    else:
        model_data['y'] = np.nan
    try:
        target_column = model_data.pop('y')
        model_data.insert(0, 'y', target_column)
    except Exception as e:
        logger.error("Error rearranging target column")
        logger.exception(e)
    model_data['ds'] = data[date_col]
    return model_data


def quantize(parsed_date):
    if parsed_date == 0:
        return None

    if parsed_date.second >= 30:
        parsed_date = parsed_date + timedelta(minutes=1)

    minutes = min(minutes_scale, key=lambda x: x - parsed_date.minute if x - parsed_date.minute > 0 else 61)
    if minutes == 60:
        parsed_date = parsed_date + timedelta(hours=1)
        parsed_date = parsed_date.replace(minute=0)
    else:
        parsed_date = parsed_date.replace(minute=minutes)

    parsed_date = parsed_date.replace(second=0)

    return parsed_date


def filler(start, end):
    delta = end - start
    for i in range(-300, delta.seconds + 601, 300):
        filler = start + timedelta(seconds=i)

        # if (filler > start) & (filler < end):
        yield filler


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_input_filename(input_path, extention='csv'):
    input_filenames = [f for f in os.listdir(input_path)
                       if os.path.isfile(os.path.join(input_path, f)) and f.endswith(extention)]
    if len(input_filenames) == 0:
        logger.error(f"No {extention} files found in {input_path}")
        return None
    elif len(input_filenames) > 1:
        logger.warning(f"Expected only one {extention} file in {input_path}, found many")
        logger.warning(f"Will return latest {extention} filename in {input_path}")
        input_filenames.sort(key=lambda x: os.stat(os.path.join(input_path, x)).st_mtime)
    input_filename = input_filenames[0]
    logger.info(f"Returning {input_filename} in input path: {input_path}")
    return input_filename


def ingest_traffic_data(input_path, date_col="datetime", error_msg="Error ingesting data", history_depth=-1,
                        rename_dict=None):
    try:
        input_filename = get_input_filename(input_path)
        traffic_data = pd.read_csv(os.path.join(input_path, input_filename))
    except Exception as e:
        logger.error(error_msg)
        logger.exception(e)
    if 'ingest_date' in traffic_data.columns:
        traffic_data.drop(columns=['ingest_date'], inplace=True)
    traffic_data[date_col] = pd.to_datetime(traffic_data[date_col], utc=True)
    min_traffic_date = traffic_data[date_col].min()
    max_traffic_date = traffic_data[date_col].max()
    if history_depth > 0:
        logger.info(f"Will use at most {history_depth} last days of history for training")
        min_traffic_date = max_traffic_date - pd.Timedelta(days=history_depth)
        sel_traffic = (traffic_data[date_col] >= min_traffic_date) & (traffic_data[date_col] <= max_traffic_date)
        traffic_data = traffic_data[sel_traffic]
    logger.info(f"Min traffic date: {min_traffic_date}")
    logger.info(f"Max traffic date: {max_traffic_date}")
    traffic_data = extract_time_parts(traffic_data, date_col, drop_ds=True)
    if rename_dict:
        traffic_data.rename(columns=rename_dict, inplace=True, errors='ignore')
    return traffic_data


def cast_object_to_string(data_frame):
    for label in data_frame.columns:
        if data_frame.dtypes[label] == 'object':
            data_frame[label] = data_frame[label].astype("str").astype("string")


def prepare_feature_data(data, prediction_date, event_time, service, region, event_time_feature_name='EventTime', data_type='training'):
    feature_data = data.copy()
    feature_data[event_time_feature_name] = event_time
    # feature_data[event_time_feature_name] = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
    feature_data["prediction_date"] = str(pd.to_datetime(prediction_date).date())
    feature_data["data_type"] = data_type
    feature_data["service"] = service
    feature_data["region"] = region
    if 'ds' in feature_data:
        feature_data['ds'] = feature_data['ds'].astype(str)
    cast_object_to_string(feature_data)
    print(feature_data.info())
    return feature_data


def wait_for_feature_group_update_complete(feature_group):
    status = feature_group.describe().get("LastUpdateStatus").get('Status')
    while status == "InProgress":
        print("Waiting for Feature Group Update")
        sleep(5)
        status = feature_group.describe().get("LastUpdateStatus").get('Status')
    if status != "Successful":
        raise RuntimeError(f"Failed to add new feature to feature group {feature_group.name}")
    print(f"FeatureGroup {feature_group.name} successfully updated.")


def wait_for_feature_group_creation_complete(feature_group):
    status = feature_group.describe().get("FeatureGroupStatus")
    while status == "Creating":
        print("Waiting for Feature Group Creation")
        sleep(5)
        status = feature_group.describe().get("FeatureGroupStatus")
    if status != "Created":
        raise RuntimeError(f"Failed to create feature group {feature_group.name}")
    print(f"FeatureGroup {feature_group.name} successfully created.")


def create_or_update_feature_group(sm_region, bucket_name, feature_group_name, feature_data,
                         feature_store_prefix, feature_group_description,
                         record_identifier_feature_name='id',
                         event_time_feature_name='EventTime'):
    boto_session = boto3.Session(region_name=sm_region)
    sagemaker_client = boto_session.client(service_name='sagemaker', region_name=sm_region)
    feature_store_runtime = boto_session.client(service_name='sagemaker-featurestore-runtime',
                                                region_name=sm_region)
    feature_store_session = Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_featurestore_runtime_client=feature_store_runtime,
        default_bucket=bucket_name
    )
    role = sagemaker.session.get_execution_role(feature_store_session)
    fgs = sagemaker_client.list_feature_groups(MaxResults=100)
    fg_names = [fg['FeatureGroupName'] for fg in fgs['FeatureGroupSummaries']]

    feature_group = FeatureGroup(name=feature_group_name,
                                 sagemaker_session=feature_store_session)
    if feature_group_name not in fg_names:
        print(f"Global feature group {feature_group_name} is not found. Will try to create it")
        feature_group.load_feature_definitions(data_frame=feature_data)
        feature_group.create(
            description=feature_group_description,
            s3_uri=f"s3://{bucket_name}/{feature_store_prefix}",
            record_identifier_name=record_identifier_feature_name,
            event_time_feature_name=event_time_feature_name,
            role_arn=role,
            enable_online_store=False
        )
        wait_for_feature_group_creation_complete(feature_group=feature_group)
    else:
        print(f"Feature group {feature_group_name} already exists")
        fg_metadata = sagemaker_client.search(
            Resource="FeatureMetadata",
            SearchExpression={'Filters': [
                    {
                        'Name': 'FeatureGroupName',
                        'Operator': 'Equals',
                        'Value': feature_group_name
                    }
        ]})

        existing_features = []
        for fm in fg_metadata.get('Results'):
            existing_features.append(fm.get('FeatureMetadata').get('FeatureName'))

        features_to_add = feature_data.columns[~feature_data.columns.isin(existing_features)]
        logger.info(f'Features to add: {features_to_add}')
        if len(features_to_add) > 0:
            for ftr in features_to_add:
                logger.info(f'Adding feature {ftr} to feature_group {feature_group_name}')
                sagemaker_client.update_feature_group(
                FeatureGroupName=feature_group_name,
                FeatureAdditions=[
                    {"FeatureName": ftr, "FeatureType": data_types_dict.get(str(feature_data[ftr].dtype))},
                    ]
                )  
                wait_for_feature_group_update_complete(feature_group=feature_group)
        else: 
            logger.info(f'All features already exist in feature_group {feature_group_name}')
    return feature_group


def export_features(feature_group_name, feature_data):
    try:
        feature_group.ingest(data_frame=feature_data, max_workers=1, wait=True)
        logger.info(f"Successfully exported features to Feature Group: {feature_group_name}")
    except Exception as e:
        logger.error(f"Failed to ingest data to Feature Group: {feature_group_name}")
        logger.exception(e)


def save_processed_data(data, output_dir, output_filename,
                        data_message="Saving output data to local output path", header=False, exclude_cols=None):
    output_data = data.copy()
    if (not isinstance(output_data, pd.DataFrame)) or output_data.empty:
        logger.error("Output data is not pandas dataframe or empty")
        return
    logger.info(data_message)
    logger.info(f"Data shape: {output_data.shape}")
    logger.info(output_data.head(2))
    try:
        output_path = os.path.join(base_dir, output_dir)
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

def filter_by_network_name(data, airings, exclude_network_name):
    data = data.merge(airings[['event_id', 'network_name']], left_on = 'id', right_on='event_id', how='left')
    data = data[~data.network_name.isin(exclude_network_name)]
    data = data.drop_duplicates(subset=['id', DATE_UTC_COL_NAME], keep="last")
    return data

def set_features_for_modeling(features=[], cols_with_order=[],
                                  binary_cols_with_order=[], scaler_features=[]):
        if args.use_league_features:
            features.append('league')
            cols_with_order.append('league')
        if args.use_media_features:
            features.append('media')
            binary_cols_with_order.append('media')
        if args.use_competition_features:
            features.extend(['division_competition', 'conference_competition'])
        if args.use_on_watch_espn:
            features.append('on_watch_espn')
        if args.use_date_features:
            features.append('day')
            cols_with_order.append('day')
            if args.encode_hour:
                features.append('hour_encoded')
                cols_with_order.append('hour_encoded')
            else:
                features.append('hour')
                cols_with_order.append('hour')
        if args.use_rank_features:
            features.extend(['rank_0', 'rank_1'])
            scaler_features.extend(['rank_0', 'rank_1'])
        if args.use_team_features:
            features.extend(team_feature_cols)
        if args.use_netbase_features:
            features.extend(['netbase_metric_impressions', 'netbase_metric_totalbuzz'])
            scaler_features.extend(['netbase_metric_impressions', 'netbase_metric_totalbuzz'])
        if args.use_n_tasks_feature:
            features.extend(['n_tasks'])
        logger.info(features)
        return features, cols_with_order, binary_cols_with_order, scaler_features

def read_netbase(path):
        return pd.concat(
            [pd.read_csv(metrics_path,
                        dtype={'id': 'object'})
            for metrics_path in sorted(glob.glob(f"{path}/*/*.csv"))]
            ).drop_duplicates(subset=['id'], keep='last')


if __name__ == "__main__":
    logger.info("Starting data ingestion and feature extraction and storage")
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket-name", type=str)
    parser.add_argument('--sm-region', type=str)
    parser.add_argument("--service", type=str)
    parser.add_argument("--region", type=str)
    parser.add_argument("--history-depth", type=int, default=-1)
    parser.add_argument("--prediction-date", type=str)
    parser.add_argument("--sport-name", type=str)
    parser.add_argument("--smoothed-extra-traffic", type=str2bool, nargs='?', default=False)
    parser.add_argument("--traffic-threshold", type=float, default=-1.0)
    parser.add_argument("--num-val-events", type=int)
    parser.add_argument("--min-sample-days", type=int)
    parser.add_argument("--use-league-features", type=str2bool, nargs='?', default=False)
    parser.add_argument("--use-media-features", type=str2bool, nargs='?', default=False)
    parser.add_argument("--use-competition-features", type=str2bool, nargs='?', default=False)
    parser.add_argument("--use-on-watch-espn", type=str2bool, nargs='?', default=False)
    parser.add_argument("--use-date-features", type=str2bool, nargs='?', default=False)
    parser.add_argument("--use-rank-features", type=str2bool, nargs='?', default=False)
    parser.add_argument("--use-team-features", type=str2bool, nargs='?', default=False)
    parser.add_argument("--use-netbase-features", type=str2bool, nargs='?', default=False)
    parser.add_argument("--use-n-tasks-feature", type=str2bool, nargs='?', default=False)
    parser.add_argument("--encode-hour", type=str2bool, nargs='?', default=False)
    parser.add_argument("--exclude-team-names", type=str, default='')
    parser.add_argument("--feature-group-name", type=str)
    parser.add_argument("--predict-horizon", type=int)
    parser.add_argument("--exclude-network-name", type=str)
    parser.add_argument("--exclude-dates-str", type=str)

    args = parser.parse_args()

    bucket_name = args.bucket_name
    history_depth = args.history_depth
    prediction_date = args.prediction_date
    sport_name = args.sport_name
    smoothed_extra_traffic = args.smoothed_extra_traffic
    feature_group_name = args.feature_group_name
    predict_horizon = args.predict_horizon
    exclude_network_name = json.loads(args.exclude_network_name)

    if feature_group_name:
        logger.info(f"Will use feature group: {feature_group_name}")
    else:
        logger.error("Error: feature group name not set")
        exit(1)
    exclude_team_names = args.exclude_team_names
    if exclude_team_names and exclude_team_names != 'none':
        exclude_team_names = [tn.strip() for tn in exclude_team_names.split(';')]
    else:
        exclude_team_names = []
    traffic_threshold = args.traffic_threshold
    num_val_events = args.num_val_events
    min_sample_days = args.min_sample_days
    logger.info(sport_name)

    if prediction_date is None:
        print("Prediction date is not defined")
        sys.exit(1)
    prediction_date = pd.to_datetime(prediction_date, utc=True)
    logger.info(f"Prediction date: {prediction_date}")

    base_dir = "/opt/ml/processing"
    input_traffic_dir = "input_traffic"
    input_baseline_dir = "input_baseline"

    input_airings_dir = "input_airings"
    input_calendar_dir = "input_calendar"

    input_watchgraph_dir = "input_watchgraph"
    input_netbase_dir = "input_netbase"

    input_nhl_rankings_dir = "input_nhl_rankings"
    input_mch_rankings_dir = "input_mch_rankings"

    scheduled_tasks_dir = "input_scheduled_tasks"
    scheduled_tasks_file = "scheduled_tasks_ts.csv"

    logger.info("Ingesting data from local path")

    try:
        logger.info("Ingesting service traffic data")
        input_traffic_path = os.path.join(base_dir, input_traffic_dir)
        service_traffic = ingest_traffic_data(input_traffic_path,
                                              date_col='datetime',
                                              error_msg="Error ingesting traffic data",
                                              history_depth=history_depth,
                                              rename_dict={'count': 'service_count'})
    except Exception as e:
        logger.exception(e)
    logger.info(service_traffic.head(2))

    try:
        logger.info("Ingesting baseline data")
        input_baseline_path = os.path.join(base_dir, input_baseline_dir)
        baseline_traffic = ingest_traffic_data(input_baseline_path,
                                               date_col='ds',
                                               error_msg="Error ingesting baseline data",
                                               history_depth=history_depth,
                                               rename_dict={'yhat': 'baseline_count'})
    except Exception as e:
        logger.exception(e)
    logger.info(baseline_traffic.head(2))

    logger.info("Ingesting airings data")
    airings = combine_from_csv(base_dir, input_airings_dir, subdir_pattern='*.csv')
    airings = airings[~airings['game_id'].isna()]
    airings['game_id'] = airings['game_id'].astype(str)
    airings.loc[airings['game_id'].str.contains('[', regex=False), 'game_id'] = None
    airings.loc[airings['game_id'] == 'nan', 'game_id'] = None
    airings = airings[~airings['game_id'].isna()]
    logger.info(airings.head(2))
    logger.info(airings.shape)

    logger.info("Ingesting event calendar data")
    calendar = combine_from_csv(base_dir, input_calendar_dir,
                                subdir_pattern='*.csv')
    calendar['id'] = calendar['id'].astype(str)
    calendar[DATE_UTC_COL_NAME] = pd.to_datetime(calendar[DATE_UTC_COL_NAME], utc=True)
    calendar['end_with_avg_duration'] = pd.to_datetime(calendar['end_with_avg_duration'], utc=True)
    if sport_name != 'hockey':
        calendar[COMPETITIONS_COL_NAME] = calendar[COMPETITIONS_COL_NAME].apply(json.loads)
        calendar = calendar[calendar['sport'] == sport_name]

    calendar['fight_approx_start_time_quantized'] = calendar[DATE_UTC_COL_NAME].apply(quantize)
    calendar['fight_approx_end_time_quantized'] = calendar['end_with_avg_duration'].apply(quantize)
    calendar['filler'] = calendar.apply(lambda x: list(filler(x['fight_approx_start_time_quantized'],
                                                              x['fight_approx_end_time_quantized'])), axis=1)
    calendar.sort_values(DATE_UTC_COL_NAME, inplace=True)
    logger.info(calendar.head(2))
    logger.info(calendar.shape)
    logger.info(f'Calendar min and max date: {calendar.date_utc.min()}, {calendar.date_utc.max()}')

    if args.use_netbase_features:
        logger.info("Ingesting netbase metrics")
        netbase = read_netbase(os.path.join(base_dir, input_netbase_dir))
        logger.info(netbase.shape)
        logger.info(netbase.head(2))
        netbase['id'] = netbase['id'].astype(str)
        netbase[DATE_UTC_COL_NAME] = pd.to_datetime(netbase[DATE_UTC_COL_NAME], utc=True)
        calendar = calendar.merge(netbase, on=['id', DATE_UTC_COL_NAME], how='left')

    logger.info("Ingesting watch graph data")
    key_fields = ['id', 'sport', 'league', 'starttime', 'count']
    all_watch_traffic = combine_from_parquets(base_dir, input_watchgraph_dir,
                                              fields=key_fields)  # TODO: read only one sport, change json to copy only required sport to container local system
    all_watch_traffic = all_watch_traffic.groupby(['id', 'sport',
                                                   'league', 'starttime']).agg({'count': 'sum'}).reset_index()
    all_watch_traffic['starttime'] = pd.to_datetime(all_watch_traffic['starttime'], utc=True)

    predictable_sports = ['football', 'basketball', 'baseball', 'hockey', 'mma']
    all_watch_traffic = all_watch_traffic[all_watch_traffic.sport.isin(predictable_sports)]
    logger.info(
        f"""Watchgraph min and max date: 
        {all_watch_traffic.query("sport==@sport_name").starttime.min()}, 
        {all_watch_traffic.query("sport==@sport_name").starttime.max()}""")
    
    event_traffic_vs_coevent = prepare_event_traffic(all_watch_traffic, sport_name)
    logger.info(event_traffic_vs_coevent.shape)

    service_traffic_vs_events_traffic = prepare_service_traffic_vs_events_traffic(service_traffic,
                                                                                  baseline_traffic,
                                                                                  event_traffic_vs_coevent)
    service_traffic_vs_events_traffic['event_id'] = service_traffic_vs_events_traffic['event_id'].astype(str)

    logger.info("Checking prediction date")
    # Sanity check for prediction date
    traffic_min_date = service_traffic_vs_events_traffic.event_time.min()
    calendar_max_date = calendar[DATE_UTC_COL_NAME].max()
    logger.info(f"Service traffic min date: {traffic_min_date}")
    logger.info(f"Service traffic max date: {service_traffic_vs_events_traffic.event_time.max()}")
    logger.info(f"Calendar max date: {calendar_max_date}")
    logger.info(f"Prediction date: {prediction_date}")
    if prediction_date < traffic_min_date + timedelta(days=min_sample_days):
        logger.error("Invalid prediction date")
        raise RuntimeError(
            f"Interval between prediction date {prediction_date} and minimum data date {traffic_min_date} "
            f"lower than min_sample_days {min_sample_days}")

    max_event_date_in_horizon = prediction_date + pd.Timedelta(days=predict_horizon)
    number_of_future_events = \
        calendar[(calendar[DATE_UTC_COL_NAME] >= prediction_date)
                 & (calendar[DATE_UTC_COL_NAME] < max_event_date_in_horizon)].shape[0]
    logger.info(f"Number of future events in horizon: {number_of_future_events}")
    if number_of_future_events == 0:
        logger.error("No events to predict. Exiting")
        raise RuntimeError(
            f"There are no events in the prediction horizon which starts from prediction date {prediction_date},"
            f"so there is no events to predict")

    if args.use_competition_features:
        calendar = extract_competition_features(calendar)

    team_cols = ['team_0_display_name', 'team_1_display_name']
    if exclude_team_names:
        logger.info(f"Will exclude team names: {exclude_team_names}")
        logger.debug(f"Calendar shape before exclusion: {calendar.shape}")
        for team_col in team_cols:
            calendar.drop(calendar[calendar[team_col].isin(exclude_team_names)].index, inplace=True)
        logger.debug(f"Calendar shape after exclusion: {calendar.shape}")

    # TODO: add --use-venue
    calendar = extract_venue(calendar)
    if args.use_date_features:
        calendar = extract_date_features(calendar, date_name=DATE_UTC_COL_NAME)

    if args.encode_hour:
        calendar['hour_encoded'] = 0
        calendar.loc[(calendar.hour >= 17) & (calendar.hour < 22), 'hour_encoded'] = 1
        calendar.loc[(calendar.hour >= 22) | (calendar.hour < 5), 'hour_encoded'] = 2

    if args.use_team_features:
        team_feature_cols = encode_competitors(calendar, team_cols)

    logger.info(f"Calendar dataframe has columns: {calendar.columns}")

    if smoothed_extra_traffic:
        logger.info("Using smoothed extra traffic as a target for model training")
        TARGET_METRIC = 'event_count_on_service_extra_traffic_scale_smoothing_avg'
    else:
        logger.info("Using unsmoothed extra traffic as a target for model training")
        TARGET_METRIC = 'event_count_on_service_extra_traffic_scale'
    TIMESTAMP_COL = DATE_UTC_COL_NAME

    logger.info(f"Will truncate traffic by threshold: {traffic_threshold}")
    model_data, future_events = prepare_model_data(service_traffic_vs_events_traffic, calendar,
                                                   TARGET_METRIC, TIMESTAMP_COL, prediction_date=prediction_date,
                                                   predict_horizon=predict_horizon, airings=airings,
                                                   threshold=traffic_threshold, filter_by_airing=False)


    # Add N_TASKS
    if args.use_n_tasks_feature:
        scheduled_tasks = pd.read_csv(
            os.path.join(base_dir, scheduled_tasks_dir, scheduled_tasks_file), 
            parse_dates=['datetime'], index_col=['datetime']).tz_localize('utc')

        scheduled_tasks = scheduled_tasks.rename(columns = {'count': 'n_tasks'})
        model_data = model_data.merge(scheduled_tasks['n_tasks'], left_on = 'date_utc', right_on = 'datetime', how = 'left')
        future_events = future_events.merge(scheduled_tasks['n_tasks'], left_on = 'date_utc', right_on = 'datetime', how = 'left')

    # DIFFERENT FILTERING
    # Exclude COVID19 pandemic period for hockey
    if sport_name == 'hockey':
        sel_model_data = (model_data[DATE_UTC_COL_NAME] <= "2020-03-13") | (model_data[DATE_UTC_COL_NAME] >= "2021-01-01")
        model_data = model_data[sel_model_data]

    
    # Specific dates exclusion
    exclude_dates = json.loads(args.exclude_dates_str)
    sel_exclusion = model_data.index.isna()
    for period in exclude_dates:
        if (len(period) == 2) and (period[0] <= period[1]):
            sel_exclusion = sel_exclusion | ((model_data[DATE_UTC_COL_NAME] >= period[0]) &\
                                                    (model_data[DATE_UTC_COL_NAME] <= period[1]))
            logger.info(f"Will exclude all original data in interval: {period[0]} - {period[1]}")
        else:
            logger.info(f"Something wrong with the period of exclusion")
    model_data = model_data[~sel_exclusion]

    logger.info('FILTER MODEL DATA BY NETWORK NAME')
    logger.info(f'model_data before filtering by network_name: {model_data.shape}')
    model_data = filter_by_network_name(model_data, airings, exclude_network_name)
    logger.info(f'model_data after filtering by network_name: {model_data.shape}')

    logger.info('FILTER FUTURE EVENTS BY NETWORK NAME')
    logger.info(f'future_events before filtering by network_name: {future_events.shape}')
    future_events = filter_by_network_name(future_events, airings, exclude_network_name)
    logger.info(f'future_events after filtering by network_name: {future_events.shape}')
    
    # cols_with_order = ['league', 'day', 'hour_encoded']
    # binary_cols_with_order = ['media']
    features, cols_with_order, binary_cols_with_order, scaler_features = set_features_for_modeling()
    if len(features) == 0:
        logger.error("No features set for modeling")
        sys.exit(1)
    else:
        logger.info("Features set successfully")
        logger.info('Features: %s' % features)
        logger.info('Cols_with_order: %s' % cols_with_order)
        logger.info('Binary_cols_with_order: %s' % binary_cols_with_order)
        logger.info('Scaler_features: %s' % scaler_features)

    model_data = generate_media_features(model_data)
    model_data = select_model_features(model_data, features, TARGET_METRIC)


    future_events = generate_media_features(future_events)
    future_events = select_model_features(future_events, features, TARGET_METRIC)

    # Feature Store Operations
    feature_store_prefix = "aiop-sport-feature-store"
    feature_group_description = f"features-unprocessed-{sport_name}"
    event_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
    logger.info(f'Feature group event time : {event_time}')

    feature_data = prepare_feature_data(model_data, prediction_date, event_time, data_type='training', service=args.service, region=args.region)
    logger.info(f"Creating feature group: {feature_group_name}")
    feature_group = create_or_update_feature_group(args.sm_region, bucket_name, feature_group_name, feature_data,
                                         feature_store_prefix, feature_group_description)

    export_features(feature_group_name, feature_data)
    feature_data = prepare_feature_data(future_events, prediction_date, event_time, data_type='prediction', service=args.service, region=args.region)
    export_features(feature_group_name, feature_data)

    logger.info(f'Calendar: {calendar.head(2)}')
    logger.info(f'Calendar media nunique {calendar.media.nunique()}')

    save_processed_data(calendar, "calendar", "calendar.csv",
                        data_message=f"Saving event calendar for {sport_name}", header=True)
