import argparse
import logging
import os
import sys
import boto3
import pandas as pd
import numpy as np
import pickle
from time import sleep
from urllib.parse import urlparse
from sagemaker.session import Session
from sagemaker.feature_store.feature_group import FeatureGroup, AthenaQuery
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

ATHENA_TIMEOUT = 60 * 15 # in seconds

TIMESTAMP_COL = 'date_utc'
TARGET_METRIC = 'event_count_on_service_extra_traffic_scale'
DATETIME_PATTERN = "%Y-%m-%d"


def run_query_in_workgroup(sagemaker_session: Session, query: AthenaQuery, workgroup: str, query_string: str, output_location: str, kms_key = None):
    catalog = query.catalog
    database = query.database

    kwargs = dict(
        QueryString=query_string, QueryExecutionContext=dict(Catalog=catalog, Database=database),
        WorkGroup=workgroup
    )

    result_config = dict(OutputLocation=output_location)
    if kms_key:
        result_config.update(
            EncryptionConfiguration=dict(EncryptionOption="SSE_KMS", KmsKey=kms_key)
        )
    kwargs.update(ResultConfiguration=result_config)

    athena_client = sagemaker_session.boto_session.client("athena", region_name=sagemaker_session.boto_region_name)
    response = athena_client.start_query_execution(**kwargs)

    query._current_query_execution_id = response["QueryExecutionId"]
    parse_result = urlparse(output_location, allow_fragments=False)
    query._result_bucket = parse_result.netloc
    query._result_file_prefix = parse_result.path.strip("/")
    return query._current_query_execution_id

def import_features(sm_region, bucket_name, feature_group_name, prediction_date, athena_query_location, service, region):
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
    feature_group = FeatureGroup(name=feature_group_name,
                                 sagemaker_session=feature_store_session)

    feature_query = feature_group.athena_query()

    run_query_in_workgroup(
        sagemaker_session=feature_store_session,
        query=feature_query,
        workgroup='aiop',
        query_string=f'select * from "{feature_query.table_name}"'+\
                      f' where prediction_date=\'{prediction_date}\'' +\
                      f' and eventtime=(select max(eventtime) from "{feature_query.table_name}"' +\
                      f' where prediction_date=\'{prediction_date}\' and service=\'{service}\' and region=\'{region}\')' +\
                      f' and service=\'{service}\'' +\
                      f' and region=\'{region}\'' +\
                      f' and is_deleted=false',
        output_location=athena_query_location
    )

    feature_query.wait()
    features_df = feature_query.as_dataframe().drop_duplicates().sort_values('ds')
    features_df.drop(columns=['eventtime', 'prediction_date', 'write_time',
                              'api_invocation_time', 'is_deleted', 'service', 'region'], inplace=True)
    return features_df


def prepare_data(data, sport_calendar, encoder=None, start_val=None, start_test=None, cols_with_order=[],
                 binary_cols_with_order=[], ts_col='ds'):
    prepared_data = data.copy()
    categories = []
    enc_columns = cols_with_order + binary_cols_with_order
    for col in enc_columns:
        categories.append(sorted(list(sport_calendar[col].dropna().unique())))
    logger.info(f'Categories are: {categories}')
    if categories and (encoder is None):
        logger.info('Fitting the encoder')
        encoder = OneHotEncoder(categories=categories, handle_unknown='ignore')
        encoder.fit(prepared_data[enc_columns])
    logger.info(f'Encoding columns {enc_columns}')
    enc_data = encoder.transform(prepared_data[enc_columns])
    enc_features_names = encoder.get_feature_names(enc_columns)
    prepared_data[enc_features_names] = enc_data.toarray()
    prepared_data.drop(columns=enc_columns, inplace=True)
    if start_val:
        train = prepared_data[prepared_data[ts_col] < start_val]
        if start_test:
            val = prepared_data[(prepared_data[ts_col] >= start_val) & (prepared_data[ts_col] < start_test)]
            test = prepared_data[prepared_data[ts_col] >= start_test]
        else:
            val = prepared_data[prepared_data[ts_col] >= start_val]
            test = None
    else:
        train = prepared_data
        val = None
        test = None
    return train, val, test, encoder


def generate_media_features(data):
    data['media'].fillna('media_unknown', inplace=True)
    data['media'] = data['media'].apply(lambda r: r.replace('|',',').split(','))
    data['media'] = data['media'].apply(tuple)
    return data


def encode_competitors(df,competitor_columns,competitors=None,prefix='team_'):
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


def select_model_features(data, features, TARGET_METRIC, date_col='date_utc'):
    model_data = data[features + ['id']]
    if TARGET_METRIC in data:
        model_data['y'] = data[TARGET_METRIC]
    else:
        model_data['y'] = None
    try:
        target_column = model_data.pop('y')
        model_data.insert(0, 'y', target_column)
    except Exception as e:
        logger.error("Error rearranging target column")
        logger.exception(e)
    if 'ds' not in model_data:
        model_data['ds'] = data[date_col]

    return model_data


def set_features_for_modeling(team_feature_cols=[]):
    features = []
    cols_with_order = []
    binary_cols_with_order = []
    scaler_features = []

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
    logger.info(f'Features are: {features}')
    return features, cols_with_order, binary_cols_with_order, scaler_features


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

def save_features(train_colums, output_dir, output_filename, header=False, exclude_cols=None):
    logger.info('Saving features list')
    feature_names = list(train_colums)
    try:
        output_path = os.path.join(base_dir, output_dir)
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        logger.info(f"Local path: {output_path}")
        if exclude_cols:
            feature_names = [i for i in feature_names if i not in exclude_cols]
        pd.Series(feature_names).to_csv(os.path.join(output_path, output_filename), index=False, header=header)
        logger.info("Data saved successfully")
    except Exception as ex:
        logger.error(ex)
        logger.error("Error saving output data")

def save_pickle_obj(object, output_dir, output_filename):
        logger.info(f'Saving {output_filename}')
        try:
            output_path = os.path.join(base_dir, output_dir)
            if not os.path.exists(output_path):
                os.mkdir(output_path)
            logger.info(f"Local path: {output_path}")
            pickle.dump(object, open(os.path.join(output_path, output_filename),'wb'))
            logger.info(f"{output_filename} saved successfully")
        except Exception as ex:
            logger.error(ex)
            logger.error(f"Error saving {output_filename}")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    logger.info("Starting feature encoding and selection")
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket-name", type=str)
    parser.add_argument('--sm-region', type=str)
    parser.add_argument("--service", type=str)
    parser.add_argument("--region", type=str)
    parser.add_argument("--prediction-date", type=str)
    parser.add_argument("--sport-name", type=str)
    parser.add_argument("--num-val-events", type=int)
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
    parser.add_argument("--feature-group-name", type=str)
    parser.add_argument("--athena-query-location", type=str)
    args = parser.parse_args()

    base_dir = "/opt/ml/processing"
    input_calendar_dir = "calendar"
    input_encoder_dir = "encoder"
    input_scaler_dir = "scaler"

    encoder_filename = 'encoder.pkl'
    scaler_filename = 'scaler.pkl'

    bucket_name = args.bucket_name
    prediction_date = args.prediction_date
    sport_name = args.sport_name
    feature_group_name = args.feature_group_name
    if feature_group_name:
        logger.info(f"Will use feature group: {feature_group_name}")
    else:
        logger.error("Error: feature group name not set")
        exit(1)

    num_val_events = args.num_val_events
    logger.info(sport_name)
    logger.debug("Attempting to import unprepared feature from Feature Store")
    feature_store_prefix = "aiop-sport-feature-store"
    logger.info(f"Prediction date: {prediction_date}")

    logger.info("Ingesting calendar data from local path")
    try:
        calendar = pd.read_csv(os.path.join(base_dir, input_calendar_dir, "calendar.csv"))
        logger.debug("Calendar data ingested successfully")
        logger.debug(calendar.columns)
    except Exception as e:
        logger.exception("Error ingesting calendar data")

    logger.info(f'Calendar: {calendar.head(2)}')
    logger.info(f'Calendar media nunique {calendar.media.nunique()}')  


    try:
        sleep(ATHENA_TIMEOUT)
        feature_df = import_features(args.sm_region, bucket_name, feature_group_name, prediction_date, 
                                                args.athena_query_location, args.service, args.region)
        logger.debug(f"Successfully imported features from {feature_group_name}")
        logger.info(f"Ingested features have shape: {feature_df.shape}")
    except Exception as e:
        logger.error("Error importing features")
        logger.exception(e)
    logger.info("Starting feature encoding and selection")

    model_data, future_events = feature_df[feature_df.data_type == 'training'], \
                                feature_df[feature_df.data_type == 'prediction']
    model_data.drop(columns=['data_type'], inplace=True)
    future_events.drop(columns=['data_type'], inplace=True)
    logger.debug(f"model_data columns: {model_data.columns}")
    logger.debug(f"model_data: {model_data.head()}")
    logger.debug(f"future_events columns: {future_events.columns}")
    logger.debug(f"future_events: {future_events.head()}")

    logger.debug(f"Number of validation events: {num_val_events}")

    start_test = prediction_date
    start_val = model_data.loc[model_data['ds'] < start_test, 'ds'].iloc[-num_val_events:].min()
    logger.info(f"Min - {model_data.ds.min()}")
    logger.info(f"Max - {model_data.ds.max()}")
    logger.info(f"Events before start_test - {model_data[model_data['ds'] < start_test].shape}")

    team_cols = ['team_0_display_name', 'team_1_display_name']
    if args.use_team_features:
        team_feature_cols = encode_competitors(calendar, team_cols)
    else:
        team_feature_cols = []
    features, cols_with_order, binary_cols_with_order, scaler_features = set_features_for_modeling(team_feature_cols)
    if features:
        logger.info("Features set successfully")
        logger.debug(features)
    else:
        logger.error("No features set for modeling")
        sys.exit(1)

    logger.info(f"Validation sample start: {start_val}")
    logger.info(f"Prediction sample start: {start_test}")
    logger.info(f"model_data start: {model_data.ds.min()}")
    logger.info(f"model_data end: {model_data.ds.max()}")
    calendar = calendar[calendar[TIMESTAMP_COL] >= model_data.ds.min()]
    logger.info(f"sport calendar start: {calendar[TIMESTAMP_COL].min()}")
    logger.info(f"sport calendar end: {calendar[TIMESTAMP_COL].max()}")
    
    logger.info(f'Calendar media nunique {calendar.media.nunique()}')

    # Include weights of instances in the model
    model_data['weight'] = np.sqrt(model_data['y'])
    weight_column = model_data.pop('weight')
    model_data.insert(1, 'weight', weight_column)

    encoder = None
    try:
        encoder = pickle.load(open(os.path.join(base_dir, input_encoder_dir, encoder_filename),'rb'))
        logger.debug("Encoder ingested successfully")
    except Exception as e:
        logger.exception("There is no availible encoder. It will be created")

    # Exclude columns from feature_store if they are not in selected features (from function set_features_for_modeling)
    model_data_columns = ['y', 'weight', 'id', 'ds'] + features
    selected_columns = [c for c in model_data.columns if c in model_data_columns]
    model_data = model_data[selected_columns]

    train, val, test, encoder = prepare_data(model_data, calendar,
                                             encoder,
                                             start_val, start_test,
                                             cols_with_order,
                                             binary_cols_with_order)

    scaler = None
    try:
        scaler = pickle.load(open(os.path.join(base_dir, input_scaler_dir, scaler_filename),'rb'))
        logger.debug("Scaler ingested successfully")
    except Exception as e:
        logger.exception("There is no availible scaler. It will be created")

    if len(scaler_features) > 0:
        logger.info(f'Scaling features {scaler_features}')
        if scaler is None:
            logger.debug("Fitting the scaler")
            scaler = StandardScaler()
            scaler.fit(train[scaler_features])
        train.loc[:, scaler_features] = scaler.transform(train[scaler_features])
        val.loc[:, scaler_features] = scaler.transform(val[scaler_features])
        if isinstance(test, pd.DataFrame) and len(test):
            test.loc[:, scaler_features] = scaler.transform(test[scaler_features])
    if train is not None:
        logger.info(f'train.shape {train.shape}')
    else:
        logger.error("Train dataset is empty")
        sys.exit(1)
    if val is not None:
        logger.info(f'val.shape {val.shape}')
    else:
        logger.info("Validataion dataset is empty")
    if test is not None:
        logger.info(f'test.shape {test.shape}')
    else:
        logger.info("Test dataset is empty")

    # future_events = generate_media_features(future_events)
    if len(scaler_features) > 0:
        future_events.loc[:, scaler_features] = scaler.transform(future_events[scaler_features])
    logger.info(f'future_events shape {future_events.shape}')

    future_events = select_model_features(future_events, features, TARGET_METRIC, date_col='ds')
    future_events, _, _, _ = prepare_data(future_events, calendar,
                                          encoder,
                                          None, None,
                                          cols_with_order, binary_cols_with_order)
    logger.info(f"Future events (head): {future_events.head(10)}")
    logger.debug(f"Prediction dataset shape: {future_events.shape}")
    if isinstance(test, pd.DataFrame) and len(test):
        test = pd.concat([test, future_events], axis=0, ignore_index=True)
    else:
        test = future_events

    logger.debug(f"Train: {train.columns}")
    logger.debug(f"Val: {val.columns}")
    logger.debug(f"Test: {test.columns}")
    logger.debug(f"Future_events: {future_events.columns}")


    # exclude_cols = ['ds', 'id', 'eventtime', 'prediction_date', 'data_type', 'write_time', 'api_invocation_time', ]
    save_processed_data(train, "output_train", "train.csv",
                        data_message="Saving data for model training", exclude_cols=['ds', 'id'])
    save_processed_data(val, "output_val", "val.csv",
                        data_message="Saving validation data", exclude_cols=['ds', 'id'])
    save_processed_data(test, "output_predict", "predict.csv",
                        data_message="Saving data for prediction", exclude_cols=['y', 'ds', 'id'])
    save_processed_data(test, "output_predict_ts", "predict_ts.csv",
                        data_message="Saving prediction data timestamps",
                        exclude_cols=[col for col in test.columns if col not in ['ds', 'id', 'y']],
                        header=True)
    save_features(train.columns, 'output_feature_names', 'feature_names.csv', exclude_cols=['ds', 'id', 'y'])

    save_pickle_obj(scaler, input_scaler_dir, scaler_filename)
    save_pickle_obj(encoder, input_encoder_dir, encoder_filename)


