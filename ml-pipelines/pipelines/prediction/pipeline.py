"""Example workflow pipeline script for baseline pipeline.

                                               . -RegisterModel
                                              .
    Process-> Train -> Evaluate -> Condition .
                                              .
                                               . -(stop)

Implements a get_pipeline(**kwargs) method.
"""
import datetime
import os
import json

import boto3
import pandas as pd
import sagemaker
import sagemaker.session
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
)
from sagemaker.sklearn.processing import ScriptProcessor
from sagemaker.workflow.parameters import (
    ParameterInteger,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import (
    ProcessingStep,
)

from pipelines.configuration import MLInstancesConfiguration

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
DATETIME_PATTERN = "%Y-%m-%d"


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

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

def get_pipeline(**kwargs):
    """Gets a SageMaker ML Pipeline instance

    Returns:
        an instance of a pipeline
    """
    region = kwargs.get('aws_region')
    sagemaker_project_aws_region = kwargs.get('sagemaker_project_aws_region')
    service = kwargs.get('service')
    role = kwargs.get('sagemaker_pipeline_role_arn')
    default_bucket = kwargs.get('default_bucket')
    temporary_bucket = kwargs.get('temporary_bucket')
    project_name = kwargs.get('sagemaker_project_name')
    model_name = kwargs.get('model_name')
    execution_date = kwargs.get('execution_date')
    prediction_date_timedelta_hours = int(kwargs.get('prediction_date_timedelta_hours'))
    # get prediction date based on execution date
    execution_date_dt = pd.to_datetime(execution_date)
    prediction_date_dt = execution_date_dt + pd.Timedelta(hours=prediction_date_timedelta_hours)
    prediction_date = prediction_date_dt.strftime(format=DATETIME_PATTERN)

    feature_group_name_suffix = kwargs.get('feature_group_name_suffix')
    kms_key_id = kwargs.get('kms_key_id')

    sport_name = kwargs.get('sport_name')

    athena_query_location = kwargs.get('athena_query_location')

    if feature_group_name_suffix is None:
        print("Error: feature group is not defined")
        print("Please set parameter 'feature_group_name_suffix' in model-specific config file")
        exit(1)
    base_job_prefix = project_name

    prediction_output_path = '/'.join(
        [
            "s3:/",
            default_bucket,
            "airflow2",
            "predictions",
            project_name,
            service,
            region,
            model_name,
            execution_date
        ]

    )

    pipeline_name = '-'.join(
        [
            project_name,
            service,
            region,
            model_name,
            kwargs.get('pipeline_type')
        ]
    )

    sagemaker_session, sagemaker_client = get_session(sagemaker_project_aws_region, temporary_bucket)

    model_group_name = f"{project_name}-{service}-{region}-{model_name}"
    model = sagemaker_client.describe_model(ModelName=model_group_name)
    package = sagemaker_client.describe_model_package(ModelPackageName=model['PrimaryContainer']['ModelPackageName'])
    customer_metadata = package.get('CustomerMetadataProperties', None)
    if customer_metadata is not None:
        encoder_path = customer_metadata.get('encoder')
        scaler_path = customer_metadata.get('scaler')
    else:
        raise RuntimeError('Pathes to scaler and encoder are unknown')

    exec_datetime: str = str(abs(hash(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f") + pipeline_name)))

    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    # parameters for pipeline execution
    instances_config: MLInstancesConfiguration = MLInstancesConfiguration.from_kwargs(kwargs_dict=kwargs)
    processing_instance_count_parameter = ParameterInteger(name="ProcessingInstanceCount",
                                                           default_value=instances_config.processing_instance_count)

    preprocess_image_uri = f"{kwargs.get('ecr_prefix')}/aiop-ml-sport-preprocessor"
    sklearn_processor = ScriptProcessor(
        image_uri=preprocess_image_uri,
        command=["python3"],
        instance_type=instances_config.processing_instance_type,
        instance_count=instances_config.processing_instance_count,
        base_job_name=f"{base_job_prefix}/sklearn-sport-preprocess",
        sagemaker_session=sagemaker_session,
        role=role,
        output_kms_key=kms_key_id
    )

    exclude_network_name_path = f"s3://{default_bucket}/airflow2/{kwargs.get('exclude_network_data_prefix')}"
    exclude_network_name = json.dumps(pd.read_json(exclude_network_name_path, typ='Series')[sport_name])

    exclude_dates_path = f"s3://{default_bucket}/airflow2/{kwargs.get('exclude_dates_prefix')}"
    service_excl_dates = pd.read_json(exclude_dates_path, typ='Series').get(service, [])
    if service_excl_dates:
        model_excl_dates = service_excl_dates.get(model_name, service_excl_dates.get('default'))
        exclude_dates_str = json.dumps(model_excl_dates)
    else:
        exclude_dates_str = json.dumps(service_excl_dates)


    steps = []
    step_ingest_and_store_features = ProcessingStep(
        name=f"collect{exec_datetime}",
        processor=sklearn_processor,
        inputs=[
            ProcessingInput(
                source=f"s3://{default_bucket}/airflow2/{kwargs.get('input_traffic_data_prefix')}/{service}/{region}/new_relic_application_summary_throughput/{execution_date}",
                destination="/opt/ml/processing/input_traffic",
            ),
            ProcessingInput(
                source=f"s3://{default_bucket}/airflow2/{kwargs.get('input_baseline_data_prefix')}/{service}/{region}/{execution_date}",
                destination="/opt/ml/processing/input_baseline",
            ),
            ProcessingInput(
                source=f"s3://{default_bucket}/airflow2/{kwargs.get('input_airings_data_prefix')}/{execution_date}",
                destination="/opt/ml/processing/input_airings",
            ),
            ProcessingInput(
                source=f"s3://{default_bucket}/airflow2/{kwargs.get('input_calendar_data_prefix')}/{execution_date}",
                destination="/opt/ml/processing/input_calendar",
            ),
            ProcessingInput(
                source=f"s3://{default_bucket}/airflow2/{kwargs.get('input_watchgraph_data_prefix')}",
                destination="/opt/ml/processing/input_watchgraph",
            ),
            ProcessingInput(
                source=f"s3://{default_bucket}/airflow2/{kwargs.get('input_netbase_data_prefix')}/{sport_name}",
                destination="/opt/ml/processing/input_netbase",
            ),
            ProcessingInput(
                source=f"s3://{default_bucket}/airflow2/{kwargs.get('input_nhl_rankings_data_prefix')}",
                destination="/opt/ml/processing/input_nhl_rankings",
            ),
            ProcessingInput(
                source=f"s3://{default_bucket}/airflow2/{kwargs.get('input_mch_rankings_data_prefix')}",
                destination="/opt/ml/processing/input_mch_rankings",
            ),
            ProcessingInput(
                source=f"s3://{default_bucket}/airflow2/{kwargs.get('input_scheduled_tasks_data_prefix')}/{service}/{region}/{execution_date}",
                destination="/opt/ml/processing/input_scheduled_tasks",
            )
        ],
        outputs=[
            ProcessingOutput(output_name="calendar", source="/opt/ml/processing/calendar"),
        ],
        code=os.path.join(BASE_DIR, "../../training/sportforecast/scripts/ingest_and_store_features.py"),
        job_arguments=["--prediction-date", prediction_date,
                       "--bucket-name", default_bucket,
                       '--sm-region', sagemaker_project_aws_region,
                       "--service", service,
                       "--region", region,
                       "--history-depth", kwargs.get('history_depth', '-1'),
                       "--num-val-events", kwargs.get('num_val_events'),
                       "--min-sample-days", kwargs.get('min_sample_days', '175'),
                       "--use-league-features", kwargs.get('use_league_features'),
                       "--use-media-features", kwargs.get('use_media_features'),
                       "--use-competition-features", kwargs.get('use_competition_features'),
                       "--use-on-watch-espn", kwargs.get('use_on_watch_espn'),
                       "--use-date-features", kwargs.get('use_date_features'),
                       "--use-rank-features", kwargs.get('use_rank_features'),
                       "--use-team-features", kwargs.get('use_team_features'),
                       "--use-netbase-features", kwargs.get('use_netbase_features'),
                       "--use-n-tasks-feature", kwargs.get('use_n_tasks_feature'),
                       "--encode-hour", kwargs.get('encode_hour'),
                       "--feature-group-name", feature_group_name_suffix,
                       "--traffic-threshold", kwargs.get('traffic_threshold'),
                       "--smoothed-extra-traffic", kwargs.get('smoothed_extra_traffic'),
                       "--sport-name", kwargs.get('sport_name'),
                       "--predict-horizon", kwargs.get('prediction_horizon_days'),
                       "--exclude-network-name", exclude_network_name,
                       "--exclude-dates-str", exclude_dates_str,
                       ]
    )
    steps.append(step_ingest_and_store_features)

    feature_ingest_output = \
        step_ingest_and_store_features.properties.ProcessingOutputConfig.Outputs[
            "calendar"
        ].S3Output.S3Uri

    step_prepare_and_select_features = ProcessingStep(
        name=f"prepare{exec_datetime}",
        processor=sklearn_processor,
        inputs=[
            ProcessingInput(
                source=feature_ingest_output,
                destination="/opt/ml/processing/calendar",
            ),
            ProcessingInput(
                source=encoder_path,
                destination="/opt/ml/processing/encoder",
            ),
            ProcessingInput(
                source=scaler_path,
                destination="/opt/ml/processing/scaler",
            )
        ],
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/output_train"),
            ProcessingOutput(output_name="val", source="/opt/ml/processing/output_val"),
            ProcessingOutput(output_name="predict", source="/opt/ml/processing/output_predict"),
            ProcessingOutput(output_name="meta_predict", source="/opt/ml/processing/output_predict_ts"),
            ProcessingOutput(output_name="feature_names", source="/opt/ml/processing/output_feature_names")
        ],
        code=os.path.join(BASE_DIR, "../../training/sportforecast/scripts/prepare_and_select_features.py"),
        job_arguments=["--prediction-date", prediction_date,
                       "--bucket-name", default_bucket,
                       '--sm-region', sagemaker_project_aws_region,
                       "--service", service,
                       "--region", region,
                       "--num-val-events", kwargs.get('num_val_events'),
                       "--use-league-features", kwargs.get('use_league_features'),
                       "--use-media-features", kwargs.get('use_media_features'),
                       "--use-competition-features", kwargs.get('use_competition_features'),
                       "--use-on-watch-espn", kwargs.get('use_on_watch_espn'),
                       "--use-date-features", kwargs.get('use_date_features'),
                       "--use-rank-features", kwargs.get('use_rank_features'),
                       "--use-team-features", kwargs.get('use_team_features'),
                       "--use-netbase-features", kwargs.get('use_netbase_features'),
                       "--use-n-tasks-feature", kwargs.get('use_n_tasks_feature'),
                       "--encode-hour", kwargs.get('encode_hour'),
                       "--feature-group-name", feature_group_name_suffix,
                       "--sport-name", kwargs.get('sport_name'),
                       "--athena-query-location", athena_query_location]
    )
    steps.append(step_prepare_and_select_features)

    # processing step for batch transform job
    image_uri = f"{kwargs.get('ecr_prefix')}/aiop-sagemaker-processing-job"
    script_prediction = ScriptProcessor(
        image_uri=image_uri,
        command=["python3"],
        instance_type=instances_config.processing_instance_type,
        instance_count=instances_config.processing_instance_count,
        base_job_name=f"{base_job_prefix}/batch-transform-job",
        sagemaker_session=sagemaker_session,
        role=role,
        output_kms_key=kms_key_id
    )

    local_output_path = '/opt/ml/processing/depends_on/output'
    step_prediction = ProcessingStep(
        name=f"predict{exec_datetime}",
        processor=script_prediction,
        code=os.path.join(BASE_DIR, "../scripts/create_transform_job.py"),
        inputs=[
            ProcessingInput(
                source=step_prepare_and_select_features.properties.ProcessingOutputConfig.Outputs[
                    "predict"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/depends_on/input",
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="success_flag",
                source=local_output_path
            ),
        ],
        job_arguments=[
            "--input-data", step_prepare_and_select_features.properties.ProcessingOutputConfig.Outputs[
                "predict"
            ].S3Output.S3Uri,
            "--output-path", prediction_output_path,
            "--local-output-path", local_output_path,
            "--model-name", model_group_name,
            "--region", sagemaker_project_aws_region
        ],
    )

    steps.append(step_prediction)

    script_add_metadata = ScriptProcessor(
        image_uri=image_uri,
        command=["python3"],
        instance_type=instances_config.processing_instance_type,
        instance_count=instances_config.processing_instance_count,
        base_job_name=f"{base_job_prefix}/processing_prediction_ts",
        sagemaker_session=sagemaker_session,
        role=role,
        output_kms_key=kms_key_id
    )

    step_add_metadata = ProcessingStep(
        name=f"postprocess{exec_datetime}",
        processor=script_add_metadata,
        code=os.path.join(BASE_DIR, "../scripts/process_predictions.py"),
        inputs=[
            ProcessingInput(
                source=step_prediction.properties.ProcessingOutputConfig.Outputs[
                    "success_flag"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/depends_on/input",
            ),
            ProcessingInput(
                source=prediction_output_path,
                destination="/opt/ml/processing/input_predictions",
            ),
            ProcessingInput(
                source=step_prepare_and_select_features.properties.ProcessingOutputConfig.Outputs[
                    "meta_predict"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/input_predictions_ts",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="predict",
                source="/opt/ml/processing/output_predictions",
                destination=prediction_output_path
            ),
        ],
        job_arguments=[
            "--metadata-columns", "ds,id",
            "--region", sagemaker_project_aws_region
        ],
    )

    steps.append(step_add_metadata)

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_count_parameter,
        ],
        steps=steps,
        sagemaker_session=sagemaker_session,
    )
    return pipeline
