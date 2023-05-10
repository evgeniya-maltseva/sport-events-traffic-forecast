import datetime
import os
import json

import boto3
import pandas as pd
import sagemaker
import sagemaker.session
from sagemaker.debugger import rule_configs, Rule, DebuggerHookConfig, CollectionConfig
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
)
from sagemaker.sklearn.processing import ScriptProcessor
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
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
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )


def get_pipeline(**kwargs):
    """Gets a SageMaker ML Pipeline instance working with on baseline data.

    Returns:
        an instance of a pipeline
    """
    region = kwargs.get('aws_region')
    sagemaker_project_aws_region = kwargs.get('sagemaker_project_aws_region')
    service = kwargs.get('service')
    role = kwargs.get('sagemaker_pipeline_role_arn')
    default_bucket = kwargs.get('default_bucket')
    temporary_bucket = kwargs.get('temporary_bucket')
    feature_group_name_suffix = kwargs.get('feature_group_name_suffix')
    model_name = kwargs.get('model_name')
    sport_name = kwargs.get('sport_name')
    if feature_group_name_suffix is None:
        print("Error: feature group is not defined")
        print("Please set parameter 'feature_group_name_suffix' in model-specific config file")
        exit(1)
    base_job_prefix = kwargs.get('sagemaker_project_name')
    execution_date = kwargs.get('execution_date')
    prediction_date_timedelta_hours = int(kwargs.get('prediction_date_timedelta_hours'))

    # get prediction date based on execution date
    execution_date_dt = pd.to_datetime(execution_date)
    prediction_date_dt = execution_date_dt + pd.Timedelta(hours=prediction_date_timedelta_hours)
    prediction_date = prediction_date_dt.strftime(format=DATETIME_PATTERN)

    environment = kwargs.get('environment')
    slack_url = kwargs.get('slack_url')
    kms_key_id = kwargs.get('kms_key_id')
    athena_query_location = kwargs.get('athena_query_location')

    exclude_network_name_path = f"s3://{default_bucket}/airflow2/{kwargs.get('exclude_network_data_prefix')}"
    exclude_network_name = json.dumps(pd.read_json(exclude_network_name_path, typ='Series')[sport_name])

    exclude_dates_path = f"s3://{default_bucket}/airflow2/{kwargs.get('exclude_dates_prefix')}"
    service_excl_dates = pd.read_json(exclude_dates_path, typ='Series').get(service, [])
    if service_excl_dates:
        model_excl_dates = service_excl_dates.get(model_name, service_excl_dates.get('default'))
        exclude_dates_str = json.dumps(model_excl_dates)
    else:
        exclude_dates_str = json.dumps(service_excl_dates)

    pipeline_name = '-'.join(
        [
            base_job_prefix,
            service,
            region,
            model_name,
            kwargs.get('pipeline_type')
        ]
    )

    sagemaker_session = get_session(sagemaker_project_aws_region, temporary_bucket)

    model_package_group_name = '-'.join(
        [
            base_job_prefix,
            service,
            region,
            model_name
        ]
    )

    encoder_path = f"s3://{default_bucket}/airflow2/{kwargs.get('encoder_path_prefix')}/{sport_name}/{service}/{region}/{execution_date}"
    scaler_path = f"s3://{default_bucket}/airflow2/{kwargs.get('scaler_path_prefix')}/{sport_name}/{service}/{region}/{execution_date}"

    exec_datetime: str = str(abs(hash(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f") + pipeline_name)))

    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    # parameters for pipeline execution
    instances_config: MLInstancesConfiguration = MLInstancesConfiguration.from_kwargs(kwargs_dict=kwargs)
    processing_instance_count_parameter = ParameterInteger(name="ProcessingInstanceCount",
                                                           default_value=instances_config.processing_instance_count)

    model_approval_status_pending = ParameterString(
        name="ModelApprovalStatusPending", default_value="PendingManualApproval"
    )
    model_approval_status_approved = ParameterString(
        name="ModelApprovalStatusApproved", default_value="Approved"
    )

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

    steps = []
    condition_cases = []

    step_ingest_and_store_features = ProcessingStep(
        name=f"collect{exec_datetime}",
        display_name="Collect Features",
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
        code=os.path.join(BASE_DIR, "scripts/ingest_and_store_features.py"),
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
                       "--exclude-team-names", kwargs.get('exclude_team_names', 'none'),
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

    for model_name, model_def in kwargs.get('model_defs').items():
        if not model_def.get("enabled"):
            continue

        step_prepare_and_select_features = ProcessingStep(
            name=f"prepare{model_name}{exec_datetime}",
            display_name=f"Select Features for {model_name}",
            processor=sklearn_processor,
            inputs=[
                ProcessingInput(
                    source=feature_ingest_output,
                    destination="/opt/ml/processing/calendar"
                )
            ],
            outputs=[
                ProcessingOutput(output_name="train", source="/opt/ml/processing/output_train"),
                ProcessingOutput(output_name="val", source="/opt/ml/processing/output_val"),
                ProcessingOutput(output_name="predict", source="/opt/ml/processing/output_predict"),
                ProcessingOutput(output_name="meta_predict", source="/opt/ml/processing/output_predict_ts"),
                ProcessingOutput(output_name="feature_names", source="/opt/ml/processing/output_feature_names"),
                ProcessingOutput(output_name="encoder", source="/opt/ml/processing/encoder", destination=encoder_path),
                ProcessingOutput(output_name="scaler", source="/opt/ml/processing/scaler", destination=scaler_path),
            ],
            code=os.path.join(BASE_DIR, "scripts/prepare_and_select_features.py"),
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

        training_image_params = model_def.get("training_image")
        training_image_uri = sagemaker.image_uris.retrieve(
            framework=training_image_params.get("framework"),
            region=sagemaker_project_aws_region,
            version=training_image_params.get('version'),
            py_version="py3",
            instance_type=instances_config.training_instance_type,
        )
        model_path = f"s3://{default_bucket}/airflow2/{base_job_prefix}/model/{model_name}"

        rules = [
            Rule.sagemaker(rule_configs.create_xgboost_report())
        ]
        save_interval = 5
        bucket_path = "s3://{}".format(temporary_bucket)

        model_estimator = Estimator(
            image_uri=training_image_uri,
            instance_type=instances_config.training_instance_type,
            instance_count=instances_config.training_instance_count,
            output_path=model_path,
            base_job_name=f"{base_job_prefix}/sport-{model_name}-train",
            sagemaker_session=sagemaker_session,
            hyperparameters=model_def.get("hyperparams"),
            role=role,
            debugger_hook_config=DebuggerHookConfig(
                s3_output_path=bucket_path,
                collection_configs=[
                    CollectionConfig(name="metrics", parameters={"save_interval": str(save_interval)}),
                    CollectionConfig(
                        name="feature_importance",
                        parameters={"save_interval": str(save_interval)},
                    ),
                    CollectionConfig(name="full_shap", parameters={"save_interval": str(save_interval)}),
                    CollectionConfig(name="average_shap", parameters={"save_interval": str(save_interval)}),
                    CollectionConfig(name="predictions", parameters={"save_interval": str(save_interval)}),
                    CollectionConfig(name="labels", parameters={"save_interval": str(save_interval)}),
                    CollectionConfig(name="trees", parameters={"save_interval": str(save_interval)}),
                ],
            ),
            rules=rules,
            output_kms_key=kms_key_id
        )
        print('Debagger rule output path: ',
              model_estimator.output_path + "/" + f"{base_job_prefix}/sport-{model_name}-train" + "/rule-output")

        step_model_train = TrainingStep(
            name=f"train{model_name}{exec_datetime}",
            display_name=f"Train {model_name} Model",
            estimator=model_estimator,
            inputs={
                "train": TrainingInput(
                    s3_data=step_prepare_and_select_features.properties.ProcessingOutputConfig.Outputs[
                        "train"
                    ].S3Output.S3Uri,
                    content_type="text/csv",
                ),
            }
        )
        steps.append(step_model_train)

        evaluation_image_params = model_def.get("evaluation_image")
        evaluation_image_uri = sagemaker.image_uris.retrieve(
            framework=evaluation_image_params.get("framework"),
            region=sagemaker_project_aws_region,
            version=evaluation_image_params.get('version'),
            image_scope=evaluation_image_params.get('image_scope'),
            py_version=evaluation_image_params.get('py_version', 'py3'),
            instance_type=instances_config.processing_instance_type,
        )
        script_eval = ScriptProcessor(
            image_uri=evaluation_image_uri,
            command=["python3"],
            instance_type=instances_config.processing_instance_type,
            instance_count=instances_config.processing_instance_count,
            base_job_name=f"{base_job_prefix}/sport-{model_name}-eval",
            sagemaker_session=sagemaker_session,
            role=role,
            output_kms_key=kms_key_id
        )
        evaluation_report = PropertyFile(
            name=f"EvaluationReport_{model_name}",
            output_name="evaluation",
            path="evaluation.json",
        )
        step_model_eval = ProcessingStep(
            name=f"evaluate{model_name}{exec_datetime}",
            display_name=f"Evaluate {model_name} Model",
            processor=script_eval,
            inputs=[
                ProcessingInput(
                    source=step_model_train.properties.ModelArtifacts.S3ModelArtifacts,
                    destination="/opt/ml/processing/model",
                ),
                ProcessingInput(
                    source=step_prepare_and_select_features.properties.ProcessingOutputConfig.Outputs[
                        "val"
                    ].S3Output.S3Uri,
                    destination="/opt/ml/processing/val",
                )
            ],
            job_arguments=[model_package_group_name, model_approval_status_approved, execution_date,
                           region, model_name, service, environment],
            outputs=[
                ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation"),
                ProcessingOutput(
                    output_name="metrics",
                    source="/opt/ml/processing/metrics",
                    destination='/'.join([
                        "s3:/",
                        default_bucket,
                        "airflow2",
                        "metrics",
                        base_job_prefix,
                        service,
                        region,
                        model_name,
                        execution_date
                    ])
                )
            ],
            code=os.path.join(BASE_DIR, f"scripts/{model_def.get('evaluation_script')}"),
            property_files=[evaluation_report]
        )
        steps.append(step_model_eval)

        model_metrics = ModelMetrics(
            model_statistics=MetricsSource(
                s3_uri="{}/evaluation.json".format(
                    step_model_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
                ),
                content_type="application/json"
            )
        )

        condition_args = {
            'name': model_name,
            'step': step_model_eval,
            'report': evaluation_report,
            'metric': kwargs.get('metric'),
            'estimator': model_estimator,
            'model_data': step_model_train.properties.ModelArtifacts.S3ModelArtifacts,
            'model_metrics': model_metrics,
            'image_uri': evaluation_image_uri
        }

        condition_cases.append(condition_args)

    def register_pending(condition_case, i, j):
        return RegisterModel(
            name=f"pending{condition_case['name']}{i}{j}{exec_datetime}",
            display_name=f"Register {condition_case['name']} model as pending",
            estimator=condition_case['estimator'],
            model_data=condition_case['model_data'],
            content_types=["text/csv"],
            response_types=["text/csv"],
            inference_instances=instances_config.register_model_instance_types,
            transform_instances=instances_config.register_model_instance_types,
            model_package_group_name=model_package_group_name,
            approval_status=model_approval_status_pending,
            model_metrics=condition_case['model_metrics'],
            customer_metadata_properties={
                'encoder': step_prepare_and_select_features.properties.ProcessingOutputConfig.Outputs["encoder"].S3Output.S3Uri,
                'scaler': step_prepare_and_select_features.properties.ProcessingOutputConfig.Outputs["scaler"].S3Output.S3Uri,
            }
        )

    def register_approved(condition_case, i, j):
        return RegisterModel(
            name=f"approved{condition_case['name']}{i}{j}{exec_datetime}",
            display_name=f"Register {condition_case['name']} model as approved",
            estimator=condition_case['estimator'],
            model_data=condition_case['model_data'],
            content_types=["text/csv"],
            response_types=["text/csv"],
            inference_instances=instances_config.register_model_instance_types,
            transform_instances=instances_config.register_model_instance_types,
            model_package_group_name=model_package_group_name,
            approval_status=model_approval_status_approved,
            model_metrics=model_metrics,
            customer_metadata_properties={
                'encoder': step_prepare_and_select_features.properties.ProcessingOutputConfig.Outputs["encoder"].S3Output.S3Uri,
                'scaler': step_prepare_and_select_features.properties.ProcessingOutputConfig.Outputs["scaler"].S3Output.S3Uri,
            }
        )

    def notify_on_approve(condition_case, i, j):
        return ProcessingStep(
            name=f"notify{condition_case['name']}{i}{j}{exec_datetime}",
            display_name=f"Notify on new {condition_case['name']} model",
            processor=ScriptProcessor(
                image_uri=f"{kwargs.get('ecr_prefix')}/aiop-ml-pipeline-v2",
                command=["python3"],
                instance_type=instances_config.meta_data_steps_instance_type,
                instance_count=instances_config.meta_data_steps_instance_count,
                base_job_name=f"{base_job_prefix}/script-notify",
                sagemaker_session=sagemaker_session,
                role=role,
                output_kms_key=kms_key_id
            ),
            job_arguments=[model_package_group_name,
                           model_approval_status_approved,
                           execution_date,
                           region,
                           sagemaker_project_aws_region,
                           condition_case['name'],
                           service,
                           environment,
                           slack_url],
            inputs=[
                ProcessingInput(
                    source=condition_case['step'].properties.ProcessingOutputConfig.Outputs[
                        "evaluation"
                    ].S3Output.S3Uri,
                    destination="/opt/ml/processing/evaluation",
                ),
            ],
            outputs=[
                ProcessingOutput(
                    output_name="metrics",
                    source="/opt/ml/processing/metrics",
                    destination='/'.join([
                        "s3:/",
                        default_bucket,
                        "airflow2",
                        "metrics",
                        base_job_prefix,
                        service,
                        region,
                        model_name,
                        execution_date
                    ])
                )
            ],
            code=os.path.join(BASE_DIR, "../notify.py")
        )
    
    steps.append(register_approved(condition_cases[0], 0, 1))
    steps.append(notify_on_approve(condition_cases[0], 0, 1))

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_count_parameter,
            model_approval_status_pending,
            model_approval_status_approved
        ],
        steps=steps,
        sagemaker_session=sagemaker_session,
    )
    return pipeline
