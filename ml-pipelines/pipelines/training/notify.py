"""Evaluation script for measuring mean squared error."""
import os
import json
import logging
import requests
import sys
import boto3
from datetime import datetime, date, timezone
from smart_open import smart_open
import numpy as np
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def notify(slack_url, icon, blocks):
    slack_data = {
        "username": 'sagemaker-alert',
        "icon_emoji": icon,
        "attachments": [
            {
                "blocks": blocks
            }
        ]
    }

    logger.debug("Sending message to Slack")
    byte_length = str(sys.getsizeof(slack_data))
    headers = {'Content-Type': "application/json", 'Content-Length': byte_length}
    response = requests.post(slack_url, data=json.dumps(slack_data), headers=headers)

    if response.status_code != 200:
        raise Exception(response.status_code, response.text)

def save_report(path, name, report_dict):
    os.makedirs(f'/opt/ml/processing/{path}', exist_ok=True)
    with open(f'/opt/ml/processing/{path}/{name}.json', 'w') as file:
        json.dump(report_dict, file)

if __name__ == "__main__":
    model_group_name = sys.argv[1]
    approval_status = sys.argv[2]
    execution_date_str = sys.argv[3]
    region = sys.argv[4]
    sagemaker_project_aws_region = sys.argv[5]
    algorithm_name = sys.argv[6]
    service = sys.argv[7]
    environment = sys.argv[8]
    slack_url = sys.argv[9]

    boto_session = boto3.Session(region_name=sagemaker_project_aws_region)
    sagemaker = boto_session.client("sagemaker")

    try:
        model = sagemaker.describe_model(ModelName=model_group_name)
        execution_date = datetime.now().replace(tzinfo=model['CreationTime'].tzinfo)#datetime.fromordinal(date.fromisoformat(execution_date_str).toordinal()).replace(tzinfo=model['CreationTime'].tzinfo)
        no_update_period = execution_date - model['CreationTime']
        package = sagemaker.describe_model_package(ModelPackageName=model['PrimaryContainer']['ModelPackageName'])

        if approval_status.lower() == 'approved':
            logger.debug("Opening evaluation report")
            with open('/opt/ml/processing/evaluation/evaluation.json') as evaluation_file:
                evaluation_report = json.load(evaluation_file)

                model_package_history = sagemaker.list_model_packages(
                    CreationTimeAfter=datetime(2015, 1, 1),
                    CreationTimeBefore=execution_date,
                    MaxResults=10,
                    ModelApprovalStatus='Approved',
                    ModelPackageGroupName=package['ModelPackageGroupName'],
                    SortBy='CreationTime',
                    SortOrder='Descending'
                )

                hist_metrics = {}
                for model_package_info in model_package_history['ModelPackageSummaryList']:
                    print(model_package_info)
                    hist_package = sagemaker.describe_model_package(ModelPackageName=model_package_info['ModelPackageArn'])

                    try:
                        with smart_open(hist_package['ModelMetrics']['ModelQuality']['Statistics']['S3Uri'], 'rb') as s3_source:
                            previous_evaluation_report = json.load(s3_source)
                            for hist_metric_name in previous_evaluation_report['regression_metrics']:
                                if hist_metric_name not in hist_metrics:
                                    hist_metrics[hist_metric_name] = []

                                hist_metrics[hist_metric_name].append(previous_evaluation_report['regression_metrics'][hist_metric_name]['value'])
                    except OSError as err:
                        logger.warning(err)

                # for hist_metric_name in hist_metrics:
                hist_metric_name = 'mape'
                metric_mean = np.mean(hist_metrics[hist_metric_name])

                if metric_mean < evaluation_report['regression_metrics'][hist_metric_name]['value'] and evaluation_report['regression_metrics'][hist_metric_name]['value'] / metric_mean > 2:
                    critical_notification_blocks = [
                        {
                            "type": "header",
                            "text": {
                                "type": "plain_text",
                                "text": "Model Accuracy Issues",
                                "emoji": True
                            }
                        },
                        {
                            "type": "divider"
                        },
                        {
                            "type": "section",
                            "text": {
                                "text": f"*Service: {region}.{service}*",
                                "type": "mrkdwn",
                            }
                        },
                        {
                            "type": "section",
                            "fields": [
                                {
                                    "type": "mrkdwn",
                                    "text": f"*Model:*\n{model_group_name}"
                                },
                                {
                                    "type": "mrkdwn",
                                    "text": f"*Metric:*\n{hist_metric_name}"
                                },
                                {
                                    "type": "mrkdwn",
                                    "text": f"*Metrics Value:*\n{evaluation_report['regression_metrics'][hist_metric_name]['value']}%"
                                }
                            ]
                        }
                    ]

                    notify(
                        icon = ':cloudy:', 
                        blocks = critical_notification_blocks,
                        slack_url = slack_url
                    )

                save_report('metrics', 'best', report_dict = {
                    'model_group_name': model_group_name,
                    'algorithm_name': algorithm_name,
                    'regression_metrics': evaluation_report['regression_metrics'],
                    'service': service,
                    'region': region,
                    'environment': environment
                })

#                 logger.debug("Creating message for notification")
                # notify(
                #     icon = ':sunny:', 
                #     title = f"New ModelPackage based on {algorithm_name} in {model_group_name}", 
                #     message = f"MAPE: {'%.2f' % (float(evaluation_report['regression_metrics']['mape']['value']))}%, MAE: {'%.2f' % float(evaluation_report['regression_metrics']['mae']['value'])}, Last update: {max(no_update_period.days, 0)} days ago",
                #     slack_url = slack_url
                # )
        else:
            no_updates_in_days = no_update_period.days

            if no_updates_in_days > 2:
                logger.info(f"No updates last {no_updates_in_days} days")

                notify(
                    icon = ':cloudy:', 
                    blocks = [
                        {
                            "type": "header",
                            "text": {
                                "type": "plain_text",
                                "text": "Outdated model detected",
                                "emoji": True
                            }
                        },
                        {
                            "type": "divider"
                        },
                        {
                            "type": "section",
                            "text": {
                                "text": f"*Service: {region}.{service}*",
                                "type": "mrkdwn",
                            }
                        },
                        {
                            "type": "section",
                            "text": {
                                "text": f"No updates last {no_updates_in_days} days",
                                "type": "mrkdwn",
                            }
                        },
                    ],
                    slack_url = slack_url
                )
    except ClientError as err:
        logger.error(err)
        notify(
            icon = ":broken_heart:",
            blocks = [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": "Failed Notifications",
                        "emoji": True
                    }
                },
                {
                    "type": "divider"
                },
                {
                    "type": "section",
                    "text": {
                        "text": f"*Service: {region}.{service}*",
                        "type": "mrkdwn",
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "text": str(err),
                        "type": "mrkdwn",
                    }
                },
            ],
            slack_url = slack_url
        )

