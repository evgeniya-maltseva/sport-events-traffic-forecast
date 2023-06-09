# Global vars
export ECR_PREFIX="your_ecr_prefix"
export AWS_REGION="us-east-1"
export SAGEMAKER_PROJECT_AWS_REGION="us-east-1"
export DEFAULT_BUCKET="sagemaker-sandbox-us-east-1"
export TEMPORARY_BUCKET="temporary-sandbox-us-east-1"
export SAGEMAKER_PIPELINE_ROLE_ARN="your-sagemaker-execution-role"
export SLACK_URL="https://hooks.slack.com/services/your-slack-url"
export ENVIRONMENT="sandbox"
export ATHENA_QUERY_LOCATION="s3://temporary-sandbox-us-east-1/feature-store/query_results/"
export KMS_KEY_ID="your_kms_key_id"
# Airflow dag vars
export SERVICE="your_service"
export PIPELINE_TYPE="training"
# export PIPELINE_TYPE="prediction"
export PIPELINE_MODULE_NAME="sportforecast"
export SAGEMAKER_PROJECT_NAME="your-project-name"
export MODEL_NAME="hockey"

export EXECUTION_DATE="2023-03-06"
export PREDICTION_HORIZON_DAYS="10"
export BASELINE_PREDICTION_HORIZON="14"
export PREDICTION_DATE_TIMEDELTA_HOURS="24"
