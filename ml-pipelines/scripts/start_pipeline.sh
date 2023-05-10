#!/usr/bin/env bash

set -x

# If it is not a container load local vars
[ -z "$ECS_TASK" ] && . "$(dirname "$0")/local_vars.sh"

python -m pipelines.run_pipeline \
          --module-name "pipelines.${PIPELINE_TYPE}.${PIPELINE_MODULE_NAME}.pipeline" \
          --role-arn $SAGEMAKER_PIPELINE_ROLE_ARN \
          --tags "[{\"Key\":\"product_name\", \"Value\":\"aiop\"}]" \
