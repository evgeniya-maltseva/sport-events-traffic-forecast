import argparse
import ast
import logging
import sys
import boto3
from datetime import datetime


logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--input-data", type=str, required=True)
    parser.add_argument("--region", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--local-output-path", type=str, required=True)

    args = parser.parse_args()

    # Todo create model name following naming convention
    model_name = args.model_name

    if model_name is None:
        print("Error: model-name value is not defined")
        sys.exit(2)

    create_transform_job(model_name,
                         args.input_data,
                         args.region,
                         args.output_path,
                         args.local_output_path)


def convert_struct(str_struct=None):
    return ast.literal_eval(str_struct) if str_struct else {}


def create_transform_job(
        model_name,
        input_data,
        region,
        output_path,
        local_output_path,
        content_type='text/csv',
        split_type='Line',
        instance_type='ml.m5.large',
        instance_count=1,
):
    sm_client = boto3.client('sagemaker', region)
    try:
        job_name = str(abs(hash(f'{model_name}-{datetime.today().strftime("%Y-%m-%d-%H-%M-%S")}')))
        transform_job = sm_client.create_transform_job(
            TransformJobName=job_name,
            ModelName=model_name,
            TransformInput={
                'DataSource': {
                    'S3DataSource': {
                        'S3DataType': 'S3Prefix',
                        'S3Uri': input_data
                    }
                },
                'ContentType': content_type,
                'SplitType': split_type
            },
            TransformOutput={
                'S3OutputPath': output_path
            },
            TransformResources={
                'InstanceType': instance_type,
                'InstanceCount': instance_count
            },
            Tags=[{"Key": "product_name", "Value": "aiop"}]
        )
        logger.info(f"Transform job name: {job_name}")
        waiter = sm_client.get_waiter('transform_job_completed_or_stopped')
        waiter.wait(TransformJobName=job_name)

        with open(f'{local_output_path}/_SUCCESS', 'w') as f:
            f.write('done')

    except Exception as e:
        print(f"Exception: {e}")
        sys.exit(1)

    logger.info(f"Transform job response: {transform_job}")


if __name__ == '__main__':
    main()
