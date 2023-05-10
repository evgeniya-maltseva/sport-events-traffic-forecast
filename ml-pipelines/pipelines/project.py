import boto3
import sagemaker.session


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


def get_project_tags(project):
    return [
        {"Key": "sagemaker:project-name", "Value": project['ProjectName']},
        {"Key": "sagemaker:project-id", "Value": project['ProjectId']}
    ]


def get_project(**kwargs):
    sagemaker_session = get_session(kwargs.get('sagemaker_project_aws_region'),
                                    kwargs.get('default_bucket'))
    project_name = kwargs.get('sagemaker_project_name')
    projects = sagemaker_session.sagemaker_client.list_projects(
        NameContains=project_name)['ProjectSummaryList']
    if projects:
        project = projects[0]
    else:
        # Project doesn't exist create a new one
        project = sagemaker_session.sagemaker_client.create_project(
            ProjectName=project_name,
            ProjectDescription='AIOP project',
            ServiceCatalogProvisioningDetails={
                'ProductId': kwargs.get('servicecatalog_product_id'),
                'ProvisioningArtifactId': kwargs.get('servicecatalog_product_version_id')
            }
        )

    return project
