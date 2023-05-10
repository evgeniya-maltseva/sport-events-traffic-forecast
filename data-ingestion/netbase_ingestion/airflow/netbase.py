from datetime import timedelta, datetime
from typing import Tuple, List, Dict, Optional

from airflow import DAG
from airflow.models import Variable
from airflow.operators.dummy import DummyOperator
from airflow.utils.trigger_rule import TriggerRule

from e2e_dag.aiop.configuration.configs import SportConfig
from e2e_dag.aiop.configuration.path_constants import ONLY_DATE_PARTITION_PATTERN, \
    NETBASE_METRICS_LOCATION, AGGREGATED_AIRINGS_LOCATION, \
    SPORT_CALENDAR_LOCATION_TEMPLATE, EXCLUDE_NETWORK_NAME_PATH, UFC_ENRICHED_CALENDAR_LOCATION_TEMPLATE
from e2e_dag.aiop.configuration.pool_constants import AIOP_NETBASE_API_POOL
from e2e_dag.aiop.configuration.template_constants import EXECUTION_DATE_OR_DS_NEXT_DAY_TEMPLATE, \
    ARTIFACT_BUCKET_TEMPLATE, DATA_BUCKET_TEMPLATE
from e2e_dag.aiop.configuration.variable_constants import COMMON_SPORTS_KEY
from e2e_dag.aiop.dq.presence_checker_factory import get_check_presence_step
from e2e_dag.aiop.etl.external_etl_steps import get_airings_sensor, get_ufc_calendar_transformation_sensor
from e2e_dag.aiop.operators.ecs_provider import EcsUser, ECSOperatorWithConfigOnS3
from e2e_dag.aiop.operators.slack_notifier import get_notify_slack_on_dag_failure
from e2e_dag.aiop.operators.upcoming_events_checker import UpcomingEventsBySportBranchOperator

artifact_bucket = ARTIFACT_BUCKET_TEMPLATE
data_bucket = DATA_BUCKET_TEMPLATE

script_path = f"s3://{artifact_bucket}/etl/netbase/netbase_ingest.py"
team_sports_script_path = f"s3://{artifact_bucket}/etl/netbase/netbase_ingest_team_sports.py"

default_args = {
    'owner': 'Evgeniia Maltseva',
    'depends_on_past': False,
    'email': ['airflow@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    f'aiop-data-netbase-ingestion',
    default_args=default_args,
    schedule_interval="0 21 * * *",
    start_date=datetime(2022, 12, 1),
    catchup=False,
    is_paused_upon_creation=True,
    tags=['v0.1'],
    on_failure_callback=get_notify_slack_on_dag_failure()
)
# next-day because of shifted earlier schedule_interval
dag_execution_date_template = EXECUTION_DATE_OR_DS_NEXT_DAY_TEMPLATE

sports: List[SportConfig] = [SportConfig.from_raw(raw_sport)
                             for raw_sport in Variable.get(COMMON_SPORTS_KEY, deserialize_json=True)]
team_sports: List[SportConfig] = [sport for sport in sports
                                  if sport.sport_name not in ["ufc", "baseball"]]
ufc_config: Optional[SportConfig] = next((sport for sport in sports if sport.sport_name == "ufc"), None)

ufc_calendar_transformation_sensor = get_ufc_calendar_transformation_sensor(dag=dag)
airings_sensor = get_airings_sensor(dag=dag)

sport_to_task_with_period_days: Dict[str, Tuple[str, int]] = {}
ingest_tasks: List[ECSOperatorWithConfigOnS3] = []
checkers: List[ECSOperatorWithConfigOnS3] = []
ecs_user = EcsUser()

if ufc_config is not None:
    ufc_ingest_task_id = "netbase_ingest_ufc"
    netbase_ingest_ufc = ecs_user.get_ecs_operator(
        task_id=ufc_ingest_task_id,
        dag=dag,
        command=["ingest.py",
                 "--s3-bucket", data_bucket,
                 "--netbase-metrics-location", NETBASE_METRICS_LOCATION,
                 "--ingest-date", dag_execution_date_template,
                 "--ufc-calendar-path-template", UFC_ENRICHED_CALENDAR_LOCATION_TEMPLATE],
        environment=[{'name': 'PYTHON_SCRIPT_S3_URL',
                      'value': script_path}],
        pool=AIOP_NETBASE_API_POOL)

    presence_checker_ufc: ECSOperatorWithConfigOnS3 = get_check_presence_step(
        task_id="data_presence_checker_ufc",
        dag=dag,
        data_bucket_template=data_bucket,
        artifact_bucket_template=artifact_bucket,
        processing_date_template=dag_execution_date_template,
        relative_paths_patterns_to_check=[f"{NETBASE_METRICS_LOCATION}/mma{ONLY_DATE_PARTITION_PATTERN}/*.csv"]
    )
    netbase_ingest_ufc >> presence_checker_ufc

    sport_to_task_with_period_days[ufc_config.calendar_sport_name] = (ufc_ingest_task_id, ufc_config.days_to_predict)
    ingest_tasks.append(netbase_ingest_ufc)
    checkers.append(presence_checker_ufc)


def get_sport_tasks(sport_name) -> Tuple[ECSOperatorWithConfigOnS3, ECSOperatorWithConfigOnS3]:
    ingest_task: ECSOperatorWithConfigOnS3 = ecs_user.get_ecs_operator(
        task_id=f'netbase_ingest_{sport_name}',
        dag=dag,
        command=["ingest.py",
                 "--s3-bucket", data_bucket,
                 "--ingest-date", dag_execution_date_template,
                 "--aggregated-airings-location", AGGREGATED_AIRINGS_LOCATION,
                 "--netbase-metrics-location", NETBASE_METRICS_LOCATION,
                 "--sport-calendar-path-template", SPORT_CALENDAR_LOCATION_TEMPLATE,
                 "--exclude-network-names-location", EXCLUDE_NETWORK_NAME_PATH,
                 "--sport-name", sport_name],
        environment=[{'name': 'PYTHON_SCRIPT_S3_URL',
                      'value': team_sports_script_path}],
        pool=AIOP_NETBASE_API_POOL)
    checker_task: ECSOperatorWithConfigOnS3 = get_check_presence_step(
        task_id=f"data_presence_checker_{sport_name}",
        dag=dag,
        data_bucket_template=data_bucket,
        artifact_bucket_template=artifact_bucket,
        processing_date_template=dag_execution_date_template,
        relative_paths_patterns_to_check=[f"{NETBASE_METRICS_LOCATION}/{sport_name}{ONLY_DATE_PARTITION_PATTERN}/*.csv"]
    )
    ingest_task >> checker_task
    return ingest_task, checker_task


for sport in team_sports:
    sport_to_task_with_period_days[sport.calendar_sport_name] = (f"netbase_ingest_{sport.sport_name}",
                                                                 sport.days_to_predict)
    ingest_sport_task, presence_checker = get_sport_tasks(sport.sport_name)
    ingest_tasks.append(ingest_sport_task)
    checkers.append(presence_checker)

check_upcoming_events_branch = UpcomingEventsBySportBranchOperator(
    sport_to_task_with_period_days=sport_to_task_with_period_days,
    execution_date=dag_execution_date_template,
    bucket=data_bucket,
    task_id="check_upcoming_events_branch")

end_step = DummyOperator(task_id="dummy_after_netbase_ingest",
                         dag=dag,
                         trigger_rule=TriggerRule.NONE_FAILED)

[ufc_calendar_transformation_sensor, airings_sensor] >> check_upcoming_events_branch >> ingest_tasks
checkers >> end_step

