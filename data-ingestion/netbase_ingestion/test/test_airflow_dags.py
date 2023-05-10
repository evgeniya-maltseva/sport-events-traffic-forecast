import pathlib
import sys

import pytest

project_root_path = pathlib.Path(__file__).parent.parent.parent
sys.path.append(str(project_root_path))
sys.path.append(str(project_root_path.joinpath("e2e_dag_builder")))

from testing_utilities.dag_testing import mock_airflow_vars_connections, check_dags_correctness, \
    get_module_airflow_folder

variables_for_mocking = {
    "artifact_bucket": "artifact_bucket",
    "data_bucket": "data_bucket",
    "environment": "environment",
    "security_group": "security_group",
    "vpc_subnet": "vpc_subnet",
    "common_sports": "[{\"sport_name\": \"baseball\", \"days_to_predict\": 10}, {\"sport_name\": \"basketball\", \"days_to_predict\": 10}, {\"sport_name\": \"hockey\", \"days_to_predict\": 10}, {\"sport_name\": \"football\", \"days_to_predict\": 10}, {\"sport_name\": \"ufc\", \"days_to_predict\": 7, \"ml_pipeline_module\": \"ufcforecast\", \"predict_output_filename\": \"predict.csv.out\", \"calendar_sport_name\": \"mma\"}]",
}


@pytest.fixture()
def patch_airflow_calls(monkeypatch):
    mock_airflow_vars_connections(monkeypatch=monkeypatch,
                                  variables=variables_for_mocking)


def test_module_dags_correctness(patch_airflow_calls):
    check_dags_correctness(get_module_airflow_folder())
