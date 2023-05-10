prediction_date = "2022-12-21"
execution_date = "2022-12-20"
predict_horizon = 10
bucket_name = "aiop-sagemaker-sandbox-us-east-1-051291311226"
history_depth = 365
calendar_cache_path = "airflow2/predictions/requests_cache"
num_val_events = 50
min_sample_days = 175
use_league_features = True
use_media_features =  True
use_competition_features = False
use_on_watch_espn =  False
use_date_features =  True
use_rank_features =  True
use_team_features =  False
encode_hour =  True
use_stats_features = False
use_netbase_features = True
use_n_tasks_feature = True
exclude_team_names = "Metro;Central;Atlantic;Pacific" 
feature_group_name = "us-east-1-fitt-unprocessed-hockey-v0"
traffic_threshold = -1
smoothed_extra_traffic = False
sport_name = "hockey"
service = "fitt"
region = "us-east-1"

s3_bucket = f"s3://{bucket_name}/airflow2"

input_traffic_path = f"{s3_bucket}/prepared/datadog_clean/{service}/{region}/new_relic_application_summary_throughput/{execution_date}"
input_baseline_path =  f"{s3_bucket}/predictions/baseline_aggregated/{service}/{region}/{execution_date}"
input_airings_dir = f"{s3_bucket}/prepared/aggregated_airings/{execution_date}"
input_calendar_dir = f"{s3_bucket}/prepared/calendar_views/{sport_name}/{execution_date}"
watch_graph_dir = f"{s3_bucket}/prepared/elb-watch-graph-api-traffic-events-network-5min-agg-within-event"
input_requests_cache_dir = f"{s3_bucket}/predictions/requests_cache"
nhl_rankings_dir = f"{s3_bucket}/prepared/nhl_rankings"
mens_college_hockey_rankings_dir = f"{s3_bucket}/prepared/mens-college-hockey_rankings"
exclude_network_name_path = f"{s3_bucket}/prepared/dictionaries/exclude_network_name/exclude_network_name.json"
input_netbase_dir = f"{s3_bucket}/prepared/netbase_metrics/{sport_name}/"
scheduled_tasks_dir = f"{s3_bucket}/prepared/scheduled_tasks_ts/{service}/{region}/{execution_date}"


best_params = {
    "alpha": "0.0",
    "colsample_bytree": "0.9",
    "eta": "0.05",
    "eval_metric": "mae",
    "gamma": "0",
    "lambda": "1.05",
    "max_depth": "10",
    "min_child_weight": "2",
    "num_round": 20,
    "objective": "reg:squarederror",
    "seed": "42",
    "subsample": "0.7",
    "verbosity": "1",
}
