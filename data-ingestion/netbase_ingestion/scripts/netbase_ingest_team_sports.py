import argparse
import logging
import re
from argparse import Namespace
from typing import List, Dict

import pandas as pd
import requests
import s3fs
from pandas._libs.tslibs.timestamps import Timestamp

from netbase_utils import *


class AirflowSkipException(Exception):
    """To not use airflow package here"""


logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

s3 = s3fs.S3FileSystem()
NETBASE_TOKEN = 'your_netbaze_token'
httpsession = requests.Session()
httpsession.headers.update({'Authorization': 'Bearer ' + NETBASE_TOKEN})

two_words_states = [
    "New Hampshire",
    "New Jersey",
    "New Mexico",
    "New York",
    "North Carolina",
    "North Dakota",
    "Rhode Island",
    "South Carolina",
    "South Dakota",
    "West Virginia",
    "South Florida",
    "Southern Utah",
    "Notre Dame",
    "Los Angeles",
    "Las Vegas",
]

TOPIC_ID = {
    'hockey': 940401,  # NHL 1887303 #NHL LAST MONTH
    'football': 1410787,  # NFL 2 years
    'basketball': 1697931,  # NBA LAST MONTH
    'baseball': 1040201,  # MLB Stats LAST MONTH
}


def parse_arguments():
    parser = argparse.ArgumentParser(description='Ingest data from DataDog')
    parser.add_argument(
        "--s3-bucket",
        help="DataDog API key",
    )
    parser.add_argument(
        "--netbase-metrics-location",
        help="Location of netbase on S3",
    )
    parser.add_argument(
        "--ingest-date",
        help="Ingest date in format Y-m-d (default: current day)",
        required=False,
        default=datetime.now().strftime("%Y-%m-%d"),
    )
    parser.add_argument(
        "--sport-name",
        help="Team sport name",
    )
    parser.add_argument(
        "--sport-calendar-path-template",
        help="Sport calendar template path dependent on date and sport"
    )

    return parser.parse_args()


def get_keywords(row: Dict, league: str):
    keywords = []
    names = row.get("name").split(' at ')
    if league == "caribbean-series":
        pass
    else:
        for name in names:
            for state in two_words_states:
                if state in name:
                    short_name = name.replace(state, '')
                    keywords.append(short_name)
            if "State" in name:
                short_name = name.split('State')[-1]
                keywords.append(short_name)
            elif "City" in name:
                short_name = name.split('City')[-1]
                keywords.append(short_name)
            else:
                short_name = ' '.join(name.split(' ')[1:])
                keywords.append(short_name)

    keywords.extend(names)
    return list(set(kw for kw in keywords if kw is not None))


if __name__ == "__main__":
    arguments: Namespace = parse_arguments()
    s3_bucket: str = arguments.s3_bucket
    sport_name: str = arguments.sport_name
    ingest_date_str = arguments.ingest_date
    ingest_date: Timestamp = pd.to_datetime(ingest_date_str)
    netbase_metrics_location: str = arguments.netbase_metrics_location

    calendar_path: str = arguments.sport_calendar_path_template.format(date=ingest_date,
                                                                       sport=sport_name)

    print("start")
    print("read calendar")
    calendar_paths: List[str] = s3.glob(path=f"{s3_bucket}/{calendar_path}")
    calendar = pd.concat([pd.read_csv(f"s3://{path}")
                          for path in calendar_paths if path.endswith('.csv')])
    calendar['date_utc'] = pd.to_datetime(calendar['date_utc'], format='%Y-%m-%d %H:%M:%S')

    start_range = ingest_date + pd.DateOffset(days=6)  # first 6 days were requested in previous dates, to avoid overlap
    end_range = ingest_date + pd.DateOffset(days=11)  # full 10 days as far as last day is included
    calendar = calendar[(calendar['date_utc'] > start_range) & (calendar['date_utc'] < end_range)]

    calendar = calendar[~calendar.event_id.isnull()]
    logger.info(f'Calendar shape without events where event_id is NaN {calendar.shape}')
    logger.info(f'Calendar min and max dates: {calendar.date_utc.min()}, {calendar.date_utc.max()}')

    events = calendar['name']
    print(f"Ingesting {sport_name}")
    if len(events) > 0:
        output_data = None
        for event_name in events:
            print(f"ingesting {event_name}")
            future_calendar = calendar.query(f'name == "{event_name}"')
            league = future_calendar['league'].values[0]

            future_calendar['netbase_topic_id'] = TOPIC_ID[sport_name]
            future_calendar[
                'netbase_sources'] = """Twitter,Facebook,Instagram,Tumblr,YouTube,Blogs,Comments,Forums,Internal,
                                        Microblogs,News,ConsumerReviews,ProfReviews,SocialNetworks,Other"""
            future_calendar['netbase_days_before_event'] = 7
            future_calendar['netbase_days_in_measured_period'] = 10
            #         future_calendar['competitions'] = future_calendar['competitions'].apply(json.loads)

            future_calendar['netbase_keywords'] = future_calendar.apply(get_keywords, args=[league], axis=1)
            future_calendar['netbase_keywords'] = future_calendar['netbase_keywords'] \
                .apply(lambda x: [re.sub('([^\w\s\._\d]|Else)', '', w)
                                  for w in x if re.sub('([^\w\s\._\d]|Else)', '', w)])
            future_calendar['ts'] = future_calendar['date_utc'].apply(convert_time)

            future_calendar = enrich_by_netbase_metric(future_calendar, ['Impressions', 'TotalBuzz'], httpsession)
            #         future_calendar['competitions'] = future_calendar['competitions'].apply(json.dumps)
            if output_data is None:
                output_data = future_calendar
            else:
                output_data = pd.concat([output_data, future_calendar], ignore_index=True)
        output_data = output_data[['id', 'date_utc', 'netbase_metric_impressions', 'netbase_metric_totalbuzz']]
        print("creating dir")
        output_data.to_csv("./netbase_metrics.csv", index=False)
        print("loading to s3")
        output_path = f"s3://{s3_bucket}/{netbase_metrics_location}/{sport_name}/{ingest_date.strftime('%Y-%m-%d')}/"
        subprocess.check_call(["aws", "s3", "cp", "./netbase_metrics.csv", output_path])
        put_success_file_to_s3(output_path)
    else:
        msg = f"There is no events in period [{ingest_date}, {end_range})"
        print(msg)
        raise AirflowSkipException(msg)
