import json
import subprocess
from datetime import datetime
from datetime import timedelta

from ratelimit import sleep_and_retry, RateLimitException

NETBASE_TOKEN = 'your_netbase_token'

def put_success_file_to_s3(full_folder_path: str,
                           success_file_name: str = "_SUCCESS"):
    if not full_folder_path.startswith("s3://"):
        full_folder_path = f"s3://{full_folder_path}"
    if not full_folder_path.endswith("/"):
        full_folder_path = f"{full_folder_path}/"
    response = subprocess.run(["aws", "s3", "cp", "-", f"{full_folder_path}{success_file_name}"], input="")
    response.check_returncode()
    return response

def get_social_metrics(httpsession, row, metric_names):
    days_before_event = row['netbase_days_before_event']
    days_in_measured_period = row['netbase_days_in_measured_period']

    end_date = row['ts'] - timedelta(days=days_before_event)
    start_date = end_date - timedelta(days=days_before_event + days_in_measured_period)

    keywords = row['netbase_keywords']
    topic_id = row['netbase_topic_id']
    sources = row['netbase_sources'].split(',')

    response = make_call_to_netbase_api(httpsession, metric_names, start_date, end_date, keywords, topic_id, sources)
    return json.loads(response.content.decode('utf-8'))


def make_call_to_netbase_api(httpsession, metric_series, start_date, end_date, keywords, topic_ids, sources,
                             hostname='https://api.netbase.com/cb/insight-api/2', endpoint='metricValues'):
    metric_series_arg = '&metricSeries='.join(metric_series)
    keywords_arg = '&keywords='.join(keywords)
    sources_arg = '&sources='.join(sources)

    url = f'{hostname}/{endpoint}?topicIds={topic_ids}&metricSeries={metric_series_arg}&sources={sources_arg}&publishedDate={start_date}&publishedDate={end_date}&keywords={keywords_arg}'
    print('URL: ' + url)
    return call_netbase_api(url, httpsession)


@sleep_and_retry
def call_netbase_api(url, httpsession):
    response = httpsession.get(url)
    print('HEADERS: ', response.headers)

    if response.status_code == 403:
        if response.headers['ERROR_MESSAGE'] == 'Rate limit exceeded':
            print('Rate limit exceeded. Sleeping for {} sec.'.format(response.headers['X-RateLimit-Reset']))
            raise RateLimitException("Rate limit exceeded" + str(response.headers['X-RateLimit-Remaining']),
                                     int(response.headers['X-RateLimit-Reset']))

    if response.status_code != 200:
        raise Exception('API response: {}\n URL:{}'.format(response.status_code, url))

    return response


def enrich_by_netbase_metric(df, metric_names, httpsession):
    new_df = df.copy()
    new_df['netbase_metrics'] = new_df.apply(lambda x: get_social_metrics(httpsession, x, metric_names), axis=1)
    new_df['netbase_start_date'] = new_df['netbase_metrics'].apply(
        lambda x: datetime.fromtimestamp(int(x.get('startDate')) / 1000))
    new_df['netbase_end_date'] = new_df['netbase_metrics'].apply(
        lambda x: datetime.fromtimestamp(int(x.get('endDate')) / 1000))

    def extract_metrics(row):
        for dataset in row['netbase_metrics'].get('metrics')[0].get('dataset'):
            row['netbase_metric_' + dataset.get('seriesName').lower()] = dataset.get('set')[0]
        return row

    new_df = new_df.apply(extract_metrics, axis=1)
    new_df.drop(['netbase_metrics'], axis=1, inplace=True)
    return new_df

def convert_time(datetime_string):
    import dateutil.parser
    return dateutil.parser.parse(str(datetime_string)) # TODO: remove it?