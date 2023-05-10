import argparse
import subprocess
from typing import Dict, Optional

import s3fs
import json
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup

s3 = s3fs.S3FileSystem()

# Add rank manually
missed_mch = {
    'Franklin Pierce': 63,
    'Stonehill': 57,
}
DATE_PATTERN = "%Y-%m-%d"


def put_success_file_to_s3(full_folder_path: str,
                           success_file_name: str = "_SUCCESS"):
    if not full_folder_path.startswith("s3://"):
        full_folder_path = f"s3://{full_folder_path}"
    if not full_folder_path.endswith("/"):
        full_folder_path = f"{full_folder_path}/"
    response = subprocess.run(["aws", "s3", "cp", "-", f"{full_folder_path}{success_file_name}"], input="")
    response.check_returncode()
    return response


def tableDataText(table):
    """Parses a html segment started with tag <table> followed 
    by multiple <tr> (table rows) and inner <td> (table data) tags. 
    It returns a list of rows with inner columns. 
    Accepts only one <th> (table header/data) in the first row.
    """

    def rowgetDataText(tr, coltag='td'):  # td (data) or th (header)
        return [td.get_text(strip=True) for td in tr.find_all(coltag)]

    rows = []
    trs = table.find_all('tr')
    headerow = rowgetDataText(trs[0], 'th')
    if headerow:  # if there is a header row include first
        rows.append(headerow)
        trs = trs[1:]
    for tr in trs:  # for every table row
        rows.append(rowgetDataText(tr, 'td'))  # data row
    return rows


def parse_arguments():
    parser = argparse.ArgumentParser(description='Ingest data from DataDog')
    parser.add_argument(
        "--s3-bucket",
        help="DataDog API key",
    )
    parser.add_argument(
        "--mch-rankings-location",
        help="Location of MCH rankings on S3",
    )
    parser.add_argument(
        "--ingest-date",
        help="Ingest date in format Y-m-d (default: current day)",
        required=False,
        default=datetime.now().strftime(DATE_PATTERN),
    )
    parser.add_argument(
        "--hockey-mapping-path",
        help="Dictionaries with team_names mapping",
        required=True,
    ),
    return parser.parse_args()


def duplicate_with_another_name(rankings_df, mapping_dict):
    two_team_names_df = rankings_df[rankings_df.team_name.isin(mapping_dict.keys())]
    two_team_names_df = two_team_names_df.replace({'team_name': mapping_dict})

    rankings_with_two_names = pd.concat([rankings_df, two_team_names_df], ignore_index=True)

    return rankings_with_two_names


def ingest_one_year(season_end_year: int,
                    team_name_mapping: Dict[str, str]) -> Optional[pd.DataFrame]:
    header = {'User-agent': 'Mozilla/5.0 (Windows; U; Windows NT 5.1; de; rv:1.9.1.5) Gecko/20091102 Firefox/3.5.5'}

    url = requests.get(f"https://www.collegehockeynews.com/ratings/krach/{season_end_year}", headers=header).text

    soup = BeautifulSoup(url, 'html.parser')

    table = soup.find('table')
    list_table = tableDataText(table)
    if list_table:
        table_header = list_table[0]
        if ['Rk', 'Team'] == table_header[:2]:
            df = pd.DataFrame(list_table[2:])[[0, 1]]
            df.columns = ['league_rank', 'team_name']
            df['year'] = season_end_year
            df['league_rank'] = df['league_rank'].astype(int)
            df = df[['team_name', 'year', 'league_rank']]

            # Replace team_names with names from calendar
            rankings_df = df.replace({"team_name": team_name_mapping})

            # Add missing rankings manually from missed_mch dict
            rows_to_add = []
            for k, v in missed_mch.items():
                if k not in rankings_df.team_name.values:
                    rows_to_add.append({'team_name': k, 'year': season_end_year, 'league_rank': v})

            return pd.concat([rankings_df, pd.DataFrame(rows_to_add)], ignore_index=True)
    return None


if __name__ == "__main__":
    arguments = parse_arguments()
    s3_bucket: str = arguments.s3_bucket
    mch_rankings_location: str = arguments.mch_rankings_location
    ingest_date_str = arguments.ingest_date
    ingest_date: datetime = datetime.strptime(ingest_date_str, DATE_PATTERN)
    hockey_mapping_path = arguments.hockey_mapping_path

    full_hockey_mapping_path = f"s3://{s3_bucket}/{hockey_mapping_path}"
    with s3.open(full_hockey_mapping_path) as f:
        hockey_dicts = json.load(f)
    mapping = hockey_dicts['mapping']
    two_names_in_calendar = hockey_dicts['two_names_in_calendar']

    years_to_ingest = range(2020, ingest_date.year + 1)
    data = [ingest_one_year(y, mapping) for y in years_to_ingest]
    rankings = pd.concat(data, ignore_index=True)
    ranking_extended = duplicate_with_another_name(rankings, two_names_in_calendar)

    ranking_extended.to_csv(f'./krach_rankings.csv', index=False)
    output_path = f"s3://{s3_bucket}/{mch_rankings_location}/{ingest_date_str}/"
    subprocess.check_call(["aws", "s3", "cp", f'./krach_rankings.csv', output_path])
    put_success_file_to_s3(output_path)
