#!/usr/bin/env python3
"""Source latest NFL game data and store in a sqlite database
"""
from datetime import datetime, timedelta
import itertools
from pathlib import Path
from urllib.error import HTTPError

import numpy as np
import pandas as pd
import pendulum
import prefect
from prefect import Flow, Parameter, task
from prefect.schedules import IntervalSchedule
from prefect.run_configs import LocalRun
from sportsipy.nfl.boxscore import Boxscore, Boxscores
import sqlalchemy

db_path = Path('~/.local/share/sportsref/nfl.db').expanduser()
db_path.parent.mkdir(parents=True, exist_ok=True)

engine = sqlalchemy.create_engine(f'sqlite:///{db_path.expanduser()}')


@task
def initialize_database():
    """Initialize a sqlite database, and create a table with the correct
    schema if it doesn't already exist. Returns the database connection engine.
    """
    engine.execute(
        """CREATE TABLE IF NOT EXISTS games (
                          gid      TEXT,
                     datetime  DATETIME,
                         date      TEXT,
                       season    BIGINT,
                         week    BIGINT,
                     duration      TEXT,
                   attendance    BIGINT,
                       winner      TEXT,
                  losing_name      TEXT,
                 winning_name      TEXT,
                  losing_abbr      TEXT,
                 winning_abbr      TEXT,
                   over_under      TEXT,
                   vegas_line      TEXT,
                         roof      TEXT,
                      stadium      TEXT,
                      surface      TEXT,
                         time      TEXT,
                      weather      TEXT,
                     won_toss      TEXT,
             away_first_downs    BIGINT,
    away_fourth_down_attempts    BIGINT,
 away_fourth_down_conversions    BIGINT,
                 away_fumbles    BIGINT,
            away_fumbles_lost    BIGINT,
           away_interceptions    BIGINT,
          away_net_pass_yards    BIGINT,
           away_pass_attempts    BIGINT,
        away_pass_completions    BIGINT,
         away_pass_touchdowns    BIGINT,
              away_pass_yards    BIGINT,
               away_penalties    BIGINT,
                  away_points    BIGINT,
           away_rush_attempts    BIGINT,
         away_rush_touchdowns    BIGINT,
              away_rush_yards    BIGINT,
     away_third_down_attempts    BIGINT,
  away_third_down_conversions    BIGINT,
      away_time_of_possession      TEXT,
            away_times_sacked    BIGINT,
             away_total_yards    BIGINT,
               away_turnovers    BIGINT,
    away_yards_from_penalties    BIGINT,
   away_yards_lost_from_sacks    BIGINT,
             home_first_downs    BIGINT,
    home_fourth_down_attempts    BIGINT,
 home_fourth_down_conversions    BIGINT,
                 home_fumbles    BIGINT,
            home_fumbles_lost    BIGINT,
           home_interceptions    BIGINT,
          home_net_pass_yards    BIGINT,
           home_pass_attempts    BIGINT,
        home_pass_completions    BIGINT,
         home_pass_touchdowns    BIGINT,
              home_pass_yards    BIGINT,
               home_penalties    BIGINT,
                  home_points    BIGINT,
           home_rush_attempts    BIGINT,
         home_rush_touchdowns    BIGINT,
              home_rush_yards    BIGINT,
     home_third_down_attempts    BIGINT,
  home_third_down_conversions    BIGINT,
      home_time_of_possession      TEXT,
            home_times_sacked    BIGINT,
             home_total_yards    BIGINT,
               home_turnovers    BIGINT,
    home_yards_from_penalties    BIGINT,
   home_yards_lost_from_sacks    BIGINT,
                            UNIQUE(gid))""")

    logger = prefect.context.get('logger')
    logger.info('successfully connected to database')

    return engine


@task
def update_database(engine, current_season):
    """Pull box score data for the specified game and store in the
    sqlite database.
    """
    latest_data = pd.read_sql(
        "SELECT DISTINCT season, week FROM games "
        "ORDER BY season DESC, week DESC LIMIT 1",
        engine
    ).squeeze()

    start_season, start_week = latest_data.values

    seasons = range(start_season, current_season + 1)
    weeks = range(1, 22)

    logger = prefect.context.get('logger')

    for season, week in itertools.product(seasons, weeks):
        if (season, week) < (start_season, start_week):
            continue

        try:
            boxscores_list = Boxscores(week, season).games.values()
        except HTTPError:
            continue

        for b in itertools.chain.from_iterable(boxscores_list):
            try:
                gid = b['boxscore']
                logger.info(f'syncing {gid}')

                df = Boxscore(gid).dataframe
                df['gid'] = gid
                df['season'] = season
                df['week'] = week

                df.to_sql('games', engine, if_exists='append', index=False)
            except TypeError:
                logger.info(f'{gid} data not yet available')
                continue
            except sqlalchemy.exc.IntegrityError:
                logger.info(f'{gid} already stored in database')
                continue


def preprocess_data():
    """Preprocess historical NFL boxscore data into a form that is suitable for
    modelling.

    Returns:
        Pandas dataframe of NFL game data
    """
    df = pd.read_sql("""
    SELECT
        datetime,
        season,
        week,
        winner,
        winning_abbr,
        winning_name,
        losing_abbr,
        losing_name,
        over_under,
        vegas_line,
        away_points,
        home_points
        FROM games""", engine)

    df['team_home'] = np.where(
        df.winner == 'Home', df.winning_name, df.losing_name)

    df['team_away'] = np.where(
        df.winner == 'Away', df.winning_name, df.losing_name)

    df['vegas_favorite'] = df.vegas_line.str.replace(
        'Pick', '0.0').str.split().str[:-1].str.join(' ').str.strip()

    df['vegas_favorite_line'] = df.vegas_line.str.replace(
       'Pick', '0.0').str.split().str[-1].astype(float)

    df['vegas_home_line'] = np.where(
        df.vegas_favorite == df.team_home,
        df.vegas_favorite_line, -df.vegas_favorite_line)

    df['vegas_over_under'] = df.over_under.str.split().str[0].astype(float)

    df['team_home'] = np.where(
        df.winner == 'Home', df.winning_abbr, df.losing_abbr)

    df['team_away'] = np.where(
        df.winner == 'Away', df.winning_abbr, df.losing_abbr)

    column_aliases = {
        'datetime': 'date',
        'season': 'season',
        'week': 'week',
        'team_away': 'team_away',
        'team_home': 'team_home',
        'away_points': 'score_away',
        'home_points': 'score_home',
        'vegas_home_line': 'vegas_home_line',
        'vegas_over_under': 'vegas_over_under'}

    df = df[
        column_aliases.keys()
    ].rename(
        column_aliases, axis=1
    ).replace({
        'ATL': 'ATL',
        'BUF': 'BUF',
        'CAR': 'CAR',
        'CHI': 'CHI',
        'CIN': 'CIN',
        'CLE': 'CLE',
        'CLT': 'IND',
        'CRD': 'ARI',
        'DAL': 'DAL',
        'DEN': 'DEN',
        'DET': 'DET',
        'GNB': 'GB',
        'HTX': 'HOU',
        'JAX': 'JAX',
        'KAN': 'KC',
        'MIA': 'MIA',
        'MIN': 'MIN',
        'NOR': 'NO',
        'NWE': 'NE',
        'NYG': 'NYG',
        'NYJ': 'NYJ',
        'OTI': 'TEN',
        'PHI': 'PHI',
        'PIT': 'PIT',
        'RAI': 'LV',
        'RAM': 'LAR',
        'RAV': 'BAL',
        'SDG': 'LAC',
        'SEA': 'SEA',
        'SFO': 'SF',
        'TAM': 'TB',
        'WAS': 'WAS'
    }).astype(
        {'date': 'datetime64[s]'}
    ).sort_values(
        by=['date', 'team_away', 'team_home'])

    return df


def upcoming_games(days=7):
    """Returns a dataframe of upcoming games played in season `season`
    and within `days` days of now.
    """
    now = datetime.now()
    season = now.year if now.month > 6 else now.year - 1

    for week in range(1, 22):
        try:
            boxscores_list = Boxscores(week, season).games.values()
        except HTTPError:
            continue

        for b in itertools.chain.from_iterable(boxscores_list):
            date = datetime.strptime(b['boxscore'][:8], '%Y%m%d')
            away = b['away_abbr'].upper()
            home = b['home_abbr'].upper()
            if 0 <= (date - now).days <= days:
                yield (date, season, week, away, home)


games = preprocess_data()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='update NFL gamedata')

    parser.add_argument(
        '--schedule', help='register the prefect flow with its backend',
        action='store_true')

    args = parser.parse_args()

    tz = 'America/New_York'

    schedule = IntervalSchedule(
        start_date=pendulum.datetime(2020, 12, 2, 8, 0, tz=tz),
        interval=timedelta(days=7),
        end_date=pendulum.datetime(2021, 2, 3, 8, 0, tz=tz))

    with Flow('update NFL game data', schedule=schedule) as flow:
        current_season = Parameter('current_season', default=2020)
        update_database(initialize_database(), current_season)

    flow.run_config = LocalRun(working_dir=Path.cwd())

    if args.schedule is True:
        flow.register(project_name='nflbot')

    flow.run(current_season=2020, run_on_schedule=args.schedule)
