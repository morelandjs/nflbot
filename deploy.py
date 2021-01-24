#!/usr/bin/env python3
"""
Source latest NFL game data, train an Elo regressor, generate a league
ranking and forecast report for the upcoming week, and post these reports
to a Slack webhook.
"""
import datetime
import itertools
import json
import os
from pathlib import Path
import requests
from urllib.error import HTTPError

import numpy as np
import optuna
import pandas as pd
import pendulum
import prefect
from prefect import Flow, Parameter, task
from prefect.schedules import IntervalSchedule
from prefect.run_configs import LocalRun
from sportsipy.nfl.boxscore import Boxscore, Boxscores
import sqlalchemy

from model import EloraTeam

team_aliases = {
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
    'WAS': 'WAS'}


@task
def create_engine():
    """Initialize a sqlite database and create a table with the correct
    schema if it doesn't already exist. Returns the database connection engine.
    """
    db_path = Path.cwd() / 'games.db'
    engine = sqlalchemy.create_engine(f'sqlite:///{db_path}')

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
def update_database(engine, current_season, debug=False):
    """Pull box score data for the specified game and store in the
    SQL database
    """
    logger = prefect.context.get('logger')

 latest_data    latest_data = pd.read_sql(
        "SELECT DISTINCT season, week FROM games "
        "ORDER BY season DESC, week DESC LIMIT 1",
        engine
    ).squeeze()

    start_season, start_week = latest_data.values

    seasons = range(start_season, current_season + 1)
    weeks = range(1, 22)

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


@task
def modelling_data(engine):
    """Table of historical NFL boxscore data and betting lines
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
    ).replace(
        team_aliases
    ).astype(
        {'date': 'datetime64[s]'}
    ).sort_values(
        by=['date', 'team_away', 'team_home'])

    df.insert(1, 'date_line', df.date - pd.Timedelta(hours=1))

    return df


def calibrate_model(games, mode, n_trials=100, debug=False):
    """Optimizes and returns elora distribution mean parameters.

    Games dataframe must have the following columns:
      * date
      * team_away
      * team_home
      * value
    """
    logger = prefect.context.get('logger')
    logger.info(f'calibrating {mode} mean parameters')

    def objective(trial):
        """
        hyperparameter objective function
        """
        kfactor = trial.suggest_loguniform('kfactor', 1e-4, 1e-2)
        regress_frac = trial.suggest_uniform('regress_frac', 0.0, 1.0)
        rest_coeff = trial.suggest_uniform('rest_coeff', -0.2, 0.2)

        elora_team = EloraTeam(games, mode, kfactor, regress_frac, rest_coeff)

        return elora_team.mean_abs_error

    study = optuna.create_study()
    study.optimize(objective, n_trials=(n_trials if debug is False else 2))

    kfactor = study.best_params['kfactor']
    regress_frac = study.best_params['regress_frac']
    rest_coeff = study.best_params['rest_coeff']

    residuals = EloraTeam(
        games, mode, kfactor, regress_frac, rest_coeff
    ).residuals()

    scale = residuals.std()

    logger.info(f'using scale = {scale}')

    return EloraTeam(
        games, mode, kfactor, regress_frac, rest_coeff, scale=scale)


@task
def train_model(games, mode, n_trials=100):
    """Time series training data of market predictions
    """
    observed, predicted = {
        'spread': (games.score_away - games.score_home, games.vegas_home_line),
        'total': (games.score_away + games.score_home, games.vegas_over_under)
    }[mode]

    residuals = observed - predicted

    comparisons = pd.DataFrame({
        'date': games.date,
        'team_away': games.team_away,
        'team_home': games.team_home,
        'value': residuals})

    print(comparisons)

    model = calibrate_model(comparisons, mode, n_trials=n_trials)

    return model


@task
def gamble(games, spread_model, threshold=0.75):
    """Find profitable bets
    """
    games['line_residual'] = spread_model.mean(
        games.date, games.team_away, games.team_home)

    games = games[games.line_residual.abs() > threshold]

    y_obs = games.score_away - games.score_home
    y_pred = games.vegas_home_line

    away_cover = (y_obs - y_pred > 0.)

    bet_away = (games.line_residual > 0.)

    success = (away_cover == bet_away)

    print(games.to_string())
    print(success.sum(axis=0), len(games))


@task
def rank(spread_model, total_model, datetime, debug=False):
    """
    Rank NFL teams at a certain point in time. The rankings are based on all
    available data preceding that moment in time.
    """
    logger = prefect.context.get('logger')
    logger.info('Rankings as of {}\n'.format(datetime))

    df = pd.DataFrame(
        spread_model.rank(datetime, order_by='mean', reverse=True),
        columns=['team', 'point spread'])

    df['point spread'] = df[['point spread']].applymap('{:4.1f}'.format)
    spread_col = '  ' + df['team'] + '  ' + df['point spread']

    df = pd.DataFrame(
        total_model.rank(datetime, order_by='mean', reverse=True),
        columns=['team', 'point total'])

    df['point total'] = df[['point total']].applymap('{:4.1f}'.format)
    total_col = '  ' + df['team'] + '  ' + df['point total']

    rankings = pd.concat([spread_col, total_col], axis=1).round(decimals=1)
    rankings.columns = [6*' ' + 'spread', 7*' ' + 'total']
    rankings.index += 1

    rankings.index.name = 'rank'
    timestamp = datetime.floor('Min')

    report = '\n'.join([
        f'*RANKING  |  @{timestamp}*\n',
        '```',
        rankings.to_string(header=True, justify='left'),
        '```',
        '*against average team on neutral field'])

    if debug is True:
        requests.post(
            os.getenv('SLACK_WEBHOOK'),
            data=json.dumps({'text': report}),
            headers={'Content-Type': 'application/json'})

    print(report)


def upcoming_games(season, days=7):
    """Returns a dataframe of games to be played in the upcoming week
    """
    now = datetime.datetime.now()

    for week in range(1, 22):
        try:
            boxscores_list = Boxscores(week, season).games.values()
        except HTTPError:
            continue

        for b in itertools.chain.from_iterable(boxscores_list):
            date = datetime.datetime.strptime(b['boxscore'][:8], '%Y%m%d')
            away = b['away_abbr'].upper()
            home = b['home_abbr'].upper()
            if 0 <= (date - now).days <= days:
                yield (date, season, week, away, home)


@task
def forecast(spread_model, total_model, debug=False):
    """Forecast outcomes for the list of games specified.
    """
    games = pd.DataFrame(
        [g for g in upcoming_games(2020)],
        columns=['date', 'season', 'week', 'team_away', 'team_home'])

    if games.empty:
        logger = prefect.context.get('logger')
        logger.warning('cannot find upcoming game schedule')
        return

    season = np.squeeze(games.season.unique())
    week = np.squeeze(games.week.unique())

    logger = prefect.context.get('logger')
    logger.info(f"Forecast for season {season} week {week}")

    report = pd.DataFrame({
        "fav": games.team_away,
        "und": "@" + games.team_home,
        "odds": spread_model.sf(
            0, games.date, games.team_away, games.team_home),
        "spread": spread_model.mean(
            games.date, games.team_away, games.team_home),
        "total": total_model.mean(
            games.date, games.team_away, games.team_home)
    }).round({'spread': 1, 'total': 1})

    report[["fav", "und"]] = report[["und", "fav"]].where(
        report["odds"] < 0.5, report[["fav", "und"]].values)

    report["spread"] = report["spread"].where(
        report["odds"] < 0.5, -report["spread"].values)

    report["one minus odds"] = 1 - report["odds"]
    report["odds"] = report[["odds", "one minus odds"]].max(axis=1)
    report["odds"] = (100*report.odds).astype(int).astype(str) + '%'

    report.drop(columns="one minus odds", inplace=True)
    report.sort_values('spread', inplace=True)

    report = '\n'.join([
        f'*FORECAST  |  SEASON {season}, WEEK {week}*\n',
        '```',
        report.to_string(index=False),
        '```'])

    if debug is True:
        requests.post(
            os.getenv('SLACK_WEBHOOK'),
            data=json.dumps({'text': report}),
            headers={'Content-Type': 'application/json'})

    print(report)


# run every Wednesday at 8 am EST
schedule = IntervalSchedule(
    start_date=pendulum.datetime(2020, 12, 2, 8, 0, tz="America/New_York"),
    interval=datetime.timedelta(days=7),
    end_date=pendulum.datetime(2021, 2, 3, 8, 0, tz="America/New_York"))


# with Flow('deploy nfl model predictions', schedule) as flow:
with Flow('deploy nfl model predictions') as flow:

    current_season = Parameter('current_season', default=2020)

    debug = Parameter('debug', default=False)

    engine = create_engine()

    update_database(engine, current_season, debug=debug)

    games = modelling_data(engine)

    spread_model = train_model(games, 'spread')

    # total_model = train_model(games, 'total')

    # gamble(games, spread_model)

    # rank(spread_model, total_model, pd.Timestamp.now(), debug=debug)

    # forecast(spread_model, total_model, debug=debug)

flow.run_config = LocalRun(
    working_dir="/home/morelandjs/nflbot")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Slack NFL predictions bot')

    parser.add_argument(
        '--debug', help='run in debug mode', action='store_true')
    args = parser.parse_args()
    kwargs = vars(args)

    # flow.register(project_name='nflbot')

    flow.run(current_season=2020, **kwargs)
