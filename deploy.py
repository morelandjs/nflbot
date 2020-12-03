#!/usr/bin/env python3
"""
Source latest NFL game data, train an Elo regressor, generate a league
ranking and forecast report for the upcoming week, and post these reports
to a Slack webhook.
"""
import datetime
import json
import os
from pathlib import Path
import requests
import time
import urllib

import numpy as np
import optuna
import pandas as pd
import pendulum
import prefect
from prefect import Flow, Parameter, task
from prefect.schedules import IntervalSchedule
from sportsreference.nfl.schedule import Schedule as NFLSchedule
from sportsreference.nfl.teams import Teams as NFLTeams

from model import EloraTeam


@task
def seasons(current_season):
    """
    Return list of available seasons
    """
    return [season for season in range(2002, current_season + 1)]


@task
def team_names():
    """
    Get team abbreviation -> team name mapping
    """
    return {t.abbreviation: t.name for t in NFLTeams()}


@task
def iter_product(list1, list2):
    """
    Compute iterable cross product of list1 and list2
    """
    return [(l1, l2) for l1 in list1 for l2 in list2]


def get_season_team(season_team, current_season):
    """
    Get all game scores for the specified season and team
    """
    logger = prefect.context.get('logger')

    season, team = season_team

    logger.info(f'syncing {season} {team}')

    cachedir = Path('~/.local/share/sportsref/nfl').expanduser()
    cachedir.mkdir(exist_ok=True, parents=True)

    cachefile = cachedir / f'{season}_{team}.pkl'

    if cachefile.exists() and season < current_season:
        return pd.read_pickle(cachefile)

    time.sleep(1)  # rate limit requests

    df = NFLSchedule(team, season).dataframe

    df['season'] = season
    df['team_abbr'] = team

    df['team_home'] = np.where(
        df.location == 'Home', team, df.opponent_abbr)
    df['team_away'] = np.where(
        df.location == 'Away', team, df.opponent_abbr)

    df['score_home'] = np.where(
        df.location == 'Home', df.points_scored, df.points_allowed)
    df['score_away'] = np.where(
        df.location == 'Away', df.points_scored, df.points_allowed)

    column_aliases = {
        'datetime': 'date',
        'season': 'season',
        'week': 'week',
        'team_home': 'team_home',
        'team_away': 'team_away',
        'score_home': 'score_home',
        'score_away': 'score_away'}

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

    df = df[
        column_aliases.keys()
    ].rename(
        columns=column_aliases
    ).replace(
        team_aliases
    ).drop_duplicates(
        subset=['date', 'team_home', 'team_away'])

    df.to_pickle(cachefile)

    return df


@task
def concatenate_games(season_team_tuples, current_season):
    """
    Concatenate list of dataframes along first axis
    """
    def games_gen(season_team_tuples):
        for season_team in season_team_tuples:
            try:
                yield get_season_team(season_team, current_season)
            except urllib.error.HTTPError:
                continue

    df_list = [games for games in games_gen(season_team_tuples)]

    df = pd.concat(
        df_list, axis=0
    ).drop_duplicates(
        subset=['date', 'team_home', 'team_away']
    ).sort_values(
        by=['date', 'team_away', 'team_home']
    ).reset_index(drop=True)

    return df


@task
def compute_rest_days(games):
    """
    Compute home and away teams days rested
    """
    game_dates = pd.concat([
        games[["date", "team_home"]].rename(
            columns={"team_home": "team"}),
        games[["date", "team_away"]].rename(
            columns={"team_away": "team"}),
    ]).sort_values(by="date")

    game_dates['date_prev'] = game_dates.date

    game_dates = pd.merge_asof(
        game_dates[['team', 'date']],
        game_dates[['team', 'date', 'date_prev']],
        on='date', by='team', allow_exact_matches=False)

    for team in ["home", "away"]:

        game_dates_team = game_dates.rename(
            columns={'date_prev': f'date_{team}_prev', 'team': f'team_{team}'})

        games = games.merge(game_dates_team, on=['date', f'team_{team}'])

    one_day = pd.Timedelta("1 days")

    games["rest_days_home"] = np.clip(
        (games.date - games.date_home_prev) / one_day, 3, 16).fillna(7)
    games["rest_days_away"] = np.clip(
        (games.date - games.date_away_prev) / one_day, 3, 16).fillna(7)

    return games


@task
def calibrate_model(games, mode, n_trials=100, debug=False):
    """
    Optimizes and returns elora distribution mean parameters.
    """
    logger = prefect.context.get('logger')
    logger.info(f'calibrating {mode} mean parameters')

    games = games.dropna(axis=0, how='any').copy()

    def objective(trial):
        """
        hyperparameter objective function
        """
        kfactor = trial.suggest_uniform('kfactor', 0.01, 0.1)
        regress_frac = trial.suggest_uniform('regress_frac', 0.0, 1.0)
        rest_coeff = trial.suggest_uniform('rest_coeff', -0.5, 0.5)

        elora_team = EloraTeam(games, mode, kfactor, regress_frac, rest_coeff)

        return elora_team.mean_abs_error

    study = optuna.create_study()
    study.optimize(objective, n_trials=(n_trials if debug is False else 2))

    kfactor = study.best_params['kfactor']
    regress_frac = study.best_params['regress_frac']
    rest_coeff = study.best_params['rest_coeff']

    scale = EloraTeam(
        games, mode, kfactor, regress_frac, rest_coeff
    ).residuals_.std()

    logger.info(f'using scale = {scale}')

    return EloraTeam(
        games, mode, kfactor, regress_frac, rest_coeff, scale=scale)


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

    if debug is False:
        requests.post(
            os.getenv('SLACK_WEBHOOK'),
            data=json.dumps({'text': report}),
            headers={'Content-Type': 'application/json'})

    print(report)


@task
def upcoming_games(games, days=7):
    """
    Returns a dataframe of games to be played in the upcoming week
    """
    upcoming_games = games[
        games.score_home.isnull() & games.score_away.isnull()]

    days_ahead = (upcoming_games.date - pd.Timestamp.now()).dt.days
    upcoming_games = upcoming_games[(0 <= days_ahead) & (days_ahead < days)]

    return upcoming_games.copy()


@task
def forecast(spread_model, total_model, games, debug=False):
    """
    Forecast outcomes for the list of games specified.
    """
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

    if debug is False:
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

    season_team_tuples = iter_product(seasons(current_season), team_names)

    games = concatenate_games(season_team_tuples, current_season)

    games = compute_rest_days(games)

    spread_model = calibrate_model(games, 'spread', debug=debug)

    total_model = calibrate_model(games, 'total', debug=debug)

    rank(spread_model, total_model, pd.Timestamp.now(), debug=debug)

    forecast(spread_model, total_model, upcoming_games(games), debug=debug)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Slack NFL predictions bot')

    parser.add_argument(
        '--debug', help='run in debug mode', action='store_true')
    args = parser.parse_args()
    kwargs = vars(args)

    # flow.register(project_name='nflbot')

    flow.run(current_season=2020, **kwargs)
