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

from hyperopt import fmin, hp, tpe, Trials
import numpy as np
import pandas as pd
import pendulum
import prefect
from prefect.schedules import IntervalSchedule
from prefect import Flow, Parameter, task
from sportsreference.nfl.schedule import Schedule as NFLSchedule
from sportsreference.nfl.teams import Teams as NFLTeams

from model import EloraTeam


# remap sportsreference abbreviations
alias = {
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


def get_season_team(season_team):
    """
    Get all game scores for the specified season and team
    """
    logger = prefect.context.get('logger')

    season, team = season_team

    logger.info(f'syncing {season} {team}')

    cachedir = Path('~/.local/share/sportsref').expanduser()
    cachedir.mkdir(exist_ok=True, parents=True)

    cachefile = cachedir / f'{season}_{team}.pkl'

    if cachefile.exists() and season < 2020:
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

    columns = {
        'datetime': 'date',
        'season': 'season',
        'week': 'week',
        'team_home': 'team_home',
        'team_away': 'team_away',
        'score_home': 'score_home',
        'score_away': 'score_away'}

    df = df[columns.keys()].rename(columns=columns)

    df.to_pickle(cachefile)

    return df


@task
def concatenate_games(season_team_tuples):
    """
    Concatenate list of dataframes along first axis
    """
    df_list = [
            get_season_team(season_team)
            for season_team in season_team_tuples]

    df = pd.concat(
        df_list, axis=0
    ).drop_duplicates(
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
def calibrate_model(games, mode):
    """
    Optimizes the EloraTeam model hyperparameters. Returns trained model
    instance using calibrated hyperparameters.
    """
    games = games.dropna(axis=0, how='any').copy()

    limits = {
        "spread": [
            ("kfactor",      0.02, 0.12),
            ("regress_frac",  0.0,  1.0),
            ("rest_coeff",  -0.50, 0.75)],
        "total": [
            ("kfactor",      0.01, 0.07),
            ("regress_frac",  0.0,  1.0),
            ("rest_coeff",   -0.5,  0.5)]}

    space = [hp.uniform(*lim) for lim in limits[mode]]

    trials = Trials()

    def calibrate_loc_params(params):
        return EloraTeam(games, mode, *params).mean_abs_error

    parameters = fmin(
        calibrate_loc_params, space, algo=tpe.suggest,
        max_evals=200, trials=trials, show_progressbar=True)

    kfactor = parameters['kfactor']
    regress_frac = parameters['regress_frac']
    rest_coeff = parameters['rest_coeff']

    limits = {
        "spread": [
            ("scale", 5., 25.)],
        "total": [
            ("scale", 5., 25.)]}

    space = [hp.uniform(*lim) for lim in limits[mode]]

    trials = Trials()

    def calibrate_scale_param(params):
        return EloraTeam(
            games, mode, kfactor, regress_frac, rest_coeff, scale=params[0]
        ).log_loss

    parameters = fmin(
        calibrate_scale_param, space, algo=tpe.suggest,
        max_evals=50, trials=trials, show_progressbar=True)

    scale = parameters['scale']

    logger = prefect.context.get('logger')

    best_fit_params = ' '.join([
        f'k={kfactor}',
        f'regress_frac={regress_frac}',
        f'rest_coeff={rest_coeff}',
        f'scale={scale}'])

    logger.info(f'best fit params: {best_fit_params}')

    elora_team = EloraTeam(
        games, mode, kfactor, regress_frac, rest_coeff, scale=scale)

    return elora_team


@task
def rank(spread_model, total_model, datetime):
    """
    Rank NFL teams at a certain point in time. The rankings are based on all
    available data preceding that moment in time.
    """
    logger = prefect.context.get('logger')
    logger.info('Rankings as of {}\n'.format(datetime))

    df = pd.DataFrame(
        spread_model.rank(datetime, order_by='mean', reverse=True),
        columns=['team', 'point spread']
    ).replace(alias)

    df['point spread'] = df[['point spread']].applymap('{:4.1f}'.format)
    spread_col = '  ' + df['team'] + '  ' + df['point spread']

    df = pd.DataFrame(
        total_model.rank(datetime, order_by='mean', reverse=True),
        columns=['team', 'point total']
    ).replace(alias)

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

    requests.post(
        os.getenv('SLACK_WEBHOOK'),
        data=json.dumps({'text': report}),
        headers={'Content-Type': 'application/json'})

    print(report)


@task
def upcoming_week(games):
    """
    Returns a dataframe of games to be played in the upcoming week
    """
    upcoming_games = games[
        games.score_home.isnull() & games.score_away.isnull()]

    upcoming_week = upcoming_games[
        (upcoming_games.week == upcoming_games.week.min())
    ].copy()

    return upcoming_week


@task
def forecast(spread_model, total_model, games):
    """
    Forecast outcomes for the list of games specified.
    """
    season = games.season.unique().item()
    week = games.week.unique().item()

    logger = prefect.context.get('logger')
    logger.info(f"Forecast for season {season} week {week}")

    report = pd.DataFrame({
        "fav": games.team_away.replace(alias),
        "und": "@" + games.team_home.replace(alias),
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

    slack_webhook = os.getenv('SLACK_WEBHOOK')

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


with Flow('deploy nfl model predictions', schedule) as flow:

    current_season = Parameter('current_season', default=2020)

    season_team_tuples = iter_product(seasons(current_season), team_names)

    games = concatenate_games(season_team_tuples)

    games = compute_rest_days(games)

    spread_model = calibrate_model(games, 'spread')

    total_model = calibrate_model(games, 'total')

    rank(spread_model, total_model, pd.Timestamp.now())

    forecast(spread_model, total_model, upcoming_week(games))


if __name__ == '__main__':

    flow.register(project_name='nflbot')

    # flow.run(current_season=2020)
