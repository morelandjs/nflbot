#!/usr/bin/env python3
import json
import os
import requests

import numpy as np
import pandas as pd

from model import EloraNFL
from data import upcoming_games


def rank(spread_model, total_model, datetime, slack_report=False):
    """
    Rank NFL teams at a certain point in time. The rankings are based on all
    available data preceding that moment in time.
    """
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

    if slack_report is True:
        requests.post(
            os.getenv('SLACK_WEBHOOK'),
            data=json.dumps({'text': report}),
            headers={'Content-Type': 'application/json'})

    print(report)


def forecast(spread_model, total_model, games, slack_report=False):
    """
    Forecast outcomes for the list of games specified.
    """
    games = pd.DataFrame(
        [g for g in upcoming_games(2020)],
        columns=['date', 'season', 'week', 'team_away', 'team_home'])

    season = np.squeeze(games.season.unique())
    week = np.squeeze(games.week.unique())

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

    if slack_report is True:
        requests.post(
            os.getenv('SLACK_WEBHOOK'),
            data=json.dumps({'text': report}),
            headers={'Content-Type': 'application/json'})

    print(report)


if __name__ == '__main__':
    from data import games

    spread_model = EloraNFL.from_cache(games, 'spread')
    total_model = EloraNFL.from_cache(games, 'total')

    time = pd.Timestamp.now()
    slack_report = False

    rank(spread_model, total_model, time,
         slack_report=slack_report)

    forecast(spread_model, total_model, upcoming_games(days=7),
             slack_report=slack_report)
