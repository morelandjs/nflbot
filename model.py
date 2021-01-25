#!/usr/bin/env python3
"""Trains team model and exposes predictor class objects"""
from functools import partial
import operator
from pathlib import Path
import pickle

from elora import Elora
import numpy as np
import optuna
import pandas as pd


class EloraNFL(Elora):
    def __init__(self, games, mode, kfactor, regress_frac, rest_coeff,
                 scale=1, burnin=512):
        """Generate NFL point-spread or point-total predictions using the Elo
        regressor algorithm (elora).

        Args:
            games (pd.DataFrame): pandas dataframe containing comparisons of
                the form (date, team_home, team_away, score_away, score_home).
            mode (str): comparison type, equal to 'spread' or 'total'.
            kfactor (float): Elo hidden rating update factor
            regress_frac (float): one minus the fractional amount used to
                regress ratings to the mean each offseason
            rest_coeff (float): prefactor that modulates the strength of rest
                effects, i.e. how much better/worse a team plays as a function
                of days between games. values can be positive or negative.
            scale (float, optional): standard deviation of the elora regressor
                predictions. default value is 1.
            burnin (int, optional): number of games to ignore from the
                beginning of the games dataframe when computing performance
                metrics. default is 512.
        """

        # training data
        self.games = games

        # hyperparameters
        self.mode = mode
        self.kfactor = kfactor
        self.regress_frac = regress_frac
        self.rest_coeff = rest_coeff
        self.scale = scale
        self.burnin = burnin

        # model operation mode: "spread" or "total"
        if self.mode not in ["spread", "total"]:
            raise ValueError(
                "Unknown mode; valid options are 'spread' and 'total'")

        # mode-specific training hyperparameters
        self.commutes, self.compare = {
            "total": (True, operator.add),
            "spread": (False, operator.sub),
        }[mode]

        # initialize base class with chosen hyperparameters
        super().__init__(
            self.kfactor,
            scale=self.scale,
            commutes=self.commutes)

        # train the model
        self.train(games)

        # compute performance metrics
        self.residuals_ = self.residuals(standardize=False)
        self.mean_abs_error = np.mean(np.abs(self.residuals_[burnin:]))
        self.rms_error = np.sqrt(np.mean(self.residuals_[burnin:]**2))

        # components for binary cross entropy loss
        tiny = 1e-5
        yp = np.clip(
            self.pdf(
                self.examples.value,
                self.examples.time,
                self.examples.label1,
                self.examples.label2,
                self.examples.bias),
            tiny, 1 - tiny)

        # binary cross entropy loss
        self.log_loss = -np.log(yp).mean()

    def regression_coeff(self, elapsed_time):
        """Regress ratings to the mean as a function of elapsed time.

        Regression fraction equals:

            self.regress_frac if elapsed_days > 90, else 1

        Args:
            elapsed_time (datetime.timedelta): elapsed time since last update

        Returns:
            coefficient used to regress a rating to its mean value
        """
        elapsed_days = elapsed_time / np.timedelta64(1, 'D')

        tiny = 1e-6
        arg = np.clip(self.regress_frac, tiny, 1 - tiny)
        factor = np.log(arg)/365.

        return np.exp(factor * elapsed_days)

    def compute_rest_days(self, games):
        """Compute rest days for home and away teams

        Args:
            games (pd.DataFrame): dataframe of NFL game records

        Returns:
            pd.DataFrame including rest day columns
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
            game_dates_team = game_dates.rename(columns={
                'date_prev': f'date_{team}_prev', 'team': f'team_{team}'})
            games = games.merge(game_dates_team, on=['date', f'team_{team}'])

        one_day = pd.Timedelta("1 days")

        games["rest_days_home"] = np.clip(
            (games.date - games.date_home_prev) / one_day, 3, 16).fillna(7)
        games["rest_days_away"] = np.clip(
            (games.date - games.date_away_prev) / one_day, 3, 16).fillna(7)

        return games.drop(columns=['date_home_prev', 'date_away_prev'])

    def bias(self, games):
        """Circumstantial bias factors which apply to a single game.

        Args:
            games (pd.DataFrame): dataframe of NFL game records

        Returns:
            pd.Series of game bias correction coefficients
        """
        games = self.compute_rest_days(games)

        rest_adv = self.rest_coeff * self.compare(
            games.rest_days_away, games.rest_days_home)

        # TODO add QB corrections

        return rest_adv

    def train(self, games):
        """Conditions the regressor on provided game data.

        Args:
            games (pd.DataFrame): dataframe of NFL game records
        """
        games.sort_values(by=['date', 'team_away', 'team_home'], inplace=True)

        games['value'] = self.compare(games.score_away, games.score_home)

        self.fit(
            games.date,
            games.team_away,
            games.team_home,
            games.value,
            biases=self.bias(games))

    def rank(self, time, order_by='mean', reverse=False):
        """Rank labels at specified 'time' according to 'order_by'
        comparison value.

        Args:
            time (np.datetime64): time to compute the ranking
            order_by (string, optional): options are 'mean' and 'win_prob'
                (default is 'mean')
            reverse (bool, optional): reverses the ranking order if true
                (default is False)

        Returns:
            ranked_labels (list of tuple): list of (label, value) pairs
        """
        value = {
            "win prob": partial(self.sf, 0),
            "mean": self.mean,
         }.get(order_by, None)

        if value is None:
            raise ValueError("no such comparison function")

        ranked_list = [
            (label, value(time, label, None, biases=-self.commutator))
            for label in self.labels]

        return sorted(ranked_list, key=lambda v: v[1], reverse=reverse)

    @classmethod
    def from_cache(cls, games, mode, n_trials=100, retrain=False):
        """Instantiate the regressor using cached hyperparameters if available,
        otherwise train and cache a new instance.

        Args:
            games (pd.DataFrame): dataframe of NFL game records
            mode (str): comparison type, equal to 'spread' or 'total'.
            n_trials (int, optional): number of optuna steps to use for
                hyperparameter optimization. default value is 100.
            retrain (bool, optional): load hyperparameters from cache if
                available and retrain is False, recalibrate hyperparameters
                otherwise. default is False.
        """
        cachefile = Path(
            f'~/.local/share/elora/elora_nfl_{mode}.pkl'
        ).expanduser()

        cachefile.parent.mkdir(parents=True, exist_ok=True)

        if not retrain and cachefile.exists():
            params = pickle.load(cachefile.open(mode='rb'))
            kfactor = params['kfactor']
            regress_frac = params['regress_frac']
            rest_coeff = params['rest_coeff']
            scale = params['scale']
            return cls(games, mode, kfactor, regress_frac, rest_coeff, scale)

        def objective(trial):
            """hyperparameter objective function
            """
            kfactor = trial.suggest_loguniform('kfactor', 0.01, 0.1)
            regress_frac = trial.suggest_uniform('regress_frac', 0.0, 1.0)
            rest_coeff = trial.suggest_uniform('rest_coeff', -0.5, 0.5)
            regressor = cls(games, mode, kfactor, regress_frac, rest_coeff)
            return regressor.mean_abs_error

        study = optuna.create_study()
        study.optimize(objective, n_trials=n_trials)

        params = study.best_params.copy()
        kfactor = params['kfactor']
        regress_frac = params['regress_frac']
        rest_coeff = params['rest_coeff']

        residuals = cls(
            games, mode, kfactor, regress_frac, rest_coeff
        ).residuals()

        scale = residuals.std()
        params.update({'scale': scale})

        pickle.dump(params, cachefile.open(mode='wb'))

        return cls(games, mode, kfactor, regress_frac, rest_coeff, scale)


if __name__ == '__main__':
    """Minimal example of how to use this module
    """
    from data import games

    nfl_spread = EloraNFL.from_cache(games, 'spread')

    rankings = pd.DataFrame(
        nfl_spread.rank(pd.Timestamp.now()),
        columns=['team', 'spread_against_average']
    ).sort_values(
        by='spread_against_average', ascending=False
    ).reset_index(drop=True)

    print(rankings)
