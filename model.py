"""Trains team model and exposes predictor class objects"""
from functools import partial
import operator

from elora import Elora
import numpy as np


class EloraTeam(Elora):
    """
    Generate NFL team point-spread or point-total predictions
    using the Elo regressor algorithm (elora)
    """
    def __init__(self, games, mode, kfactor, regress_frac, rest_coeff,
                 scale=1, burnin=512):

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
        """
        Regress ratings to the mean as a function of elapsed time.

        Regression fraction equals:

            self.regress_frac if elapsed_days > 90, else 1
        """
        elapsed_days = elapsed_time / np.timedelta64(1, 'D')

        tiny = 1e-6
        arg = np.clip(self.regress_frac, tiny, 1 - tiny)
        factor = np.log(arg)/365.

        return np.exp(factor * elapsed_days)

    def bias(self, games):
        """
        Circumstantial bias factors which apply to a single game.
        """
        rest_adv = self.rest_coeff * self.compare(
            games.rest_days_away, games.rest_days_home)

        # TODO add QB corrections

        return rest_adv

    def train(self, games):
        """
        Trains the Margin Elo (MELO) model on the historical game data.
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
        """
        Rank labels at specified 'time' according to 'order_by'
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
