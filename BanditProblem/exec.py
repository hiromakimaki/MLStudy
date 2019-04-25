"""
Summary:
  Execute bandit games and show the results.

Usage:
  python exec.py

Requirements:
  numpy, pandas

Reference:
  Chapter 3 of "バンディット問題の理論とアルゴリズム（機械学習プロフェッショナルシリーズ）"(https://www.kspub.co.jp/book/detail/1529175.html)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(0)


class BernoulliBandit:
    def __init__(self, probs):
        self._probs = probs
    
    @property
    def num_arms(self):
        return len(self._probs)

    def get_rewards(self, arm_index, num=1):
        mu = self._probs[arm_index]
        return np.random.choice([0, 1], num, p=[1 - mu, mu])


class Player:
    def __init__(self, strategy):
        self._arms = None
        self._rewards = None
        self._strategy = strategy

    @property
    def arms(self):
        return self._arms

    @property
    def rewards(self):
        return self._rewards

    @property
    def strategy(self):
        return self._strategy

    def play(self, num_choices, bandit):
        self._arms = np.zeros(num_choices).astype(np.int)
        self._rewards = np.zeros(num_choices).astype(np.int)
        for i in range(num_choices):
            arm = self._strategy.choice_arm(self._arms[:i], self._rewards[:i], bandit.num_arms, num_choices)
            assert 0 <= arm <= bandit.num_arms
            self._arms[i] = arm
            self._rewards[i] = bandit.get_rewards(arm)
        print('The game finished.')


class Strategy:
    def choice_arm(self, arms, rewards, num_arms, num_choices):
        raise NotImplementedError('Please implement this method!')


class EpsGreedyStrategy(Strategy):
    def __init__(self, eps=0.3):
        super().__init__()
        self._eps = eps

    def choice_arm(self, arms, rewards, num_arms, num_choices):
        assert len(arms) == len(rewards)
        next_index = len(arms)
        term_for_search = np.ceil(num_choices * self._eps).astype(np.int)
        term_for_search = np.max([term_for_search, num_arms]) # search all arms at least one time
        if next_index == 0: # Initial search
            return 0
        elif next_index < term_for_search:
            latest_arm = arms[-1]
            return (latest_arm + 1) % num_arms # search the next arm
        # Choice the next arm greedy
        df = pd.DataFrame({'arm': arms[:term_for_search], 'reward': rewards[:term_for_search]})
        return df.groupby('arm').sum().sort_values('reward', ascending=False).index[0]


class UCBStrategy(Strategy):
    def choice_arm(self, arms, rewards, num_arms, num_choices):
        """
        Maximize the following score:
            score = mu + sqrt(log t / (2 N(t)))
        """
        assert len(arms) == len(rewards)
        t = len(arms)
        if t == 0: # Initial search
            return 0
        elif t < num_arms:
            latest_arm = arms[-1]
            return (latest_arm + 1) % num_arms # search the next arm
        df = pd.DataFrame({'arm': arms, 'reward': rewards})
        df = df.groupby('arm').mean() + np.sqrt(0.5 * np.log(t) / df.groupby('arm').count())
        return df.sort_values('reward', ascending=False).index[0]


class IMEDtrategy(Strategy):
    def choice_arm(self, arms, rewards, num_arms, num_choices):
        """
        Minimize the following score:
            score =  N(t) * d(mu, mu_max) + log N(t)
        """
        assert len(arms) == len(rewards)
        t = len(arms)
        if t == 0: # Initial search
            return 0
        elif t < num_arms:
            latest_arm = arms[-1]
            return (latest_arm + 1) % num_arms # search the next arm

        def kl_div(p, q):
            assert (0 <= p <= 1) and (0 <= q <= 1)
            if q == 0 or q == 1:
                if p == q:
                    return 0
                else:
                    return np.inf
            d = 0
            if p > 0:
                d += p * np.log(p / q)
            if p < 1:
                d += (1 - p) * np.log((1 - p) / (1 - q))
            return d
        vect_kl_div = np.vectorize(kl_div, otypes=[np.float])

        df = pd.DataFrame({'arm': arms, 'reward': rewards})
        df_mean = df.groupby('arm').mean()
        df_cnt = df.groupby('arm').count()
        max_mean = df_mean.max()
        df_kldiv = pd.DataFrame(vect_kl_div(df_mean, max_mean), columns=df_mean.columns, index = df_mean.index)
        df_score = df_cnt * df_kldiv + np.log(df_cnt)
        return df_score.sort_values('reward', ascending=True).index[0]


def visualize_result(arms, rewards, player_name):
    # arms
    df_arms = pd.DataFrame({'arm': arms, 'index': np.arange(0, len(arms)), 'value': np.ones(len(arms))})
    df_arms_selected_ratio = pd.pivot_table(df_arms, index='index', columns='arm', values='value', fill_value=0)\
        .apply(lambda x: x.cumsum(), axis=0)\
        .apply(lambda x: x / x.sum() * 100, axis=1)
    for arm in df_arms_selected_ratio.columns:
        plt.plot(df_arms_selected_ratio[arm], label='arm {}'.format(arm))
    plt.legend()
    plt.title('Selected ratio of arm until each round in {}'.format(player_name))
    plt.show()


def main():
    bandit = BernoulliBandit([0.4, 0.5, 0.6])

    strategy_list = [EpsGreedyStrategy(), UCBStrategy(), IMEDtrategy()]
    for strategy in strategy_list:
        player = Player(strategy)
        player.play(50, bandit)
        print("*** {} ***".format(player.strategy.__class__.__name__))
        print("Arms   :", player.arms)
        print("Rewards:", player.rewards)
        visualize_result(player.arms, player.rewards, player.strategy.__class__.__name__)


if __name__ == '__main__':
    main()