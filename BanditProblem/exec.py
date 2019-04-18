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


class EpsGreedyPlayer:
    def __init__(self, eps=0.3):
        assert 0 < eps < 1
        self._arms = None
        self._rewards = None
        self._eps = eps
    
    @property
    def arms(self):
        return self._arms
    
    @property
    def rewards(self):
        return self._rewards
    
    @property
    def eps(self):
        return self._eps
    
    def set_eps(self, eps):
        assert 0 < eps < 1
        self._eps = eps
    
    @staticmethod
    def choice_arm(arms, rewards, num_arms, num_choices, eps):
        assert len(arms) == len(rewards)
        next_index = len(arms)
        term_for_search = np.ceil(num_choices * eps).astype(np.int)
        term_for_search = np.max([term_for_search, num_arms]) # search all arms at least one time
        if next_index == 0: # Initial search
            return 0
        elif next_index < term_for_search:
            latest_arm = arms[-1]
            return (latest_arm + 1) % num_arms # search the next arm
        # Choice the next arm greedy
        df = pd.DataFrame({'arm': arms[:term_for_search], 'reward': rewards[:term_for_search]})
        return df.groupby('arm').sum().sort_values('reward', ascending=False).index[0]

    def play(self, num_choices, bandit):
        self._arms = np.zeros(num_choices).astype(np.int)
        self._rewards = np.zeros(num_choices).astype(np.int)
        for i in range(num_choices):
            arm = self.choice_arm(self._arms[:i], self._rewards[:i], bandit.num_arms, num_choices, self._eps)
            assert 0 <= arm <= bandit.num_arms
            self._arms[i] = arm
            self._rewards[i] = bandit.get_rewards(arm)
        print('The game finished.')


class UCBPlayer:
    def __init__(self):
        self._arms = None
        self._rewards = None

    @property
    def arms(self):
        return self._arms

    @property
    def rewards(self):
        return self._rewards

    @staticmethod
    def choice_arm(arms, rewards, num_arms):
        """
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
        df = df.groupby('arm').mean() + np.sqrt(0.5 * np.log(t) / df.groupby('arm').sum())
        return df.sort_values('reward', ascending=False).index[0]

    def play(self, num_choices, bandit):
        self._arms = np.zeros(num_choices).astype(np.int)
        self._rewards = np.zeros(num_choices).astype(np.int)
        for i in range(num_choices):
            arm = self.choice_arm(self._arms[:i], self._rewards[:i], bandit.num_arms)
            assert 0 <= arm <= bandit.num_arms
            self._arms[i] = arm
            self._rewards[i] = bandit.get_rewards(arm)
        print('The game finished.')


def main():
    bandit = BernoulliBandit([0.4, 0.5, 0.6])

    player = EpsGreedyPlayer()
    player.play(30, bandit)
    print("*** {} ***".format(player.__class__.__name__))
    print("Arms   :", player.arms)
    print("Rewards:", player.rewards)

    player = UCBPlayer()
    player.play(30, bandit)
    print("*** {} ***".format(player.__class__.__name__))
    print("Arms   :", player.arms)
    print("Rewards:", player.rewards)

if __name__ == '__main__':
    main()