"""
Summary:
    Solve `Division Problem` by the simulated annealing method.

Usage:
    python sample.py

Requirements:
    numpy, scipy
"""

import numpy as np
from scipy import stats

np.random.seed(0)


def get_data():
    # return np.array([2, 10, 3, 8, 5, 7, 9, 5, 3, 2]) # http://www.iba.t.u-tokyo.ac.jp/~iba/AI/divide.pdf
    # return np.array([2, 5, 8, 6, 7, 2, 3, 10, 4, 11]) # http://www.kochi-tech.ac.jp/library/ron/2013/2013info/1140345.pdf
    return np.array(range(17))


def get_initial_spins(n):
    return np.ones(n)


def get_conditional_prob(target_index, spins, data, beta):
    """
        return [p(p_upward_spin), p(p_downward_spin)]
    """
    exponent = 4 * beta * data[target_index] * (np.dot(spins, data) - spins[target_index] * data[target_index])
    # To prevent overflow during calculating 'np.exp'
    if exponent > 0:
        p_downward_spin = 1 / (1 + np.exp(-exponent))
        prob = [1 - p_downward_spin, p_downward_spin]
    else:
        p_upward_spin = 1 / (1 + np.exp(exponent))
        prob = [p_upward_spin, 1 - p_upward_spin]
    return np.array(prob)


def get_division(spins, data):
    is_group_a = np.where(spins > 0)
    is_group_b = np.where(spins < 0)
    group_a = data[is_group_a]
    group_b = data[is_group_b]
    return group_a, group_b


def spins_to_num(spins):
    n_spins = len(spins)
    return sum([2**(n_spins-1-i) for i, s in enumerate(spins) if s > 0])


def num_to_spins(n_spins, num):
    bin_str = format(num, 'b').zfill(n_spins)
    return np.array([1 if s == '1' else -1 for s in bin_str])


def get_mcmc_sample(spins, data, beta):
    assert len(spins) == len(data)
    current_spins = np.copy(spins)
    mcmc_samples = list()
    n = len(data)
    max_iter = n * 100
    cnt = 0
    while cnt < max_iter:
        indices = np.random.permutation(range(n))
        for k in indices:
            if cnt >= max_iter:
                break
            cnt += 1
            prob = get_conditional_prob(k, current_spins, data, beta)
            current_spins[k] = np.random.choice([1, -1], size=1, p=prob)[0]
            if cnt < int(max_iter * 0.7):
                # burn in
                continue
            # save mcmc samples
            mcmc_samples.append(spins_to_num(current_spins))
    mcmc_mode = stats.mode(mcmc_samples).mode[0]
    return num_to_spins(n, mcmc_mode)


def get_hamiltonian(spins, data):
    return (np.dot(spins, data))**2


def main():
    # Load data
    data = get_data()
    print('Data: {}'.format(data))
    # Constants
    n = len(data)
    n_iter = n * 2
    # Variables to be updated
    spins = get_initial_spins(n)
    beta = 0.005
    # Main process
    for i in range(n_iter):
        spins = get_mcmc_sample(spins, data, beta)
        group_a, group_b = get_division(spins, data)
        print('Completed {}/{}-th process. Sums: {}(A) / {}(B)'.format(i+1, n_iter, group_a.sum(), group_b.sum()))
        beta += 0.005
    # Show result
    print('<Result>')
    group_a, group_b = get_division(spins, data)
    print('  Hamiltonian: {}'.format(get_hamiltonian(spins, data)))
    print('  Group A: {} / Sum: {}'.format(group_a, group_a.sum()))
    print('  Group B: {} / Sum: {}'.format(group_b, group_b.sum()))


if __name__=='__main__':
    main()
