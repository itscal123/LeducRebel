import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def get_paths_and_agents(args):
    """
    Create paths and agent list
    Currently supported: ReBeL, CFR
    """
    paths = ['experiments/leduc_holdem_rebel_result/performance.csv']
    agents = ['rebel']
    if args.agent == 'cfr':
        paths = ['experiments/leduc_holdem_cfr_result/performance.csv']
        agents = ['cfr']
    elif args.agent == 'all':
        paths.append('experiments/leduc_holdem_cfr_result/performance.csv')
        agents.append('cfr')

    now = datetime.now().strftime('%d_%m_%Y_%H%M')
    save_path = os.path.join('experiments/leduc_holdem_result_plot/', args.agent + '_' + now + '.pdf')
    return paths, save_path, agents


def plot_and_save(args):
    """
    Plot and save as .pdf files
    """
    paths, save_path, agents = get_paths_and_agents(args)
    for path, agent in zip(paths, agents):
        data = np.genfromtxt(path, delimiter=',', skip_header=1).T
        plt.plot(data[0], data[1], label=agent)
        plt.legend()
    plt.xlabel('Time step')
    plt.ylabel('Reward')
    plt.gcf().savefig(save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Collect and plot data')
    parser.add_argument(
        '--agent',
        type=str,
        default='rebel'
    )
    args = parser.parse_args()
    plot_and_save(args)
