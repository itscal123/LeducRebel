''' An example of playing randomly in RLCard
'''
import sys

sys.path.append('../')

import argparse
import pprint

import rlcard
from rlcard.agents import RandomAgent

from rlcard.utils import (
    tournament,
    Logger,
    plot_curve,
)

def run(args, i):
    # Make environment
    env = rlcard.make(
        args.env,
    )


    # Initialize random agent
    agent = RandomAgent(num_actions=env.num_actions)

    """
    # Set agents
    agent = RandomAgent(num_actions=env.num_actions)
    env.set_agents([agent for _ in range(env.num_players)])

    # Generate data from the environment
    trajectories, player_wins = env.run(is_training=False)
    # Print out the trajectories
    print('\nTrajectories:')
    print(trajectories)
    print('\nSample raw observation:')
    pprint.pprint(trajectories[0][0]['raw_obs'])
    print('\nSample raw legal_actions:')
    pprint.pprint(trajectories[0][0]['raw_legal_actions'])
    """
    # Evaluate Rebel against random
    env.set_agents([
        agent,
        RandomAgent(num_actions=env.num_actions),
    ])

    # Start training
    log_dir = args.log_dir[:-1] + f'/run_{i}/'
    env.timestep = 0
    with Logger(log_dir) as logger:
        for episode in range(args.num_episodes):
            print('\rIteration {}'.format(episode), end='')
            # Evaluate the performance. Play with Random agents.
            if episode % args.evaluate_every == 0:
                logger.log_performance(
                    env.timestep,
                    tournament(
                        env,
                        args.num_eval_games
                    )[0]
                )

        # Get the paths
        csv_path, fig_path = logger.csv_path, logger.fig_path

    # Plot the learning curve
    plot_curve(csv_path, fig_path, 'random')

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Random example in RLCard")
    parser.add_argument(
        '--env',
        type=str,
        default='leduc-holdem',
        choices=[
            'blackjack',
            'leduc-holdem',
            'limit-holdem',
            'doudizhu',
            'mahjong',
            'no-limit-holdem',
            'uno',
            'gin-rummy',
            'bridge',
        ],
    )

    parser.add_argument(
        '--num_episodes',
        type=int,
        default=10000,
    )
    parser.add_argument(
        '--num_eval_games',
        type=int,
        default=500,
    )
    parser.add_argument(
        '--evaluate_every',
        type=int,
        default=100,
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='experiments/leduc_holdem_random_result/',
    )


    args = parser.parse_args()
    for i in range(1,11):
        run(args,i)