''' An example of solve Leduc Hold'em with Rebel
'''
import os
import argparse
import sys

sys.path.append('../')

import rlcard
from rlcard.agents import (
    RebelAgent,
    RandomAgent,
)
from rlcard.utils import (
    tournament,
    Logger,
    plot_curve,
)

def train(args, i):
    # Make environments, Rebel only supports Leduc Holdem
    env = rlcard.make(
        'leduc-holdem',
        config={
            'allow_step_back': True,
        }
    )
    eval_env = rlcard.make(
        'leduc-holdem',
    )


    # Initilize Rebel Agent
    agent = RebelAgent(
        env,
        os.path.join(
            args.log_dir,
            'rebel_model',
        ),
    )

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
            agent.train()
            print('\rIteration {}'.format(episode), end='')
            # Evaluate the performance. Play with Random agents.
            if episode % args.evaluate_every == 0:
                agent.save() # Save model
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
    plot_curve(csv_path, fig_path, 'rebel')

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Rebel example in RLCard")
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
    )
    parser.add_argument(
        '--num_episodes',
        type=int,
        default=5000,
    )
    parser.add_argument(
        '--num_eval_games',
        type=int,
        default=2000,
    )
    parser.add_argument(
        '--evaluate_every',
        type=int,
        default=100,
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='experiments/leduc_holdem_rebel_result/',
    )

    args = parser.parse_args()

    # 10 runs
    for i in range(1,11):
        train(args, i)
    
