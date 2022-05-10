import numpy as np
import collections

import random
import os
import pickle
import torch
import torch.nn as nn

from copy import deepcopy
from collections import namedtuple
from rlcard.utils.utils import *



Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'legal_actions', 'done'])

class RebelAgent():
    ''' 
    Implement CFR (chance sampling) algorithm
    '''

    def __init__(
        self, env, model_path='./rebel_model', 
        replay_memory_size=20000,
        replay_memory_init_size=100,
        discount_factor=0.99,
        batch_size=32,
        train_value_every=1,
        update_policy_every=1000,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay_steps=20000,
        num_actions=None, 
        learning_rate=3e-4, 
        state_shape=None,
        mlp_layers=None, 
        device=None):
        '''
        Initilize Agent
        ------------
        args:
            env: Env object
            model_path: str
            num_actions (int): number of actions
            learning_rate (float): learning rate of the value/policy networks
            state_shape (list): list of the space of the state vector
            mlp_layers (list): layer number and dimension of each layer in the network
            device (torch device): whether to use CPU or GPU
        '''
        self.use_raw = False
        self.num_actions = num_actions or env.num_actions
        self.replay_memory_init_size = replay_memory_init_size
        self.env = env
        self.model_path = model_path
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.state_shape = state_shape or env.state_shape[0]
        self.mlp_layers= mlp_layers or [64, 64]
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.update_policy_every = update_policy_every
        self.train_value_every = train_value_every

        # Total timesteps
        self.total_t = 0

        # Total training step
        self.train_t = 0

        # The epsilon decay scheduler
        self.epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

        # A policy is a dict state_str -> action probabilities
        self.policy = collections.defaultdict(list)
        self.average_policy = collections.defaultdict(np.array)

        # Regret is a dict state_str -> action regrets
        self.regrets = collections.defaultdict(np.array)

        # Initialize value/policy networks
        self.valueNetwork = Estimator(
            num_actions=self.num_actions, 
            learning_rate=learning_rate, 
            state_shape=self.state_shape, 
            mlp_layers=self.mlp_layers, 
            device=self.device
        )

        self.policyNetwork = Estimator(
            num_actions=self.num_actions, 
            learning_rate=learning_rate, 
            state_shape=self.state_shape, 
            mlp_layers=self.mlp_layers, 
            device=self.device
        )

        # Create replay memory
        self.memory = Memory(replay_memory_size, batch_size)    

        self.iteration = 0

    # TODO: rename
    def train(self):
        ''' Do one iteration of CFR
        '''
        self.iteration += 1
        # Firstly, traverse tree to compute counterfactual regret for each player
        # The regrets are recorded in traversal
        for player_id in range(self.env.num_players):
            self.env.reset()
            probs = np.ones(self.env.num_players)
            self.traverse_tree(probs, player_id)

        # Update policy
        self.update_policy()


    # TODO: Implement Rebel Logic from paper and rename
    def train2(self):
        """
        Rebel algorithm for RL and Search for Imperfect-Information Games
        Rough pseudocode:
        while !ISTERMINAL(βr) do
            G ← CONSTRUCTSUBGAME(βr)
            π_bar, π_t_warm ← INITIALIZEPOLICY(G, θπ)
            G ← SETLEAFVALUES(G, π _bar, π_t_warm , θv)
            v(βr) ← COMPUTEEV(G, π_t_warm )
            tsample ∼ unif {t_warm + 1, T }
            for t = (t_warm + 1)..T do
                if t = tsample then
                    βr′ ←SAMPLELEAF(G,π_t − 1)
                π_t UPDATEPOLICY(G, π_t − 1)
                π_bar ← t π _bar + 1 π t
                G ← SETLEAFVALUES(G, π_bar, π_t, θv)
                v(βr)← t v(βr)+ 1 COMPUTEEV(G,π_t) t+1 t+1
            Add {βr , v(βr )} to D_v 
            for β ∈ G do
                Add {β, π_bar(β)} to D_π 
            βr ← βr′
        """
        self.train()

        trajectories, payoffs = self.env.run(is_training=True)

        # Reorganaize the data to be state, action, reward, next_state, done
        trajectories = reorganize(trajectories, payoffs)

        # Feed transitions into agent memory, and train the agent
        # Here, we assume that DQN always plays the first position
        # and the other players play randomly (if any)
        for ts in trajectories[0]:
            self.feed(ts)
        
        


    def feed(self, ts):
        ''' 
        Store data in to replay buffer and train the agent. There are two stages.
        In stage 1, populate the memory without training
        In stage 2, train the agent every several timesteps

        Args:
            ts (list): a list of 5 elements that represent the transition
        '''
        (state, action, reward, next_state, done) = tuple(ts)
        self.feed_memory(state['obs'], action, reward, next_state['obs'], list(next_state['legal_actions'].keys()), done)
        self.total_t += 1
        tmp = self.total_t - self.replay_memory_init_size
        if tmp>=0 and tmp%self.train_value_every == 0:
            self.trainValueNetwork()


    def step(self, state):
        ''' 
        Predict the action for genrating training data but
        have the predictions disconnected from the computation graph

        Args:
            state (numpy.array): current state

        Returns:
            action (int): an action id
        '''
        values = self.predict(state)
        epsilon = self.epsilons[min(self.total_t, self.epsilon_decay_steps-1)]
        legal_actions = list(state['legal_actions'].keys())
        probs = np.ones(len(legal_actions), dtype=float) * epsilon / len(legal_actions)
        best_action_idx = legal_actions.index(np.argmax(values))
        probs[best_action_idx] += (1.0 - epsilon)
        action_idx = np.random.choice(np.arange(len(probs)), p=probs)

        return legal_actions[action_idx]


    def eval_step(self, state):
        '''
        Predict the action for evaluation purpose.

        Args:
            state (numpy.array): current state

        Returns:
            action (int): an action id
            info (dict): A dictionary containing information
        '''
        values = self.predict(state)
        best_action = np.argmax(values)

        info = {}
        info['values'] = {state['raw_legal_actions'][i]: float(values[list(state['legal_actions'].keys())[i]]) for i in range(len(state['legal_actions']))}

        return best_action, info


    def predict(self, state):
        '''
        Predict the masked values

        Args:
            state (numpy.array): current state

        Returns:
            values (numpy.array): a 1-d array where each entry represents values/policies
        '''
        
        values = self.valueNetwork.predict_nograd(np.expand_dims(state['obs'], 0))[0]
        masked_values = -np.inf * np.ones(self.num_actions, dtype=float)
        legal_actions = list(state['legal_actions'].keys())
        masked_values[legal_actions] = values[legal_actions]

        return masked_values


    def trainValueNetwork(self):
        '''
        Train the value network

        Returns:
            loss (float): The loss of the current batch.
        '''
        state_batch, action_batch, reward_batch, next_state_batch, legal_actions_batch, done_batch = self.memory.sample()

        # Calculate best next actions using value network
        values_next = self.valueNetwork.predict_nograd(next_state_batch)
        legal_actions = []
        for b in range(self.batch_size):
            legal_actions.extend([i + b * self.num_actions for i in legal_actions_batch[b]])
        masked_values = -np.inf * np.ones(self.num_actions * self.batch_size, dtype=float)
        masked_values[legal_actions] = values_next.flatten()[legal_actions]
        masked_values = masked_values.reshape((self.batch_size, self.num_actions))
        best_actions = np.argmax(masked_values, axis=1)

        # Evaluate best next actions using policy network
        values_next_target = self.policyNetwork.predict_nograd(next_state_batch)
        target_batch = reward_batch + np.invert(done_batch).astype(np.float32) * \
            self.discount_factor * values_next_target[np.arange(self.batch_size), best_actions]

        # Perform gradient descent update
        state_batch = np.array(state_batch)

        loss = self.valueNetwork.update(state_batch, action_batch, target_batch)
        print('\rINFO - Step {}, rl-loss: {}'.format(self.total_t, loss), end='')

        # Update the policy network
        if self.train_t % self.update_policy_every == 0:
            self.policyNetwork = deepcopy(self.valueNetwork)
            print("\nINFO - Copied model parameters to target network.")

        self.train_t += 1


    def feed_memory(self, state, action, reward, next_state, legal_actions, done):
        '''
        Feed transition to memory

        Args:
            state (numpy.array): the current state
            action (int): the performed action ID
            reward (float): the reward received
            next_state (numpy.array): the next state after performing the action
            legal_actions (list): the legal actions of the next state
            done (boolean): whether the episode is finished
        '''
        self.memory.save(state, action, reward, next_state, legal_actions, done)


    def set_device(self, device):
        self.device = device
        self.valueNetwork.device = device
        self.policyNetwork.device = device


    def traverse_tree(self, probs, player_id):
        ''' Traverse the game tree, update the regrets

        Args:
            probs: The reach probability of the current node
            player_id: The player to update the value

        Returns:
            state_utilities (list): The expected utilities for all the players
        '''
        if self.env.is_over():
            return self.env.get_payoffs()

        current_player = self.env.get_player_id()

        action_utilities = {}
        state_utility = np.zeros(self.env.num_players)
        obs, legal_actions = self.get_state(current_player)
        action_probs = self.action_probs(obs, legal_actions, self.policy)

        for action in legal_actions:
            action_prob = action_probs[action]
            new_probs = probs.copy()
            new_probs[current_player] *= action_prob

            # Keep traversing the child state
            self.env.step(action)
            utility = self.traverse_tree(new_probs, player_id)
            self.env.step_back()

            state_utility += action_prob * utility
            action_utilities[action] = utility

        if not current_player == player_id:
            return state_utility

        # If it is current player, we record the policy and compute regret
        player_prob = probs[current_player]
        counterfactual_prob = (np.prod(probs[:current_player]) *
                                np.prod(probs[current_player + 1:]))
        player_state_utility = state_utility[current_player]

        if obs not in self.regrets:
            self.regrets[obs] = np.zeros(self.env.num_actions)
        if obs not in self.average_policy:
            self.average_policy[obs] = np.zeros(self.env.num_actions)
        for action in legal_actions:
            action_prob = action_probs[action]
            regret = counterfactual_prob * (action_utilities[action][current_player]
                    - player_state_utility)
            self.regrets[obs][action] += regret
            self.average_policy[obs][action] += self.iteration * player_prob * action_prob
        return state_utility


    def update_policy(self):
        ''' Update policy based on the current regrets
        '''
        for obs in self.regrets:
            self.policy[obs] = self.regret_matching(obs)


    def regret_matching(self, obs):
        ''' Apply regret matching

        Args:
            obs (string): The state_str
        '''
        regret = self.regrets[obs]
        positive_regret_sum = sum([r for r in regret if r > 0])

        action_probs = np.zeros(self.env.num_actions)
        if positive_regret_sum > 0:
            for action in range(self.env.num_actions):
                action_probs[action] = max(0.0, regret[action] / positive_regret_sum)
        else:
            for action in range(self.env.num_actions):
                action_probs[action] = 1.0 / self.env.num_actions
        return action_probs


    def action_probs(self, obs, legal_actions, policy):
        ''' Obtain the action probabilities of the current state

        Args:
            obs (str): state_str
            legal_actions (list): List of leagel actions
            player_id (int): The current player
            policy (dict): The used policy

        Returns:
            (tuple) that contains:
                action_probs(numpy.array): The action probabilities
                legal_actions (list): Indices of legal actions
        '''
        if obs not in policy.keys():
            action_probs = np.array([1.0/self.env.num_actions for _ in range(self.env.num_actions)])
            self.policy[obs] = action_probs
        else:
            action_probs = policy[obs]
        action_probs = remove_illegal(action_probs, legal_actions)
        return action_probs


    def get_state(self, player_id):
        ''' Get state_str of the player

        Args:
            player_id (int): The player id

        Returns:
            (tuple) that contains:
                state (str): The state str
                legal_actions (list): Indices of legal actions
        '''
        state = self.env.get_state(player_id)
        return state['obs'].tostring(), list(state['legal_actions'].keys())


    def save(self):
        ''' Save model
        '''
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        policy_file = open(os.path.join(self.model_path, 'rebel_policy.pkl'),'wb')
        pickle.dump(self.policy, policy_file)
        policy_file.close()

        average_policy_file = open(os.path.join(self.model_path, 'rebel_average_policy.pkl'),'wb')
        pickle.dump(self.average_policy, average_policy_file)
        average_policy_file.close()

        regrets_file = open(os.path.join(self.model_path, 'rebel_regrets.pkl'),'wb')
        pickle.dump(self.regrets, regrets_file)
        regrets_file.close()

        iteration_file = open(os.path.join(self.model_path, 'rebel_iteration.pkl'),'wb')
        pickle.dump(self.iteration, iteration_file)
        iteration_file.close()

    def load(self):
        ''' Load model
        '''
        if not os.path.exists(self.model_path):
            return

        policy_file = open(os.path.join(self.model_path, 'rebel_policy.pkl'),'rb')
        self.policy = pickle.load(policy_file)
        policy_file.close()

        average_policy_file = open(os.path.join(self.model_path, 'rebel_average_policy.pkl'),'rb')
        self.average_policy = pickle.load(average_policy_file)
        average_policy_file.close()

        regrets_file = open(os.path.join(self.model_path, 'rebel_regrets.pkl'),'rb')
        self.regrets = pickle.load(regrets_file)
        regrets_file.close()

        iteration_file = open(os.path.join(self.model_path, 'rebel_iteration.pkl'),'rb')
        self.iteration = pickle.load(iteration_file)
        iteration_file.close()

class Estimator(object):
    '''
    Estimator neural network.
    This network is used for both the value network and the policy network. All methods input/output np.ndarray.
    '''
    def __init__(self, num_actions=2, learning_rate=0.001, state_shape=None, mlp_layers=None, device=None):
        ''' Initilalize an Estimator object.

        Args:
            num_actions (int): the number output actions
            state_shape (list): the shape of the state space
            mlp_layers (list): size of outputs of mlp layers
            device (torch.device): whether to use cpu or gpu
        '''
        self.num_actions = num_actions
        self.learning_rate=learning_rate
        self.state_shape = state_shape
        self.mlp_layers = mlp_layers
        self.device = device

        # set up estimator model and place it in eval mode
        model = EstimatorNetwork(num_actions, state_shape, mlp_layers)
        model = model.to(self.device)
        self.model = model
        self.model.eval()

        # initialize the weights using Xavier init
        for p in self.model.parameters():
            if len(p.data.shape) > 1:
                nn.init.xavier_uniform_(p.data)

        # set up loss function
        self.mse_loss = nn.MSELoss(reduction='mean')

        # set up optimizer
        self.optimizer =  torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)


    def predict_nograd(self, s):
        ''' Predicts action values, but prediction is not included
            in the computation graph.  It is used to predict optimal next
            actions.

        Args:
          s (np.ndarray): (batch, state_len)

        Returns:
          np.ndarray of shape (batch_size, NUM_VALID_ACTIONS) containing the estimated
          action values.
        '''
        with torch.no_grad():
            s = torch.from_numpy(s).float().to(self.device)
            model_as = self.model(s).cpu().numpy()
        return model_as


    def update(self, s, a, y):
        ''' Updates the estimator towards the given targets.

        Args:
          s (np.ndarray): (batch, state_shape) state representation
          a (np.ndarray): (batch,) integer sampled actions
          y (np.ndarray): (batch,) value of optimal actions according to Q-target

        Returns:
          The calculated loss on the batch.
        '''
        self.optimizer.zero_grad()

        self.model.train()

        s = torch.from_numpy(s).float().to(self.device)
        a = torch.from_numpy(a).long().to(self.device)
        y = torch.from_numpy(y).float().to(self.device)

        # (batch, state_shape) -> (batch, num_actions)
        model_as = self.model(s)

        # (batch, num_actions) -> (batch, )
        actions = torch.gather(model_as, dim=-1, index=a.unsqueeze(-1)).squeeze(-1)

        # update model
        batch_loss = self.mse_loss(actions, y)
        batch_loss.backward()
        self.optimizer.step()
        batch_loss = batch_loss.item()

        self.model.eval()

        return batch_loss


class EstimatorNetwork(nn.Module):
    ''' The function approximation network for Estimator
        It is just a series of GELU layers. All in/out are torch.tensor
    '''

    def __init__(self, num_actions=2, state_shape=None, mlp_layers=None):
        ''' Initialize the value network

        Args:
            num_actions (int): number of legal actions
            state_shape (list): shape of state tensor
            mlp_layers (list): output size of each fc layer
        '''
        super(EstimatorNetwork, self).__init__()

        self.num_actions = num_actions
        self.state_shape = state_shape
        self.mlp_layers = mlp_layers

        # build the value network
        layer_dims = [np.prod(self.state_shape)] + self.mlp_layers
        fc = [nn.Flatten()]
        fc.append(nn.BatchNorm1d(layer_dims[0]))
        for i in range(len(layer_dims)-1):
            fc.append(nn.Linear(layer_dims[i], layer_dims[i+1], bias=True))
            fc.append(nn.GELU())
        fc.append(nn.Linear(layer_dims[-1], self.num_actions, bias=True))
        self.fc_layers = nn.Sequential(*fc)


    def forward(self, s):
        ''' Predict action values

        Args:
            s  (Tensor): (batch, state_shape)
        '''
        return self.fc_layers(s)


class Memory(object):
    ''' Memory for saving transitions
    '''

    def __init__(self, memory_size, batch_size):
        ''' Initialize
        Args:
            memory_size (int): the size of the memroy buffer
        '''
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = []


    def save(self, state, action, reward, next_state, legal_actions, done):
        ''' Save transition into memory

        Args:
            state (numpy.array): the current state
            action (int): the performed action ID
            reward (float): the reward received
            next_state (numpy.array): the next state after performing the action
            legal_actions (list): the legal actions of the next state
            done (boolean): whether the episode is finished
        '''
        if len(self.memory) == self.memory_size:
            self.memory.pop(0)
        transition = Transition(state, action, reward, next_state, legal_actions, done)
        self.memory.append(transition)


    def sample(self):
        ''' Sample a minibatch from the replay memory

        Returns:
            state_batch (list): a batch of states
            action_batch (list): a batch of actions
            reward_batch (list): a batch of rewards
            next_state_batch (list): a batch of states
            done_batch (list): a batch of dones
        '''
        samples = random.sample(self.memory, self.batch_size)
        return map(np.array, zip(*samples))