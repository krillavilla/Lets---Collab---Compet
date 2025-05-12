import numpy as np
import random
import copy
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.nn.functional as F

class RewardNormalizer:
    """Normalizes rewards using a running mean and standard deviation."""

    def __init__(self, epsilon=1e-8):
        """Initialize a RewardNormalizer object.

        Params
        ======
            epsilon (float): small constant to avoid division by zero
        """
        self.mean = 0.0
        self.std = 1.0
        self.count = 0
        self.epsilon = epsilon

    def normalize(self, reward):
        """Normalize a reward using the running statistics."""
        # Update running statistics
        self.count += 1
        delta = reward - self.mean
        self.mean += delta / self.count
        delta2 = reward - self.mean
        self.std = np.sqrt(((self.count - 1) * self.std**2 + delta * delta2) / self.count)

        # Normalize the reward
        return reward / (self.std + self.epsilon)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, fc1_units=128, fc2_units=64, activation_fn='relu', use_batch_norm=True):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
            activation_fn (str): Activation function to use ('relu', 'leaky_relu', or 'elu')
            use_batch_norm (bool): Whether to use batch normalization
        """
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.activation_fn = activation_fn
        self.use_batch_norm = use_batch_norm

        # Batch normalization layers
        if self.use_batch_norm:
            self.bn1 = nn.BatchNorm1d(fc1_units)
            self.bn2 = nn.BatchNorm1d(fc2_units)

        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        if state.dim() == 1:
            state = torch.unsqueeze(state, 0)

        # Apply the selected activation function with batch normalization if enabled
        if self.use_batch_norm:
            if self.activation_fn == 'relu':
                x = F.relu(self.bn1(self.fc1(state)))
                x = F.relu(self.bn2(self.fc2(x)))
            elif self.activation_fn == 'leaky_relu':
                x = F.leaky_relu(self.bn1(self.fc1(state)))
                x = F.leaky_relu(self.bn2(self.fc2(x)))
            elif self.activation_fn == 'elu':
                x = F.elu(self.bn1(self.fc1(state)))
                x = F.elu(self.bn2(self.fc2(x)))
            else:
                # Default to ReLU if an invalid activation function is specified
                x = F.relu(self.bn1(self.fc1(state)))
                x = F.relu(self.bn2(self.fc2(x)))
        else:
            # Without batch normalization
            if self.activation_fn == 'relu':
                x = F.relu(self.fc1(state))
                x = F.relu(self.fc2(x))
            elif self.activation_fn == 'leaky_relu':
                x = F.leaky_relu(self.fc1(state))
                x = F.leaky_relu(self.fc2(x))
            elif self.activation_fn == 'elu':
                x = F.elu(self.fc1(state))
                x = F.elu(self.fc2(x))
            else:
                # Default to ReLU if an invalid activation function is specified
                x = F.relu(self.fc1(state))
                x = F.relu(self.fc2(x))

        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, fc1_units=128, fc2_units=64, activation_fn='relu', use_batch_norm=True):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            fc1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
            activation_fn (str): Activation function to use ('relu', 'leaky_relu', or 'elu')
            use_batch_norm (bool): Whether to use batch normalization
        """
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units + action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.activation_fn = activation_fn
        self.use_batch_norm = use_batch_norm

        # Batch normalization layers
        if self.use_batch_norm:
            self.bn1 = nn.BatchNorm1d(fc1_units)
            self.bn2 = nn.BatchNorm1d(fc2_units)

        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        if state.dim() == 1:
            state = torch.unsqueeze(state, 0)

        # Apply the selected activation function with batch normalization if enabled
        if self.use_batch_norm:
            if self.activation_fn == 'relu':
                xs = F.relu(self.bn1(self.fc1(state)))
                x = torch.cat((xs, action), dim=1)
                x = F.relu(self.bn2(self.fc2(x)))
            elif self.activation_fn == 'leaky_relu':
                xs = F.leaky_relu(self.bn1(self.fc1(state)))
                x = torch.cat((xs, action), dim=1)
                x = F.leaky_relu(self.bn2(self.fc2(x)))
            elif self.activation_fn == 'elu':
                xs = F.elu(self.bn1(self.fc1(state)))
                x = torch.cat((xs, action), dim=1)
                x = F.elu(self.bn2(self.fc2(x)))
            else:
                # Default to ReLU if an invalid activation function is specified
                xs = F.relu(self.bn1(self.fc1(state)))
                x = torch.cat((xs, action), dim=1)
                x = F.relu(self.bn2(self.fc2(x)))
        else:
            # Without batch normalization
            if self.activation_fn == 'relu':
                xs = F.relu(self.fc1(state))
                x = torch.cat((xs, action), dim=1)
                x = F.relu(self.fc2(x))
            elif self.activation_fn == 'leaky_relu':
                xs = F.leaky_relu(self.fc1(state))
                x = torch.cat((xs, action), dim=1)
                x = F.leaky_relu(self.fc2(x))
            elif self.activation_fn == 'elu':
                xs = F.elu(self.fc1(state))
                x = torch.cat((xs, action), dim=1)
                x = F.elu(self.fc2(x))
            else:
                # Default to ReLU if an invalid activation function is specified
                xs = F.relu(self.fc1(state))
                x = torch.cat((xs, action), dim=1)
                x = F.relu(self.fc2(x))

        return self.fc3(x)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.5):
        """Initialize parameters and noise process.

        Params
        ======
            size (int): dimension of the action space
            seed (int): random seed
            mu (float): mean of the noise distribution (typically 0)
            theta (float): parameter controlling the mean reversion rate
            sigma (float): parameter controlling the scale of the noise (increased for more exploration)
        """
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = np.random.seed(seed)
        self.size = size
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(len(x))
        self.state = x + dx
        return self.state

    def decay_noise(self, decay_rate=0.995):
        """Decay the sigma parameter to reduce noise over time."""
        self.sigma *= decay_rate

    def decay_sigma(self, decay_rate=0.995):
        """Legacy method for backward compatibility. Use decay_noise instead."""
        self.decay_noise(decay_rate)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class DDPGAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, random_seed, buffer_size=int(1e5), batch_size=64,
                 gamma=0.98, tau=1e-3, lr_actor=1e-3, lr_critic=5e-4, weight_decay=0,
                 network_size='small', activation_fn='relu', use_batch_norm=True, 
                 grad_clip=1.0):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            buffer_size (int): replay buffer size
            batch_size (int): minibatch size
            gamma (float): discount factor
            tau (float): for soft update of target parameters
            lr_actor (float): learning rate for actor
            lr_critic (float): learning rate for critic
            weight_decay (float): L2 weight decay for critic
            network_size (str): size of the network ('small' or 'large')
            activation_fn (str): activation function to use ('relu', 'leaky_relu', or 'elu')
            use_batch_norm (bool): whether to use batch normalization
            grad_clip (float): value for gradient clipping
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.grad_clip = grad_clip

        # Set network sizes based on the network_size parameter
        if network_size == 'large':
            fc1_units, fc2_units = 256, 128
        else:  # 'small' or any other value
            fc1_units, fc2_units = 128, 64

        # Store the use_batch_norm parameter
        self.use_batch_norm = use_batch_norm

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, fc1_units, fc2_units, activation_fn, use_batch_norm).to(device)
        self.actor_target = Actor(state_size, action_size, fc1_units, fc2_units, activation_fn, use_batch_norm).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor_local.parameters(), lr=lr_actor)

        # Learning rate scheduler for actor
        self.actor_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.actor_optimizer, gamma=0.99)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, fc1_units, fc2_units, activation_fn, use_batch_norm).to(device)
        self.critic_target = Critic(state_size, action_size, fc1_units, fc2_units, activation_fn, use_batch_norm).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic_local.parameters(), lr=lr_critic, weight_decay=weight_decay)

        # Learning rate scheduler for critic
        self.critic_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.critic_optimizer, gamma=0.99)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, random_seed)

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        losses = None
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            losses = self.learn(experiences, self.gamma)

        return losses

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        """Reset the agent's noise process and other internal states if needed."""
        self.noise.reset()
        # Additional reset operations can be added here if needed

    def step_schedulers(self):
        """Step the learning rate schedulers."""
        self.actor_scheduler.step()
        self.critic_scheduler.step()

    def get_lr(self):
        """Get the current learning rates."""
        return {
            'actor_lr': self.actor_optimizer.param_groups[0]['lr'],
            'critic_lr': self.critic_optimizer.param_groups[0]['lr']
        }

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), self.grad_clip)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), self.grad_clip)
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)

        # Return losses for logging
        return critic_loss.item(), actor_loss.item()

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class MultiAgentDDPG:
    """Multiple DDPG agents that interact with each other."""

    def __init__(self, state_size, action_size, num_agents, random_seed, buffer_size=int(1e5), batch_size=64,
                 gamma=0.98, tau=1e-3, lr_actor=1e-3, lr_critic=5e-4, weight_decay=0,
                 network_size='small', activation_fn='relu', use_batch_norm=True, grad_clip=1.0):
        """Initialize a MultiAgentDDPG object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            num_agents (int): number of agents
            random_seed (int): random seed
            buffer_size (int): replay buffer size
            batch_size (int): minibatch size
            gamma (float): discount factor
            tau (float): for soft update of target parameters
            lr_actor (float): learning rate for actor
            lr_critic (float): learning rate for critic
            weight_decay (float): L2 weight decay for critic
            network_size (str): size of the network ('small' or 'large')
            activation_fn (str): activation function to use ('relu', 'leaky_relu', or 'elu')
            use_batch_norm (bool): whether to use batch normalization
        """
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = random.seed(random_seed)

        # Store hyperparameters
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.weight_decay = weight_decay
        self.network_size = network_size
        self.activation_fn = activation_fn
        self.use_batch_norm = use_batch_norm
        self.grad_clip = grad_clip

        # Create multiple agents with the specified hyperparameters
        self.agents = [DDPGAgent(state_size, action_size, random_seed, 
                                buffer_size=buffer_size, 
                                batch_size=batch_size,
                                gamma=gamma, 
                                tau=tau, 
                                lr_actor=lr_actor, 
                                lr_critic=lr_critic, 
                                weight_decay=weight_decay,
                                network_size=network_size,
                                activation_fn=activation_fn,
                                use_batch_norm=use_batch_norm,
                                grad_clip=grad_clip) for _ in range(num_agents)]

    def reset(self):
        """Reset all the agents."""
        for agent in self.agents:
            agent.reset()

    def act(self, states, add_noise=True):
        """Get actions from all agents based on current policy."""
        actions = []
        for agent, state in zip(self.agents, states):
            actions.append(agent.act(state, add_noise))
        return actions

    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn.

        Returns:
            losses (list): List of (critic_loss, actor_loss) tuples for each agent, or None if not learning
        """
        losses = []
        for i, agent in enumerate(self.agents):
            agent_losses = agent.step(states[i], actions[i], rewards[i], next_states[i], dones[i])
            losses.append(agent_losses)

        return losses

    def decay_noise(self, decay_rate=0.995):
        """Decay the noise parameter for all agents to reduce exploration over time."""
        for agent in self.agents:
            agent.noise.decay_noise(decay_rate)

    def step_schedulers(self):
        """Step the learning rate schedulers for all agents."""
        for agent in self.agents:
            agent.step_schedulers()

    def get_lr(self):
        """Get the current learning rates for all agents."""
        return [agent.get_lr() for agent in self.agents]

    def save_checkpoint(self, filename):
        """Save the agents' models to a file."""
        checkpoint = {
            'agent_0_actor_local': self.agents[0].actor_local.state_dict(),
            'agent_0_actor_target': self.agents[0].actor_target.state_dict(),
            'agent_0_critic_local': self.agents[0].critic_local.state_dict(),
            'agent_0_critic_target': self.agents[0].critic_target.state_dict(),
            'agent_1_actor_local': self.agents[1].actor_local.state_dict(),
            'agent_1_actor_target': self.agents[1].actor_target.state_dict(),
            'agent_1_critic_local': self.agents[1].critic_local.state_dict(),
            'agent_1_critic_target': self.agents[1].critic_target.state_dict(),
        }
        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename):
        """Load the agents' models from a file."""
        checkpoint = torch.load(filename)
        self.agents[0].actor_local.load_state_dict(checkpoint['agent_0_actor_local'])
        self.agents[0].actor_target.load_state_dict(checkpoint['agent_0_actor_target'])
        self.agents[0].critic_local.load_state_dict(checkpoint['agent_0_critic_local'])
        self.agents[0].critic_target.load_state_dict(checkpoint['agent_0_critic_target'])
        self.agents[1].actor_local.load_state_dict(checkpoint['agent_1_actor_local'])
        self.agents[1].actor_target.load_state_dict(checkpoint['agent_1_actor_target'])
        self.agents[1].critic_local.load_state_dict(checkpoint['agent_1_critic_local'])
        self.agents[1].critic_target.load_state_dict(checkpoint['agent_1_critic_target'])
