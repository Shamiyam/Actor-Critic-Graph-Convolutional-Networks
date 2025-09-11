import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import os
import random

# Set seeds for reproducibility
def set_seeds(seed=42):
    """Set seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Create Actor network
class Actor(nn.Module):
    """Actor network that outputs action probabilities given states."""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        """Forward pass through the network."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)  # Softmax for probability distribution
        return x

# Create Critic network
class Critic(nn.Module):
    """Critic network that estimates the value function of states."""
    def __init__(self, state_dim, hidden_dim=128):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)  # Output a single value
        
    def forward(self, x):
        """Forward pass through the network."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # No activation, raw value estimate
        return x

# Actor-Critic agent
class ActorCritic:
    """Actor-Critic agent that learns to interact with an environment."""
    def __init__(self, state_dim, action_dim, lr_actor=0.0003, lr_critic=0.001, gamma=0.99):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.gamma = gamma
        
    def select_action(self, state):
        """Select an action based on the current policy."""
        state = torch.FloatTensor(state)
        probs = self.actor(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)
    
    def update(self, state, action, reward, next_state, done, log_prob):
        """Update actor and critic networks."""
        # Convert to tensors
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        reward = torch.FloatTensor([reward])
        done = torch.FloatTensor([1.0 if done else 0.0])
        
        # Calculate advantage and value targets
        next_value = self.critic(next_state) * (1 - done)
        target_value = reward + self.gamma * next_value
        value = self.critic(state)
        
        # Calculate losses
        advantage = target_value.detach() - value
        actor_loss = -log_prob * advantage.detach()
        critic_loss = F.mse_loss(value, target_value.detach())
        
        # Update networks
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()
        
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()
        
        return actor_loss.item(), critic_loss.item()

def train(env_name, num_episodes=1000, hidden_dim=128, lr_actor=0.0003, lr_critic=0.001, gamma=0.99, seed=42):
    """Train the Actor-Critic agent."""
    # Set seeds for reproducibility
    set_seeds(seed)
    
    # Create environment
    env = gym.make(env_name)
    
    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Create agent
    agent = ActorCritic(state_dim, action_dim, lr_actor, lr_critic, gamma)
    
    # Lists to store results
    all_episode_rewards = []
    actor_losses = []
    critic_losses = []
    
    # Training loop
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        episode_actor_loss = 0
        episode_critic_loss = 0
        steps = 0
        
        while not done:
            action, log_prob = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            actor_loss, critic_loss = agent.update(state, action, reward, next_state, done, log_prob)
            episode_actor_loss += actor_loss
            episode_critic_loss += critic_loss
            
            state = next_state
            episode_reward += reward
            steps += 1
        
        # Store results
        all_episode_rewards.append(episode_reward)
        actor_losses.append(episode_actor_loss / steps)
        critic_losses.append(episode_critic_loss / steps)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(all_episode_rewards[-10:])
            print(f"Episode {episode+1}, Average Reward (last 10): {avg_reward:.2f}")
    
    return all_episode_rewards, actor_losses, critic_losses, agent

def evaluate(env_name, agent, num_episodes=10, render=False):
    """Evaluate the trained agent."""
    env = gym.make(env_name)
    
    all_rewards = []
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            if render:
                env.render()
            
            action, _ = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            state = next_state
            episode_reward += reward
        
        all_rewards.append(episode_reward)
        print(f"Test Episode {episode+1}: Reward = {episode_reward:.2f}")
    
    avg_reward = np.mean(all_rewards)
    print(f"Average Test Reward: {avg_reward:.2f}")
    
    return all_rewards

def plot_results(episode_rewards, actor_losses, critic_losses, experiment_name):
    """Plot and save training results."""
    # Create plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)
    
    # Plot episode rewards
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards)
    plt.title(f'Episode Rewards - {experiment_name}')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.savefig(f"plots/rewards_{experiment_name}.png")
    plt.close()
    
    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(actor_losses, label='Actor Loss')
    plt.plot(critic_losses, label='Critic Loss')
    plt.title(f'Training Losses - {experiment_name}')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"plots/losses_{experiment_name}.png")
    plt.close()

if __name__ == "__main__":
    # Set environment
    env_name = "LunarLander-v2"
    
    # Run experiment 1: Default hyperparameters
    print("Running Experiment 1...")
    rewards1, actor_losses1, critic_losses1, agent1 = train(
        env_name, 
        num_episodes=300,
        hidden_dim=128,
        lr_actor=0.0003,
        lr_critic=0.001,
        gamma=0.99
    )
    plot_results(rewards1, actor_losses1, critic_losses1, "experiment1")
    
    # Run experiment 2: Different hyperparameters
    print("\nRunning Experiment 2...")
    rewards2, actor_losses2, critic_losses2, agent2 = train(
        env_name, 
        num_episodes=300,
        hidden_dim=256,  # Larger network
        lr_actor=0.0001,  # Lower learning rate
        lr_critic=0.0005,  # Lower learning rate
        gamma=0.99
    )
    plot_results(rewards2, actor_losses2, critic_losses2, "experiment2")
    
    # Compare results
    plt.figure(figsize=(10, 5))
    plt.plot(rewards1, label='Experiment 1')
    plt.plot(rewards2, label='Experiment 2')
    plt.title('Episode Rewards Comparison')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.savefig("plots/rewards_comparison.png")
    plt.close()
    
    # Evaluate the better agent
    print("\nEvaluating the trained agent...")
    better_agent = agent1 if np.mean(rewards1[-50:]) > np.mean(rewards2[-50:]) else agent2
    test_rewards = evaluate(env_name, better_agent, num_episodes=10)