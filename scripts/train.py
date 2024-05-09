#%%
import torch
import numpy as np

%load_ext autoreload
%autoreload 2 
from agents import GNN, Critic
from environment import CustomEnvironment

env = CustomEnvironment()
actor = GNN()
critic = Critic()

actor_optimizer = torch.optim.Adam(actor.parameters(), lr=0.001)
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=0.001)

"""
Actor should output [E,] array with expected reward for each edge "action"
Critic should output scalar value with expected reward for each state "graph"
"""

# Main training loop
num_episodes = 5
gamma = 0.99
#%%

env.set_params()
data, reward, termination, _ = env.reset()

#%%
"""
State:  (4,)
Action probs:  torch.Size([2])
Action:  0
torch.Size([1])
State value:  tensor([24.5412], grad_fn=<ViewBackward0>)
Advantage:  tensor([0.4657], grad_fn=<SubBackward0>)
tensor([-0.], grad_fn=<MulBackward0>)
tensor([0.2169], grad_fn=<PowBackward0>)
"""

for episode in range(num_episodes):
    data, episode_reward, termination, _ = env.reset()
    episode_reward = 0

    while not env.termination:  # Limit the number of time steps
        # Choose an action using the actor
        action_probs = torch.nn.Softmax()(actor(data))
        print("Action probs: ", action_probs.shape)
        num_edges = len(action_probs)
        action = np.random.choice(num_edges, p=action_probs.detach().numpy())
        print("Action: ", action)

        # Take the chosen action and observe the next state and reward
        next_data, reward, termination, _ = env.step(action)

        # Compute the advantage
        c1, c2, c3 = critic(data)
        state_value = c1.mean() + c2.mean() + c3.mean()
        c1, c2, c3 = critic(next_data)
        next_state_value = c1.mean() + c2.mean() + c3.mean() # NOTE: do not do .detach().item() here in order to keep grad_fn
        advantage = reward + gamma * next_state_value - state_value
        print("Advantage: ", advantage)

        # Compute actor and critic losses
        p = torch.nn.Softmax()(action_probs[action])
        actor_loss = -torch.log(p) * advantage
        print(actor_loss)
        critic_loss = torch.square(advantage)
        print(critic_loss)

        actor_loss.backward(retain_graph=True) # needed: otherwise intermediary results get deleted before critic_loss can backprop
        actor_optimizer.step()

        critic_loss.backward()
        critic_optimizer.step()

        episode_reward += reward

        if done:
            break

if episode % 10 == 0:
    print(f"Episode {episode}, Reward: {episode_reward}")

env.close()
# %%
