import torch
from torch.distributions import Categorical
import torch.nn.functional as F
import torch.optim as optim

algo_code = "pg"
header = "episode,step,percent,reward"

# Hyperparameters
batch_size = 2000 # how many target steps per update
gamma = 0.95  # Discount factor
lr = 0.0005  # Learning rate

total_reward = 0
episode = 0
last_sucessful = 0

saved_log_probs = [] # To store log probabilities of actions taken, used for calculating the policy gradient
rewards = [] # To store rewards for each episode, used for calculating returns
saved_returns = [] # To store the discounted returns for each episode, used for calculating the policy gradient

step_counter = 0
history = [] # To store episode, step, reward for plotting

m = None
o = None

def init(model_class, cp_path, device, train):
    global m, o, lr, episode

    m = model_class().to(device)

    if train:
        o = optim.Adam(m.parameters(), lr=lr)
    else:
        m.eval()

    if cp_path is not None:
        print("loading cp...")
        checkpoint = torch.load(
            cp_path,
            map_location=device,
            weights_only=True
        )

        # Restore model and optimizer
        m.load_state_dict(checkpoint["model"])

        if train:
            o.load_state_dict(checkpoint["optimizer"])

        episode = checkpoint["episode"]

        print(f"Checkpoint loaded! Step: {checkpoint['step']}, Episode: {episode}")

        return checkpoint["step"]
    else:
        print("no cp found")
        return 0

def prepare(state, device):
    return state.to(device=device, dtype=torch.float32) / 255.0

@torch.no_grad()
def infer(device, state):
    logits = m(prepare(state, device))
    return logits.argmax(dim=-1).item()

def step(device, step, percent, reward, state, dead=False, won=False, quit=False):

    global total_reward, episode, step_counter, last_sucessful
    global rewards, saved_log_probs, saved_returns

    if reward is not None:
        rewards.append(reward)

        total_reward += reward

        step_counter += 1

    if quit:
        return
    elif dead or won:
        episode += 1
        history.append([episode, step, percent, total_reward])
        print(f"Episode {episode} complete! | Step: {step}, Percent {percent}%, Total reward: {total_reward}")
        total_reward = 0

        # Calculate discounted returns
        G = 0
        returns = []
        for r in reversed(rewards):
            G = r + gamma * G
            returns.append(G)
        returns.reverse()
        saved_returns.extend(returns)
        rewards = [] # Clear rewards for the next episode

        if step_counter >= batch_size:
            assert len(saved_log_probs) == len(saved_returns)

            returns = torch.tensor(saved_returns).to(device)
            returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)

            log_probs = torch.cat(saved_log_probs)

            loss = -(log_probs * returns).sum() # Eτ∼πθ​​[t=0∑T ​∇θ​logπθ​(at​∣st​)⋅Gt​] (negative for gradient descent)
            o.zero_grad()
            loss.backward()
            o.step()

            saved_log_probs = []
            saved_returns = []
            step_counter = 0
            last_sucessful = step
    else:
        if state is None:
            raise ValueError("Player not dead or won or quit but no state passed!")
        
        logits = m(prepare(state, device))
        probs = F.softmax(logits, dim=-1)

        # Create a distribution object 
        # This "wraps" the probabilities but keeps the gradient link to the model weights
        dist = Categorical(probs)
        action = dist.sample()
        saved_log_probs.append(dist.log_prob(action))

        return action.item()