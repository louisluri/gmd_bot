import torch
from torch.distributions import Categorical
import torch.nn.functional as F
import torch.optim as optim
import helpers as h

if torch.cuda.is_available():
    device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
else:
    device = torch.device("cpu")

code = "pg"
header = "episode,step,optim_step,percent,reward"

# Hyperparameters
batch_size = 4000 # how many target steps per update
gamma = 0.99  # Discount factor
lr = 0.0001  # Learning rate

saved_log_probs = [] # To store log probabilities of actions taken, used for calculating the policy gradient
rewards = [] # To store rewards for each episode, used for calculating returns
saved_returns = [] # To store the discounted returns for each episode, used for calculating the policy gradient

step_counter = 0
total_reward = 0
history = [] # To store episode, step, reward for plotting

m = None
o = None

save_folder = None

step = 0
episode = 0
optim_step = 0 # gradient step count

def init(model_class, train, load_file, folder, resume_level):
    global m, o
    global save_folder
    global step, episode, optim_step

    save_folder = folder

    m = model_class().to(device)

    if train:
        o = optim.Adam(m.parameters(), lr=lr)
    else:
        m.eval()

    if load_file is not None:
        print("loading cp...")
        checkpoint = torch.load(
            load_file,
            map_location=device,
            weights_only=True
        )

        # Restore model and optimizer
        m.load_state_dict(checkpoint["model"])

        if train:
            o.load_state_dict(checkpoint["optimizer"])

            for group in o.param_groups:
                group["lr"] = lr

            step = checkpoint['step']
            #optim_step = checkpoint["optim_step"]

            if resume_level:
                episode = checkpoint["episode"]
                
            print(f"Learning rate: {o.param_groups[0]['lr']:.8e}")

        print(f"Checkpoint loaded! Step: {step}, Episode: {episode}, Optim_step: {optim_step}")
    else:
        print("no cp found")

    print(f"Total parameters {sum(p.numel() for p in m.parameters()):,}\n")

def prepare(state):
    return state.to(device=device, dtype=torch.float32) / 255.0

@torch.no_grad()
def infer(state):
    logits = m(prepare(state))
    return logits.argmax(dim=-1).item()

def step_model(percent, reward, state, dead=False, won=False, quit=False, save_pending=False):

    global step, episode, optim_step
    global total_reward, step_counter
    global rewards, saved_log_probs, saved_returns, history

    if reward is not None:
        rewards.append(reward)
        total_reward += reward
        step += 1
        step_counter += 1

    if quit:
        return
    elif dead or won:
        episode += 1
        history.append([episode, step, optim_step, percent, total_reward])
        print(f"Episode {episode} complete! | Step: {step}, Total reward: {total_reward}")
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
            # Clip gradient norm to 1.0
            grad_norm = torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=1.0)
            o.step()

            saved_log_probs = []
            saved_returns = []
            step_counter = 0

            optim_step += 1
            print("Gradient update complete")
            print(f"Norm before clipping: {grad_norm.item():.4f}\n")

            if save_pending:
                h.save_checkpoint(m, o, step, episode, optim_step, history, save_folder)
                history = []
                return True # Tell "run.py" that the save was completed
    else:
        if state is None:
            raise ValueError("Player not dead or won or quit but no state passed!")
        
        logits = m(prepare(state))
        probs = F.softmax(logits, dim=-1)

        # Create a distribution object 
        # This "wraps" the probabilities but keeps the gradient link to the model weights
        dist = Categorical(probs)
        action = dist.sample()
        saved_log_probs.append(dist.log_prob(action))

        return action.item()