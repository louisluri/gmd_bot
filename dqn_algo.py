import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import copy
import random
import helpers as h

if torch.cuda.is_available():
    device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
else:
    device = torch.device("cpu")

code = "dqn"
header = "episode,step,optim_step,eps,percent,reward"

def e_greedy_action_selection(action_values, eps):
    if random.random() < eps:
        return random.randint(0, action_values.size(-1) - 1)  # Explore
    else:
        return torch.argmax(action_values, dim=-1).item()  # Exploit

def soft_update(model, target_model, tau):
    for target_param, param in zip(target_model.parameters(), model.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

memory = deque(maxlen=50000)

# Hyperparameters
batch_size = 128
min_to_train = 2000
train_steps = 200000 # How many steps to decay epsilon for
gamma = 0.99  # Discount factor
lr = 0.001  # Learning rate
eps_max = 0.9  # Exploration rate
eps_min = 0.05
tau = 0.005 # Small value, e.g., 0.5% update

last_state = None
last_action = None
current_eps = None

total_reward = 0

level_steps = 0 # tracks steps in the current run
history = [] # To store episode, step, epsilon, reward for plotting

m = None
o = None
target_m = None

save_folder = None

step = 0
episode = 0
optim_step = 0 # gradient step count

def init(model_class, train, load_file, folder, resume_level):

    global m, o, target_m
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
            optim_step = checkpoint["optim_step"]

            if resume_level:
                episode = checkpoint["episode"]

            print(f"Learning rate: {o.param_groups[0]['lr']:.8e}")

        print(f"Checkpoint loaded! Step: {step}, Episode: {episode}, Optim_step: {optim_step}")
    else:
        print("no cp found")

    if train:
        target_m = copy.deepcopy(m).to(device)
        
        # Ensure the target model is in eval mode and doesn't track gradients
        target_m.eval()
        for param in target_m.parameters():
            param.requires_grad = False

    print(f"Total parameters {sum(p.numel() for p in m.parameters()):,}\n")

def prepare(state):
    return state.to(device=device, dtype=torch.float32) / 255.0

@torch.no_grad()
def infer(state):
    q_values = m(prepare(state))
    return q_values.argmax(dim=-1).item()

def step_model(percent, reward, state, dead=False, won=False, quit=False, save_pending=False):

    global step, episode, optim_step
    global total_reward, level_steps, history
    global last_state, last_action, current_eps

    if reward is not None:
        if state is None:
            memory.append((last_state, last_action, reward, torch.zeros_like(last_state), (dead or won)))
        else:
            memory.append((last_state, last_action, reward, state, (dead or won)))
        total_reward += reward
        level_steps += 1
        step += 1

    if quit:
        return
    elif dead or won:
        episode += 1
        diff = torch.norm(torch.stack([torch.norm(p1 - p2) for p1, p2 in zip(m.parameters(), target_m.parameters())]))
        history.append([episode, step, optim_step, current_eps, percent, total_reward])
        print(f"Episode {episode} complete! | Step: {step}, Epsilon (ε): {current_eps:.4f}, Diff: {diff:.4f}, Total reward: {total_reward}")
        total_reward = 0
        last_state = None
        last_action = None

    # train on data thats in memory
    if level_steps >= min_to_train:
        batch = random.sample(memory, batch_size)

        states, actions, rewards, next_states, terminateds = zip(*batch)

        state_batch = prepare(torch.cat(states, dim=0))
        action_batch = torch.tensor(actions).unsqueeze(1) # Needs to be [[a1], [a2]...]
        reward_batch = torch.tensor(rewards).float()
        next_state_batch = prepare(torch.cat(next_states, dim=0))
        terminal_batch = torch.tensor(terminateds).float()

        q_values = m(state_batch).gather(1, action_batch.to(device)).squeeze(1)

        target_m.eval()
        with torch.no_grad():
            next_q_values = target_m(next_state_batch).max(1)[0]

        target_qs = reward_batch.clone().to(device)
        non_final_mask = (terminal_batch == 0)
        target_qs[non_final_mask] += gamma * next_q_values[non_final_mask]

        l = F.smooth_l1_loss(q_values, target_qs) # Huber loss, less sensitive to outliers than MSE
        o.zero_grad()
        l.backward()
        o.step()

        soft_update(m, target_m, tau)

        optim_step += 1

    if dead or won:
        if save_pending:
            h.save_checkpoint(m, o, step, episode, optim_step, history, save_folder)
            history = []
            return True
    else:
        current_eps = max(eps_min, eps_max - (eps_max - eps_min) * (level_steps / train_steps))  # Linear decay of epsilon

        m.eval()  # Set to evaluation mode
        with torch.no_grad(): # Dont waste resources on calculating gradients
            q_values = m(prepare(state))
        m.train()
        
        action = e_greedy_action_selection(q_values, current_eps)

        last_state = state.detach().cpu().clone()
        last_action = action
        return action