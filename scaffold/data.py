"""Data pipeline: generate expert trajectories -> BC dataset."""
import random, numpy as np, torch
from torch.utils.data import Dataset

def generate_trajectories(env_fn, n_games=1000, solver_depth=4, opponent="solver"):
    """
    Roll out games where one or both players use env.best_action().
    Returns list of trajectories: each is list of (obs, action, reward).
    """
    trajs = []
    for _ in range(n_games):
        env = env_fn()
        obs = env.reset()
        traj = []
        done = False
        # alternate who starts randomly
        while not done:
            legal = env.legal_actions()
            # expert move (solver) vs random opponent
            use_expert = True
            if opponent == "mixed":
                # 50% expert, 50% random -> diverse data
                use_expert = random.random() < 0.7
            elif opponent == "random":
                use_expert = random.random() < 0.5
            if use_expert and hasattr(env, "best_action"):
                try:
                    a = env.best_action(depth=solver_depth)
                    if a not in legal:
                        a = random.choice(legal)
                except Exception:
                    a = random.choice(legal)
            else:
                a = random.choice(legal)
            next_obs, reward, done, info = env.step(a)
            traj.append((obs.copy(), a, reward))
            obs = next_obs
        trajs.append(traj)
    return trajs

class BCTrajectoryDataset(Dataset):
    """Sliding windows over trajectories for BC.
    Each sample: (obs_seq (T, obs_dim), target_action)
    We predict the *next* action given history; target is last action.
    For simplicity, seq_len = full trajectory truncated/padded to max_len.
    """
    def __init__(self, trajs, max_len=32):
        self.samples = []
        for traj in trajs:
            obs_list = [o for o,a,r in traj]
            act_list = [a for o,a,r in traj]
            # for each prefix length >=1, predict the action at that step
            # need at least 1 obs to predict 1 action
            for t in range(len(traj)):
                # sequence up to t inclusive
                seq = np.stack(obs_list[max(0, t - max_len + 1): t+1], axis=0)  # (L, D)
                # pad if shorter than max_len? no, variable length handled by mask
                target = act_list[t]
                self.samples.append((seq, target))

    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        seq, target = self.samples[i]
        return torch.from_numpy(seq).float(), torch.tensor(target, dtype=torch.long)

def collate_fn(batch):
    """Pad obs_seqs to max T in batch."""
    seqs, targets = zip(*batch)
    T_max = max(s.shape[0] for s in seqs)
    D = seqs[0].shape[1]
    B = len(seqs)
    padded = torch.zeros(B, T_max, D)
    mask = torch.zeros(B, T_max, dtype=torch.long)
    for i, s in enumerate(seqs):
        T = s.shape[0]
        padded[i, :T] = s
        mask[i, :T] = 1
    targets = torch.stack(targets)
    return padded, mask, targets
