"""CPU training loop: BC with cross-entropy, masked seq loss."""
import time, torch, torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from scaffold.data import generate_trajectories, BCTrajectoryDataset, collate_fn
from scaffold.envs import make as make_env
from scaffold.models.transformer import TinyTransformer

def train(env_name="tictactoe", n_games=500, max_len=16,
          d_model=64, n_layer=2, n_head=2,
          batch_size=32, epochs=5, lr=3e-4, val_frac=0.15, num_workers=0):
    # --- threads (CPU hackathon realism) ---
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)

    env_fn = lambda: make_env(env_name)
    probe = env_fn()
    obs_dim, n_actions = probe.obs_dim, probe.n_actions
    print(f"Env={env_name} obs_dim={obs_dim} n_actions={n_actions}")

    # --- data ---
    t0 = time.time()
    trajs = generate_trajectories(env_fn, n_games=n_games, solver_depth=3, opponent="mixed")
    print(f"Generated {len(trajs)} trajs, {sum(len(t) for t in trajs)} steps in {time.time()-t0:.1f}s")

    ds = BCTrajectoryDataset(trajs, max_len=max_len)
    n_val = max(1, int(len(ds)*val_frac))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)
    print(f"Dataset: {len(train_ds)} train / {len(val_ds)} val samples")

    # --- model ---
    model = TinyTransformer(obs_dim, n_actions, d_model=d_model, n_layer=n_layer, n_head=n_head, max_len=max_len)
    print(f"Params: {model.param_count():,}  (d_model={d_model} n_layer={n_layer} n_head={n_head})")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    # --- loop ---
    best = float("inf")
    for epoch in range(1, epochs+1):
        model.train()
        t_loss, t_acc, t_n = 0.0, 0, 0
        t_start = time.time()
        for obs_seq, mask, target in train_loader:
            logits = model(obs_seq, mask)  # (B,T,A)
            # we predict action for last position only: gather last valid
            # simpler: loss only on last token of each sequence
            # find last valid index per row
            last_idx = mask.sum(dim=1) - 1  # (B,)
            last_logits = logits[torch.arange(len(logits)), last_idx]  # (B,A)
            loss = F.cross_entropy(last_logits, target)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            # stats
            t_loss += loss.item() * len(target)
            t_acc += (last_logits.argmax(dim=-1) == target).sum().item()
            t_n += len(target)

        # val
        model.eval()
        v_loss, v_acc, v_n = 0.0, 0, 0
        with torch.no_grad():
            for obs_seq, mask, target in val_loader:
                logits = model(obs_seq, mask)
                last_idx = mask.sum(dim=1) - 1
                last_logits = logits[torch.arange(len(logits)), last_idx]
                v_loss += F.cross_entropy(last_logits, target).item() * len(target)
                v_acc += (last_logits.argmax(dim=-1) == target).sum().item()
                v_n += len(target)

        dt = time.time() - t_start
        print(f"epoch {epoch:02d}  train loss {t_loss/t_n:.4f} acc {t_acc/t_n:.3f}"
              f"  |  val loss {v_loss/v_n:.4f} acc {v_acc/v_n:.3f}  ({dt:.1f}s, {t_n/dt:.0f} samp/s)")
        if v_loss/v_n < best:
            best = v_loss/v_n
            torch.save(model.state_dict(), "scaffold/best.pt")
            print("  -> saved scaffold/best.pt")

    print(f"Done. Best val loss {best:.4f}")
    return model

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="tictactoe")
    p.add_argument("--n_games", type=int, default=500)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--n_layer", type=int, default=2)
    p.add_argument("--n_head", type=int, default=2)
    p.add_argument("--max_len", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=32)
    args = p.parse_args()
    train(env_name=args.env, n_games=args.n_games, epochs=args.epochs,
          d_model=args.d_model, n_layer=args.n_layer, n_head=args.n_head,
          max_len=args.max_len, batch_size=args.batch_size)
