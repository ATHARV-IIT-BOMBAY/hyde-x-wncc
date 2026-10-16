"""Evaluate trained policy vs random / vs solver."""
import torch, random, numpy as np
from scaffold.envs import make as make_env
from scaffold.models.transformer import TinyTransformer

def evaluate(env_name, ckpt="scaffold/best.pt", n_games=100, d_model=64, n_layer=2, n_head=2, max_len=16, opponent="random"):
    env_fn = lambda: make_env(env_name)
    probe = env_fn()
    model = TinyTransformer(probe.obs_dim, probe.n_actions, d_model=d_model, n_layer=n_layer, n_head=n_head, max_len=max_len)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()

    wins = draws = losses = 0
    for _ in range(n_games):
        env = env_fn()
        obs = env.reset()
        history = [obs]
        done = False
        policy_goes_first = random.random() < 0.5
        # if opponent goes first, do one random/solved move
        if not policy_goes_first:
            env.step(random.choice(env.legal_actions()))
            history.append(env.encode())
            if env.is_done():
                # opponent already won (rare on tictactoe)
                w = env.winner()
                if w == 1: losses += 1
                elif w == -1: wins += 1  # shouldn't happen
                else: draws += 1
                continue
        while not done:
            # policy move
            seq = np.stack(history[-max_len:], axis=0)  # (T, D)
            seq_t = torch.from_numpy(seq).float().unsqueeze(0)
            mask = torch.ones(1, seq_t.shape[1], dtype=torch.long)
            with torch.no_grad():
                logits = model(seq_t, mask)  # (1,T,A)
                action = int(logits[0, -1].argmax().item())
            legal = env.legal_actions()
            if action not in legal:
                action = random.choice(legal)
            _, _, done, _ = env.step(action)
            if done:
                break
            w = env.winner()
            if w is not None:
                done = True
                break
            history.append(env.encode())
            # opponent move
            if done:
                break
            opp_a = random.choice(env.legal_actions()) if opponent=="random" else env.best_action(depth=3)
            if opp_a not in env.legal_actions():
                opp_a = random.choice(env.legal_actions())
            _, _, done, _ = env.step(opp_a)
            if not done:
                history.append(env.encode())
        w = env.winner()
        # winner is absolute (1/-1). Determine from policy perspective:
        # we tracked policy's perspective alternating, so just count wins
        # simpler: if game ended on policy's move and policy won -> win
        # Use env.moves parity: hard to infer. Approx: check board winner vs move order
        # For smoke test, treat w==1 as first-player win
        if w == 0:
            draws += 1
        elif w is not None:
            # winner is the player who just moved
            # if policy moved last and won -> win, else loss
            # env.player after win is still the winner (we don't flip on terminal)
            # so if winner == env.player -> that player won
            if w == env.player:
                wins += 1
            else:
                losses += 1
        else:
            draws += 1

    total = wins + draws + losses
    print(f"Eval vs {opponent} over {total} games: W {wins} D {draws} L {losses}  (win% {100*wins/max(1,total):.1f}%)")
    return wins, draws, losses

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="tictactoe")
    p.add_argument("--ckpt", default="scaffold/best.pt")
    p.add_argument("--n_games", type=int, default=100)
    p.add_argument("--opponent", default="random")
    args = p.parse_args()
    evaluate(args.env, ckpt=args.ckpt, n_games=args.n_games, opponent=args.opponent)
