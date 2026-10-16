You are my hackathon pair-programmer for tomorrow. Read this entire context before doing anything.

### WHO I AM
- 2nd-year Mechanical Engineering, IIT Bombay
- Codeforces Candidate Master, strong in Python + C++, comfortable with PyTorch/TensorFlow, ML/DL fundamentals
- Built low-latency C++ simulation engines and the ML models for SIH 2026 Nationals project
- Assume solid coding + ML literacy. Keep explanations concise, don't over-explain basics, but DO flag non-obvious tradeoffs (especially CPU vs GPU, sample efficiency, encoding decisions)
- Team: 4 members finalized - Atharv (me, 2nd yr Mech), Shrot Vijayvergiya (Econ, strong CV/ML), Barad Dhanush (Electrical Dual, Systems/AI), Ishaan Singla (Civil, Data Analysis). Team name: "Attention is All We Need" (registered 13 Oct 00:57, approved 13 Oct 22:08, 64 teams total, application-based selection). WhatsApp group active, no new messages since 14 Oct 00:07.

### THE HACKATHON - Hyde × WnCC AI Builder Hackathon
- Organizers: Hyde (applied AI lab, founders ex-Palantir + ex-McKinsey, talent from Google DeepMind/Optiver, backed by General Catalyst + Bain Capital Ventures, specialization: post-training with RL) + WnCC (Web and Coding Club, IITB)
- Venue: LA001, IIT Bombay
- Date: Saturday 17 Oct 2026, 10 AM - 8 PM (10 hours, one day)
- Schedule:
  09:15-09:50 Registration (tell name + team name + show college ID, collect attendance card, find placard - ALL 4 must arrive TOGETHER)
  10:00 Welcome
  10:05 What is Hyde, and why Hyde
  10:20 IITB @ Hyde
  10:30 Problem statement walkthrough (MOST IMPORTANT 20 MIN)
  10:50 Q&A
  11:00 Build begins
  13:00-13:30 Lunch (on them)
  13:30 Build continues
  17:00 Judging begins (+ snacks)
  18:00 Top 6 announced
  18:20 Top 6 present
  19:25 Winners
  19:30 Prizes/photos
  19:45 Wrap-up
- Prizes: ₹2.5L cash + internship/job opportunities for top performers + unlimited Hyde Inference tokens till end 2026 + goodies/refreshments
- Ground rules: Be on time (registration closes 9:50 sharp), stay till end, play fair (build your own solution, no copying), keep LA001 clean, phones on silent, mentors from Hyde on floor all day, follow WhatsApp/Luma updates. T&C: All work must be built DURING the hackathon - pre-built solutions disqualified. Judges final. Hyde may use resumes for hiring.
- Links: Luma https://luma.com/sd8b1f0r, Updates WhatsApp https://chat.whatsapp.com/LKkIipw3okmE46AiXl4BHg, Hyde Inference https://inference.hyde.tech/?auth=signin (sign in with IITB email + OTP, set password) and OpenCode setup https://inference.hyde.tech/docs/opencode. If not provisioned, fill https://docs.google.com/forms/d/e/1FAIpQLScJaJX8nknhrDOOLWJwI94AFVKYmNtjKeQ8jilSTpkX9xbCww/viewform. Help: inference@hyde.ai
- Challenge core: "Train a small transformer to play a game using only a CPU — No GPUs, No giant models, just clever engineering, experimentation, and teamwork." Contest is explicitly about squeezing performance from small models under tight compute.

### THE GAME - CATAN (revealed today 14 Oct 8 AM by Arushi Goyal)
Message: "THE game is CATAN 🎲 - Hey Settlers! Tomorrow you'll build a bot that can sit at the table and trade, bluff, and build its way to victory. It goes head-to-head with our reference bots and every other team in the room. The board is full of dice rolls, sneaky trades, and things you can't see coming, so your bot will have to make smart moves with incomplete info and limited compute. Use search, self-play, learned value functions, imitation, or that wild idea you had in the shower. AI coding assistants are totally welcome. Your bot is judged on how consistently it wins, plus how well you can explain the strategy behind it. Full problem statement, constraints, rules, and eval details drop on hackathon day. One day. Limited compute. May the best strategy win! 🏆"
Rules deck (11-page PDF: Hyde@IITB.pdf, Drive https://drive.google.com/file/d/1ftEFhs8wcwiJ5NvM5524NWubRgbnT_OE/view): This is a 2-PLAYER simplified Catan variant (not 3-4 player):
- Board: Hexagonal island (~19 tiles), each hex has terrain + number (2-12)
- 5 Resources: Wood (Forest), Brick (Hills), Sheep/Wool (Pasture), Wheat/Grain (Fields), Ore (Mountains)
- Turn = Roll 2 dice (2-12) -> collect from adjacent settlements/cities if number hits. Example roll 8 shown. Then Trade, then Build.
- Building costs (Page 6): Road = Brick+Wood (0 VP, connects network), Settlement = Brick+Wood+Sheep+Wheat (+1 VP, collects resources, distance rule), City = 2 Wheat + 3 Ore (upgrades settlement, produces 2 resources, +2 VP), Dev Card = Sheep+Wheat+Ore
- Dev Cards: Knight (move robber + steal 1 resource), Road Building (2 free roads), Year of Plenty (take any 2 resources), Victory Point (hidden +1 VP)
- Robber: On roll 7, blocks a tile (no production) and steal 1 resource from opponent. No discard-half rule shown in deck.
- Trading: Bank 4:1, Ports 3:1 (any) and 2:1 (specific). Deck ONLY shows bank/ports - player-to-player trading NOT shown, must confirm tomorrow whether p2p is allowed.
- Win: First to 10 VP via Settlement +1, City +2, VP card +1, Largest Army (3 Knights = +2), Longest Road (5 connected segments = +2). Note Hyde uses 3 knights / 5 roads - shorter than some house rules, very contestable in 2P.

KEY PROPERTIES FOR AI: Partially observable (hidden opponent hand + hidden dev cards), stochastic (dice), 3-4 player normally but this is 2P (simpler diplomacy, single opponent), hierarchical action space (where to build road/settlement/city? which dev card? where to place robber? what to trade 4:1/3:1/2:1? - not a flat 7 actions).

### CRITICAL: OFFICIAL PROBLEM STATEMENT NOT YET RELEASED
The full problem statement, constraints (max model size, inference time per move, is transformer mandatory or suggested), rules, and eval details (how many games, vs whom, round-robin vs elimination, seed handling) DROP TOMORROW at 10:30 in LA001. The Catan deck was only a preview. We must parse the statement in 20 min at 10:30 and adapt. Do NOT assume action/observation API.

### WHAT WE HAVE BUILT TILL NOW (FROZEN SCAFFOLD)
Folder: "/Users/atharv/hyde wncc hackathon" - now clean (nanoGPT and decision-transformer clones deleted after reference, only stubs removed tonight 23:39). Remaining:
- scaffold/envs/__init__.py - REGISTRY + make(name) so swapping games = one file
- scaffold/envs/tictactoe.py - 3x3 perfect solver, for smoke tests
- scaffold/envs/connect4.py - 6x7 + negamax solver for expert data
- scaffold/models/transformer.py - TinyTransformer: causal, obs->d_model projection + pos_emb + TransformerEncoder, params ~102k at d_model=64 n_layer=2 n_head=2 max_len=16 (~104k for connect4). Has act() for inference, param_count().
- scaffold/data.py - generate_trajectories(env_fn, n_games, solver_depth, opponent) + BCTrajectoryDataset (sliding windows, padded batches via collate_fn)
- scaffold/train.py - CPU loop: torch.set_num_threads(4), AdamW, cross-entropy on last token of sequence, saves scaffold/best.pt, reports train/val loss/acc and samp/s. Args: --env --n_games --epochs --d_model --n_layer --n_head --max_len --batch_size
- scaffold/eval.py - Evaluate vs random/solver, handles illegal moves via fallback, but had shape mismatch if ckpt from different env - retrain before eval.
- scaffold/README.md - usage
- PLAN_TOMORROW.md (14KB) - full game-day plan (see below)

Verified on CPU (torch 2.14.1 CPU-only, cuda:False, 4 threads, no Vercel CLI needed, gymnasium 1.4.0 + pettingzoo 1.27.0 installed):
- tictactoe: 300 trajs in 0.6s, 102k params, ~8k samp/s, val acc ~0.51, eval W89 D11 L0 (89% win vs random) - BC learns
- connect4: 300 trajs in 53s (SOLVER-BOUND, depth 3 bottleneck - hackathon fix: depth=2 or mixed opponent), ~5.8k samp/s training (0.7s/epoch at 64d)
- Checkpoint is shared scaffold/best.pt - overwrites if training both envs, use separate ckpt names on the day.

This scaffold is GENERIC (interface = reset/step/legal_actions/is_done/winner/encode) - allowed despite "pre-built disqualified" because it's not a Catan solution. Tomorrow we add ONE file scaffold/envs/catan.py wrapping THEIR env. Training must happen AFTER 11:00 (keep git commits timestamped after 11:00 as proof).

### PLAN FOR TOMORROW (from PLAN_TOMORROW.md - summarized)
TONIGHT: All 4 sign into Hyde Inference + OpenCode with IITB email, join WhatsApp, skim Hyde@IITB.pdf page 6 (building costs), charge laptops/phones, meet at 8:45 to reach LA001 9:15 together with college IDs + headphones/mouse.

10:30-10:50: One person SCRIBES verbatim. Extract 6 fields: 1) Env API (do they give pip package? example code?), 2) Action space (flat int vs hierarchical dict? trading representation? p2p allowed?), 3) Observation space (what is visible? opponent resource counts or just hand_size? dev card counts? robber pos?), 4) Evaluation (how many games, vs reference bots + round-robin? Elo? seeds fixed or random?), 5) Constraints (max params? inference ms/move? is transformer required or "small model" suggested?), 6) Starter code/reference bots. Ask at 10:50 if not answered: seeds deterministic? trading rules? inference limit / is transformer mandatory?

11:00-13:00 PHASE A - HEURISTIC BASELINE (do NOT train until this wins):
Split 2+2: Atharv+Barad = Env Wrapping (45 min) -> create scaffold/envs/catan.py wrapping THEIR env to our interface, register in REGISTRY, make python -m scaffold.train --env catan not crash. Shrot+Ishaan = bots/heuristic.py - rule-based bot using ONLY legal_actions(): Priority: 1) City if possible (fast VP, doubles production), 2) Settlement scored by dice probability weights (6,8=5 | 5,9=4 | 4,10=3...), 3) Knight if can steal (progress to 3 = +2), 4) Road toward best settlement or 4th/5th for Longest Road, 5) Trade: if need 1 resource for City/Settlement and have 4 of another -> 4:1, if have 2:1 port excess -> use it, else pass, 6) Robber: block opponent's best hex (most settlements on 6/8). Test at 12:30: heuristic vs random, heuristic vs heuristic, must finish with no illegal moves. Fallback at 13:00 if wrapping stuck: 2 fix env, 2 keep tuning heuristic - heuristic alone can win.

13:30-16:00 PHASE B - TRANSFORMER RANKER (CPU-realistic: BC on self-play, NOT online RL/DT):
Why: PPO is sample-inefficient on CPU, DT return-conditioning noisy with dice. BC on heuristic vs heuristic = generate once (cheap) + supervised batches (cache-friendly, ~5k samp/s). Steps: 1) Generate 1000 heuristic vs heuristic trajectories (target ~15k steps, 40-60 moves/game in 2P, if slow drop to 500) - modify data.py to call heuristic.act() not env.best_action(). 2) Model: TinyTransformer obs_dim~50-100, d_model 64-128, n_layer 2-3, n_head 2-4, max_len 32 (not 64, Catan games short, shorter= faster + less dice overfit), params 80k-150k, must be <50ms inference. Mask illegal logits to -inf. 3) Train: python -m scaffold.train --env catan --n_games 1000 --epochs 8 --d_model 64 --n_layer 2 --batch_size 32 (expect ~15-20s training, data gen dominates). Val acc 0.45-0.65 is good for Catan (stochastic). 4) Inference: candidates = legal_actions() -> if >20 filter heuristic topk to 8 -> transformer scores candidates -> argmax -> fallback random if illegal. Key pattern: enumerate -> filter -> rank, not generate. 5) Optional one-iteration self-play loop (BC vs Heuristic -> retrain) only if time after 15:00.

16:00-17:00 PHASE C - HARDEN + PITCH:
Harden (30 min): Run 50 games each heuristic vs heuristic, transformer vs heuristic, transformer vs random, log win rates, fix illegal moves, tune Longest Road (if at 4/5 prioritize road) and Largest Army (if at 2/3 prioritize knights), quantize only if >100ms/move. Pitch (30 min): Prepare 2-min talk: 1) Problem framing (2P Catan is partially observable+stochastic so history-dependent ranking, not perfect search, 15s), 2) Architecture (heuristic filter 15->8 + tiny causal transformer 64d/2L ~100k via BC on 1000 self-play trajs, CPU 8k samp/s <20s training, 30s), 3) Catan insights (exploit 2P: 5 roads + 3 knights short so race them, greedy 4:1/2:1 trader only to complete City/Settlement beats learned trading on CPU, 30s), 4) Results (heuristic beats random 70%+, transformer adds X% vs heuristic, zero illegal in 100 games, 15s). Have slide: legal_actions -> filter -> transformer -> action + encode() vector layout + win-rate table.

State Encoding encode() vector (~60-80 dim): [0:5] your resources (wood,brick,sheep,wheat,ore)/10, [5:10] opp resources or [hand_size/10,0,0,0,0] if hidden, [10:12] VPs/10, [12:14] Largest Army/Longest Road flags, [14:16] dev cards, [16:18] robber pos + last dice one-hot, [18:48] board graph flattened (your settlement/city/road vs opp), [48:60] port ownership + turn + last 6 dice. Don't encode opp hidden hand exactly - use hand_size+dev_count proxies, let transformer learn history.

Risks: API is weird -> use their example bot as template; transformer won't train (val 0.3) -> SUBMIT HEURISTIC (tuned heuristic that wins 60% and explainable beats broken transformer); env bug -> split team; inference slow -> drop to 64d/2L/max_len 16, prune candidates to 5; disqualification fear -> no weights from tonight, train after 11:00.

YOUR JOB NOW: When I paste the official problem statement tomorrow at ~10:30, immediately parse it into those 6 fields, tell me the 3 most critical decisions, and help me adapt scaffold/envs/catan.py + encode() + heuristic in that order. Constraints: CPU-only (mirror hackathon, no CUDA), keep models small (80-150k), Python/PyTorch main + C++ for hot paths if needed, honest benchmarking. Ask me only what you need.