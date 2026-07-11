"""
Full Hearthstone Benchmark: All board sizes (2x2..7x7), all complexity levels (1..5)
Methods: ILP, Pure ML, Greedy, ML + ILP Repair.
Generates:
1. time_comparison.png (2x2 subplots): ILP, Greedy, ML, Repair runtime vs Board Size
2. level5_comparison.png (single plot): 4 methods at Level 5 vs Board Size
3. quality_comparison.png (2x2 subplots): ILP, Greedy, ML, Repair quality vs Board Size
"""

import pulp
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random
import os

os.makedirs("figures", exist_ok=True)

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

# ==================== CONFIGURATION ====================
MAX_M = 7
MAX_N = 10
MAX_ATTACK = 8
MAX_HEALTH = 10
MAX_WEIGHT = 5
PROB_DEATHRATTLE = 0.3
PROB_WINDFURY = 0.2
PROB_AURA = 0.2
PROB_RETALIATION = 0.2
PROB_HEALER = 0.2
MAX_SUMMON_HEALTH = 3
DEATHRATTLE_DEPTH = 2
NUM_TRAIN_SAMPLES = 20000
NUM_TEST_PER_SIZE = 200
EPOCHS = 80
BATCH_SIZE = 256
HIDDEN_DIM = 128
ML_PRED_THRESHOLD = 0.35
# =======================================================


# =============================================================
# PHASE 1: ILP SOLVER (WITH FIXED ACTION SUPPORT)
# =============================================================
def solve_hearthstone_ilp(attackers, defenders, weights, deathrattle, windfury,
                          summon_health, aura_buff, retaliation, healer,
                          m, n_orig, complexity_level, fixed_action=None):
    """
    complexity_level: 1..5
    If fixed_action is provided, x variables are fixed,
    and the solver only computes the objective value (no optimization).
    """
    # Build defender list (with summons for Level 3+)
    all_defenders = defenders[:]
    all_weights = weights[:]
    summon_indices = [-1] * n_orig

    if complexity_level >= 3:
        for j in range(n_orig):
            if deathrattle[j] and summon_health[j] > 0:
                summon_idx = len(all_defenders)
                all_defenders.append(summon_health[j])
                all_weights.append(0)
                summon_indices[j] = summon_idx

    n_all = len(all_defenders)
    if n_all > MAX_N:
        n_all = MAX_N
        all_defenders = all_defenders[:MAX_N]
        all_weights = all_weights[:MAX_N]

    prob = pulp.LpProblem("Hearthstone", pulp.LpMaximize)

    x = pulp.LpVariable.dicts("x", ((i, j) for i in range(m) for j in range(n_all)), cat='Binary')
    e = pulp.LpVariable.dicts("e", (j for j in range(n_all)), cat='Binary')

    if fixed_action is not None:
        for i in range(m):
            for j in range(n_all):
                idx = i * n_all + j
                if idx < len(fixed_action):
                    prob += x[i, j] == fixed_action[idx]
                else:
                    prob += x[i, j] == 0

    if complexity_level >= 2:
        shield_break = pulp.LpVariable.dicts("sb", (j for j in range(n_all)), cat='Binary')

    d = None
    if complexity_level >= 4:
        d = pulp.LpVariable.dicts("d", ((i, j) for i in range(m) for j in range(n_all)),
                                  lowBound=0, cat='Integer')
        M_big = 1000
        for i in range(m):
            for j in range(n_all):
                prob += d[i, j] <= attackers[i] * x[i, j]
                if complexity_level >= 4 and retaliation[j]:
                    prob += d[i, j] <= (attackers[i] - 1) * x[i, j] + M_big * (1 - x[i, j])

    eff_health = {}
    for j in range(n_all):
        base_hp = all_defenders[j]
        shield_bonus = 1 if complexity_level >= 2 else 0
        healing_bonus = 1 if (complexity_level >= 5 and healer[j]) else 0
        eff_health[j] = base_hp + shield_bonus + healing_bonus

    prob += pulp.lpSum(weights[j] * e[j] for j in range(min(n_orig, n_all)))

    for i in range(m):
        max_attacks = 2 if (complexity_level >= 2 and windfury[i]) else 1
        prob += pulp.lpSum(x[i, j] for j in range(n_all)) <= max_attacks

    for j in range(n_all):
        if complexity_level >= 2:
            prob += shield_break[j] <= pulp.lpSum(x[i, j] for i in range(m))
            if complexity_level >= 4:
                prob += pulp.lpSum(d[i, j] for i in range(m)) >= eff_health[j] * e[j]
            else:
                prob += pulp.lpSum(attackers[i] * x[i, j] for i in range(m)) >= eff_health[j] * e[j]
            prob += e[j] <= shield_break[j]
        else:
            if complexity_level >= 4:
                prob += pulp.lpSum(d[i, j] for i in range(m)) >= eff_health[j] * e[j]
            else:
                prob += pulp.lpSum(attackers[i] * x[i, j] for i in range(m)) >= eff_health[j] * e[j]
        if complexity_level >= 4:
            prob += pulp.lpSum(d[i, j] for i in range(m)) <= 1000 * e[j]
        else:
            prob += pulp.lpSum(attackers[i] * x[i, j] for i in range(m)) <= 1000 * e[j]

    if complexity_level >= 3:
        for j in range(min(n_orig, n_all)):
            if summon_indices[j] != -1 and summon_indices[j] < n_all:
                prob += e[j] <= e[summon_indices[j]]

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if prob.status != 1:
        return None, None, None

    action_flat = [0] * (MAX_M * MAX_N)
    for i in range(m):
        for j in range(min(n_all, MAX_N)):
            action_flat[i * MAX_N + j] = int(x[i, j].varValue)

    elim_flat = [0] * MAX_N
    for j in range(min(n_all, MAX_N)):
        elim_flat[j] = int(e[j].varValue)

    return pulp.value(prob.objective), action_flat, elim_flat


# =============================================================
# PHASE 2: GENERATE TRAINING DATA (FOR EACH COMPLEXITY LEVEL)
# =============================================================
def generate_training_data(complexity_level):
    X, y = [], []
    for _ in range(NUM_TRAIN_SAMPLES):
        m = random.randint(2, MAX_M)
        n_orig = random.randint(2, 7)
        attackers = [random.randint(1, MAX_ATTACK) for _ in range(m)]
        defenders = [random.randint(1, MAX_HEALTH) for _ in range(n_orig)]
        weights = [random.randint(1, MAX_WEIGHT) for _ in range(n_orig)]

        deathrattle = [False] * n_orig
        summon_health = [0] * n_orig
        if complexity_level >= 3:
            total_summons = 0
            for j in range(n_orig):
                if random.random() < PROB_DEATHRATTLE and total_summons < (MAX_N - n_orig):
                    deathrattle[j] = True
                    summon_health[j] = random.randint(1, MAX_SUMMON_HEALTH)
                    total_summons += 1

        windfury = [False] * m
        if complexity_level >= 2:
            for i in range(m):
                if random.random() < PROB_WINDFURY:
                    windfury[i] = True

        aura_buff = [False] * m
        if complexity_level >= 3:
            for i in range(m):
                if random.random() < PROB_AURA:
                    aura_buff[i] = True

        n_all_est = n_orig + sum(1 for h in summon_health if h > 0)
        if n_all_est > MAX_N:
            n_all_est = MAX_N

        retaliation = [False] * n_all_est
        healer = [False] * n_all_est
        if complexity_level >= 4:
            for j in range(n_all_est):
                if random.random() < PROB_RETALIATION:
                    retaliation[j] = True
        if complexity_level >= 5:
            for j in range(n_all_est):
                if random.random() < PROB_HEALER:
                    healer[j] = True

        _, action, _ = solve_hearthstone_ilp(attackers, defenders, weights,
                                             deathrattle, windfury, summon_health,
                                             aura_buff, retaliation, healer,
                                             m, n_orig, complexity_level)
        if action is None:
            continue

        n_all = n_orig + sum(1 for h in summon_health if h > 0)
        if n_all > MAX_N:
            n_all = MAX_N

        total_atk = sum(attackers)
        total_hp = sum(defenders)
        mean_atk = np.mean(attackers) if attackers else 0
        std_atk = np.std(attackers) if len(attackers) > 1 else 0
        mean_hp = np.mean(defenders) if defenders else 0
        std_hp = np.std(defenders) if len(defenders) > 1 else 0
        has_deathrattle = 1 if any(deathrattle) else 0
        has_windfury = 1 if any(windfury) else 0
        has_aura = 1 if any(aura_buff) else 0
        has_retaliation = 1 if any(retaliation) else 0
        has_healer = 1 if any(healer) else 0

        for i in range(m):
            for j in range(n_all):
                idx = i * MAX_N + j
                if idx >= len(action):
                    continue
                hp_val = defenders[j] if j < len(defenders) else summon_health[j - len(defenders)]
                w_val = weights[j] if j < len(weights) else 0
                feat = [
                    attackers[i] / MAX_ATTACK,
                    hp_val / MAX_HEALTH,
                    w_val / MAX_WEIGHT,
                    total_atk / (m * MAX_ATTACK),
                    total_hp / (n_orig * MAX_HEALTH),
                    mean_atk / MAX_ATTACK,
                    std_atk / MAX_ATTACK,
                    mean_hp / MAX_HEALTH,
                    std_hp / MAX_HEALTH,
                    m / MAX_M,
                    n_all / MAX_N,
                    has_deathrattle,
                    has_windfury,
                    has_aura,
                    has_retaliation,
                    has_healer,
                    int(deathrattle[j] if j < n_orig else False),
                    int(windfury[i]),
                    int(aura_buff[i]),
                    int(retaliation[j] if j < n_all else False),
                    int(healer[j] if j < n_all else False)
                ]
                X.append(feat)
                y.append(action[idx])

    if not X:
        return np.array([]), np.array([])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# =============================================================
# PHASE 3: NEURAL NETWORK
# =============================================================
class HearthstoneMLP(nn.Module):
    def __init__(self, input_dim=21, hidden_dim=128, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, output_dim)
        )

    def forward(self, x):
        return self.net(x)


def train_nn(X_train, y_train):
    if len(X_train) == 0:
        return None

    n_ones = np.sum(y_train)
    n_zeros = len(y_train) - n_ones
    pos_weight = torch.tensor([n_zeros / max(n_ones, 1)])
    print(f"  Class balance: zeros={n_zeros}, ones={n_ones}, pos_weight={pos_weight.item():.2f}")

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    dataset = TensorDataset(X_t, y_t)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = HearthstoneMLP(input_dim=X_train.shape[1])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                preds = (torch.sigmoid(model(X_t[:1000])) > 0.5).float()
                pos_pred = preds.sum().item()
                print(f"  Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(dataloader):.4f}, "
                      f"Positive preds: {pos_pred}/{len(preds)} ({100*pos_pred/len(preds):.1f}%)")
            model.train()

    return model


def nn_predict(model, attackers, defenders, weights, deathrattle, windfury,
               summon_health, aura_buff, retaliation, healer,
               m, n_orig, complexity_level):
    if model is None:
        return []

    n_all = n_orig + sum(1 for h in summon_health if h > 0)
    if n_all > MAX_N:
        n_all = MAX_N

    total_atk = sum(attackers)
    total_hp = sum(defenders)
    mean_atk = np.mean(attackers) if attackers else 0
    std_atk = np.std(attackers) if len(attackers) > 1 else 0
    mean_hp = np.mean(defenders) if defenders else 0
    std_hp = np.std(defenders) if len(defenders) > 1 else 0
    has_deathrattle = 1 if any(deathrattle) else 0
    has_windfury = 1 if any(windfury) else 0
    has_aura = 1 if any(aura_buff) else 0
    has_retaliation = 1 if any(retaliation) else 0
    has_healer = 1 if any(healer) else 0

    X_pred = []
    for i in range(m):
        for j in range(n_all):
            hp_val = defenders[j] if j < len(defenders) else summon_health[j - len(defenders)]
            w_val = weights[j] if j < len(weights) else 0
            feat = [
                attackers[i] / MAX_ATTACK,
                hp_val / MAX_HEALTH,
                w_val / MAX_WEIGHT,
                total_atk / (m * MAX_ATTACK),
                total_hp / (n_orig * MAX_HEALTH),
                mean_atk / MAX_ATTACK,
                std_atk / MAX_ATTACK,
                mean_hp / MAX_HEALTH,
                std_hp / MAX_HEALTH,
                m / MAX_M,
                n_all / MAX_N,
                has_deathrattle,
                has_windfury,
                has_aura,
                has_retaliation,
                has_healer,
                int(deathrattle[j] if j < n_orig else False),
                int(windfury[i]),
                int(aura_buff[i]),
                int(retaliation[j] if j < n_all else False),
                int(healer[j] if j < n_all else False)
            ]
            X_pred.append(feat)

    X_t = torch.tensor(X_pred, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        logits = model(X_t)
        probs = torch.sigmoid(logits).numpy().flatten()

    threshold = ML_PRED_THRESHOLD
    return [1 if p > threshold else 0 for p in probs]


# =============================================================
# PHASE 4: REPAIR 
# =============================================================
def repair_ilp(attackers, defenders, weights, deathrattle, windfury,
               summon_health, aura_buff, retaliation, healer,
               m, n_orig, ml_action, complexity_level):
    n_all = n_orig + sum(1 for h in summon_health if h > 0)
    if n_all > MAX_N:
        n_all = MAX_N

    summon_indices = [-1] * n_orig
    next_summon_idx = n_orig
    for j in range(n_orig):
        if deathrattle[j] and summon_health[j] > 0:
            summon_indices[j] = next_summon_idx
            next_summon_idx += 1

    edges = []
    for i in range(m):
        for j in range(n_all):
            idx = i * n_all + j
            if ml_action and idx < len(ml_action) and ml_action[idx] == 1:
                edges.append((i, j))

    if not edges:
        return [0] * (m * n_all), 0

    prob = pulp.LpProblem("Repair", pulp.LpMaximize)
    y = pulp.LpVariable.dicts("y", range(len(edges)), cat='Binary')
    e = pulp.LpVariable.dicts("e", (j for j in range(n_all)), cat='Binary')

    prob += pulp.lpSum(weights[j] * e[j] for j in range(min(n_orig, n_all)))

    eff_health = {}
    for j in range(n_all):
        base_hp = defenders[j] if j < len(defenders) else summon_health[j - len(defenders)]
        shield_bonus = 1 if complexity_level >= 2 else 0
        healing_bonus = 1 if (complexity_level >= 5 and healer[j]) else 0
        eff_health[j] = base_hp + shield_bonus + healing_bonus

    for i in range(m):
        max_attacks = 2 if (complexity_level >= 2 and windfury[i]) else 1
        prob += pulp.lpSum(y[k] for k, (ii, jj) in enumerate(edges) if ii == i) <= max_attacks

    for j in range(n_all):
        total_damage = pulp.lpSum(
            (attackers[i] - (1 if (complexity_level >= 4 and retaliation[j] and attackers[i] > 0) else 0)) * y[k]
            for k, (i, jj) in enumerate(edges) if jj == j
        )
        prob += total_damage >= eff_health[j] * e[j]
        prob += total_damage <= 1000 * e[j]

    if complexity_level >= 3:
        for j in range(min(n_orig, n_all)):
            if summon_indices[j] != -1 and summon_indices[j] < n_all:
                prob += e[j] <= e[summon_indices[j]]

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if prob.status != 1:
        return [0] * (m * n_all), 0

    repaired = [0] * (m * n_all)
    for k, (i, j) in enumerate(edges):
        if y[k].varValue > 0.5:
            repaired[i * n_all + j] = 1
    return repaired, 0


# =============================================================
# HELPER FUNCTIONS
# =============================================================
def check_feasibility(action, attackers, windfury, m, n_all):
    if not action:
        return False
    used = [0] * m
    for i in range(m):
        for j in range(n_all):
            idx = i * n_all + j
            if idx < len(action) and action[idx] == 1:
                used[i] += 1
        max_attacks = 2 if windfury[i] else 1
        if used[i] > max_attacks:
            return False
    return True


def compute_value(action, defenders, weights, attackers, m, n_all, n_orig):
    if not action:
        return 0
    damage = [0] * n_all
    for i in range(m):
        for j in range(n_all):
            idx = i * n_all + j
            if idx < len(action) and action[idx] == 1:
                damage[j] += attackers[i]
    value = 0
    for j in range(min(n_orig, n_all)):
        if damage[j] >= defenders[j]:
            value += weights[j]
    return value


def evaluate_action_with_full_ilp(action, attackers, defenders, weights, deathrattle,
                                  windfury, summon_health, aura_buff, retaliation,
                                  healer, m, n_orig, complexity_level):
    val, _, _ = solve_hearthstone_ilp(attackers, defenders, weights, deathrattle,
                                      windfury, summon_health, aura_buff, retaliation,
                                      healer, m, n_orig, complexity_level,
                                      fixed_action=action)
    if val is None:
        return 0.0
    return val


def generate_random_board(m, n_orig, complexity_level):
    attackers = [random.randint(1, MAX_ATTACK) for _ in range(m)]
    defenders = [random.randint(1, MAX_HEALTH) for _ in range(n_orig)]
    weights = [random.randint(1, MAX_WEIGHT) for _ in range(n_orig)]

    deathrattle = [False] * n_orig
    summon_health = [0] * n_orig
    if complexity_level >= 3:
        total_summons = 0
        for j in range(n_orig):
            if random.random() < PROB_DEATHRATTLE and total_summons < (MAX_N - n_orig):
                deathrattle[j] = True
                summon_health[j] = random.randint(1, MAX_SUMMON_HEALTH)
                total_summons += 1

    windfury = [False] * m
    if complexity_level >= 2:
        for i in range(m):
            if random.random() < PROB_WINDFURY:
                windfury[i] = True

    aura_buff = [False] * m
    if complexity_level >= 3:
        for i in range(m):
            if random.random() < PROB_AURA:
                aura_buff[i] = True

    n_all_est = n_orig + sum(1 for h in summon_health if h > 0)
    if n_all_est > MAX_N:
        n_all_est = MAX_N

    retaliation = [False] * n_all_est
    healer = [False] * n_all_est
    if complexity_level >= 4:
        for j in range(n_all_est):
            if random.random() < PROB_RETALIATION:
                retaliation[j] = True
    if complexity_level >= 5:
        for j in range(n_all_est):
            if random.random() < PROB_HEALER:
                healer[j] = True

    return attackers, defenders, weights, deathrattle, summon_health, windfury, aura_buff, retaliation, healer


# =============================================================
# BASELINE: GREEDY
# =============================================================
def greedy_solve(attackers, defenders, weights, deathrattle, windfury,
                 summon_health, aura_buff, retaliation, healer,
                 m, n_orig, complexity_level):
    n_all = n_orig + sum(1 for h in summon_health if h > 0)
    if n_all > MAX_N:
        n_all = MAX_N
    edges = []
    for i in range(m):
        for j in range(n_all):
            hp = defenders[j] if j < len(defenders) else summon_health[j - len(defenders)]
            w = weights[j] if j < len(weights) else 0
            damage = attackers[i]
            if hp > 0:
                if damage >= hp:
                    score = w * 1000 + (1.0 / hp)
                else:
                    score = w / hp
            else:
                score = 0
            edges.append((i, j, score))
    edges.sort(key=lambda x: x[2], reverse=True)
    action = [0] * (m * n_all)
    attacker_count = [0] * m
    for i, j, score in edges:
        max_attacks = 2 if (complexity_level >= 2 and windfury[i]) else 1
        if attacker_count[i] < max_attacks and score > 0:
            action[i * n_all + j] = 1
            attacker_count[i] += 1
    return action


# =============================================================
# MAIN
# =============================================================
if __name__ == "__main__":
    print("=" * 80)
    print("Full Benchmark: All board sizes (2x2..7x7), all complexity levels (1..5)")
    print("=" * 80)

    complexity_levels = [
        (1, "Level 1 (Basic)"),
        (2, "Level 2 (+Shield+Windfury)"),
        (3, "Level 3 (+Deathrattle+Aura)"),
        (4, "Level 4 (+Retaliation)"),
        (5, "Level 5 (+Healing Aura)"),
    ]
    board_sizes = [(2,2), (3,3), (4,4), (5,5), (6,6), (7,7)]
    board_labels = [f"{m}x{n}" for m,n in board_sizes]

    all_data = {}

    for level, level_name in complexity_levels:
        print(f"\n{'='*80}")
        print(f"Training and testing: {level_name}")
        print(f"{'='*80}")

        print("[Phase 2] Generating training data...")
        X_train, y_train = generate_training_data(level)
        print(f"  Training samples: {len(X_train)}, features: {X_train.shape[1] if len(X_train)>0 else 0}")

        print("[Phase 3] Training Neural Network...")
        model = train_nn(X_train, y_train)
        if model is None:
            print(f"  No training data for level {level}. Skipping...")
            continue
        print("  Model trained!")

        print("[Phase 4] Evaluating...")
        level_data = {}
        for m, n_orig in board_sizes:
            label = f"{m}x{n_orig}"
            ilp_times, ml_times, greedy_times, repair_times = [], [], [], []
            ilp_vals, ml_vals, greedy_vals, repair_vals = [], [], [], []
            ml_feas, greedy_feas, repair_feas = [], [], []

            for _ in range(NUM_TEST_PER_SIZE):
                attackers, defenders, weights, dr, sh, wf, aura, ret, healer = generate_random_board(m, n_orig, level)
                n_all = n_orig + sum(1 for h in sh if h > 0)
                if n_all > MAX_N:
                    n_all = MAX_N

                # ILP
                start = time.perf_counter()
                ilp_val, _, _ = solve_hearthstone_ilp(attackers, defenders, weights, dr, wf, sh, aura, ret, healer, m, n_orig, level)
                ilp_t = (time.perf_counter() - start) * 1000
                if ilp_val is None:
                    continue
                ilp_times.append(ilp_t)
                ilp_vals.append(ilp_val)

                # Pure ML
                start = time.perf_counter()
                ml_action = nn_predict(model, attackers, defenders, weights, dr, wf, sh, aura, ret, healer, m, n_orig, level)
                ml_t = (time.perf_counter() - start) * 1000
                ml_times.append(ml_t)
                ml_feas.append(1 if check_feasibility(ml_action, attackers, wf, m, n_all) else 0)
                ml_vals.append(compute_value(ml_action, defenders, weights, attackers, m, n_all, n_orig))

                # Greedy
                start = time.perf_counter()
                greedy_action = greedy_solve(attackers, defenders, weights, dr, wf, sh, aura, ret, healer, m, n_orig, level)
                greedy_t = (time.perf_counter() - start) * 1000
                greedy_times.append(greedy_t)
                greedy_feas.append(1 if check_feasibility(greedy_action, attackers, wf, m, n_all) else 0)
                greedy_vals.append(compute_value(greedy_action, defenders, weights, attackers, m, n_all, n_orig))

                # ML + ILP Repair
                start = time.perf_counter()
                rep_action, _ = repair_ilp(attackers, defenders, weights, dr, wf, sh, aura, ret, healer, m, n_orig, ml_action, level)
                rep_t = (time.perf_counter() - start) * 1000
                repair_times.append(rep_t)
                repair_feas.append(1 if check_feasibility(rep_action, attackers, wf, m, n_all) else 0)
                rep_val = evaluate_action_with_full_ilp(
                    rep_action, attackers, defenders, weights,
                    dr, wf, sh, aura, ret, healer,
                    m, n_orig, level
                )
                repair_vals.append(rep_val)

            level_data[label] = {
                'ilp_time': np.mean(ilp_times),
                'ml_time': np.mean(ml_times),
                'greedy_time': np.mean(greedy_times),
                'repair_time': np.mean(repair_times),
                'ilp_val': np.mean(ilp_vals),
                'ml_val': np.mean(ml_vals),
                'greedy_val': np.mean(greedy_vals),
                'repair_val': np.mean(repair_vals),
                'ml_feas': np.mean(ml_feas) * 100,
                'greedy_feas': np.mean(greedy_feas) * 100,
                'repair_feas': np.mean(repair_feas) * 100,
            }
            print(f"  {label}: ILP={np.mean(ilp_times):.2f}ms, Repair={np.mean(repair_times):.2f}ms, ML Feas={np.mean(ml_feas)*100:.1f}%")

        all_data[level_name] = level_data

    # =============================================================
    # PRINT SUMMARY TABLE (7x7 only)
    # =============================================================
    print("\n\n" + "=" * 120)
    print("SUMMARY TABLE: Performance on 7x7 boards across all complexity levels")
    print("=" * 120)
    print(f"{'Level':<30} {'ILP(ms)':<10} {'ML(ms)':<8} {'ML Feas%':<10} {'Greedy(ms)':<10} {'Greedy%':<10} {'Repair(ms)':<10} {'Repair%':<10} {'Rep Feas%':<10}")
    print("-" * 120)
    for level_name, data in all_data.items():
        board_7x7 = data['7x7']
        ilp_time = board_7x7['ilp_time']
        ml_time = board_7x7['ml_time']
        ml_feas = board_7x7['ml_feas']
        greedy_time = board_7x7['greedy_time']
        greedy_quality = 100 * board_7x7['greedy_val'] / board_7x7['ilp_val'] if board_7x7['ilp_val'] > 0 else 0
        repair_time = board_7x7['repair_time']
        repair_quality = 100 * board_7x7['repair_val'] / board_7x7['ilp_val'] if board_7x7['ilp_val'] > 0 else 0
        repair_feas = board_7x7['repair_feas']
        print(f"{level_name:<30} {ilp_time:<10.2f} {ml_time:<8.2f} {ml_feas:<10.1f} {greedy_time:<10.2f} {greedy_quality:<10.1f} {repair_time:<10.2f} {repair_quality:<10.1f} {repair_feas:<10.1f}")

    print("=" * 120)

    # =============================================================
    # PLOTTING
    # =============================================================
    board_labels = ['2x2', '3x3', '4x4', '5x5', '6x6', '7x7']
    level_names = list(all_data.keys())
    level_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    level_labels = ['L1 (Basic)', 'L2 (+Shield+WF)', 'L3 (+DR+Aura)', 'L4 (+Retal.)', 'L5 (+Heal)']
    method_colors = {'ILP': '#D55E00', 'Greedy': '#CC79A7', 'ML': '#0072B2', 'Repair': '#009E73'}

    def extract_data(level_name, board_label, metric):
        return all_data[level_name][board_label][metric]

    def extract_quality(level_name, board_label, method):
        ilp_val = all_data[level_name][board_label]['ilp_val']
        if method == 'ilp':
            return 100.0
        elif method == 'repair':
            val = all_data[level_name][board_label]['repair_val']
        elif method == 'greedy':
            val = all_data[level_name][board_label]['greedy_val']
        elif method == 'ml':
            val = all_data[level_name][board_label]['ml_val']
        else:
            return 0.0
        return 100 * val / ilp_val if ilp_val > 0 else 0.0

    # ---------------------------
    # FIGURE 1: time_comparison (2x2 subplots)
    # ---------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Runtime Comparison Across Complexity Levels', fontsize=16, y=0.98)

    # (a) ILP Runtime
    ax = axes[0, 0]
    for idx, level_name in enumerate(level_names):
        times = [extract_data(level_name, b, 'ilp_time') for b in board_labels]
        ax.plot(board_labels, times, 'o-', lw=2, markersize=6, color=level_colors[idx], label=level_labels[idx])
    ax.set_xlabel('Board Size')
    ax.set_ylabel('ILP Time (ms)')
    ax.set_title('(a) ILP Runtime')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    # (b) Greedy Runtime
    ax = axes[0, 1]
    for idx, level_name in enumerate(level_names):
        times = [extract_data(level_name, b, 'greedy_time') for b in board_labels]
        ax.plot(board_labels, times, 'o-', lw=2, markersize=6, color=level_colors[idx], label=level_labels[idx])
    ax.set_xlabel('Board Size')
    ax.set_ylabel('Greedy Time (ms)')
    ax.set_title('(b) Greedy Runtime')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    # (c) ML Runtime
    ax = axes[1, 0]
    for idx, level_name in enumerate(level_names):
        times = [extract_data(level_name, b, 'ml_time') for b in board_labels]
        ax.plot(board_labels, times, 'o-', lw=2, markersize=6, color=level_colors[idx], label=level_labels[idx])
    ax.set_xlabel('Board Size')
    ax.set_ylabel('ML Time (ms)')
    ax.set_title('(c) ML Runtime')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    # (d) Repair Runtime
    ax = axes[1, 1]
    for idx, level_name in enumerate(level_names):
        times = [extract_data(level_name, b, 'repair_time') for b in board_labels]
        ax.plot(board_labels, times, 'o-', lw=2, markersize=6, color=level_colors[idx], label=level_labels[idx])
    ax.set_xlabel('Board Size')
    ax.set_ylabel('Repair Time (ms)')
    ax.set_title('(d) ML+ILP Repair Runtime')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/time_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Figure 1 saved: figures/time_comparison.png")

    # ---------------------------
    # FIGURE 2: level5_comparison (single plot)
    # ---------------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    level5_name = level_names[-1]
    ilp_times = [extract_data(level5_name, b, 'ilp_time') for b in board_labels]
    repair_times = [extract_data(level5_name, b, 'repair_time') for b in board_labels]
    greedy_times = [extract_data(level5_name, b, 'greedy_time') for b in board_labels]
    ml_times = [extract_data(level5_name, b, 'ml_time') for b in board_labels]
    ax.plot(board_labels, ilp_times, 'o-', lw=2.5, markersize=9, color=method_colors['ILP'], label='ILP Exact')
    ax.plot(board_labels, repair_times, 'D-', lw=2.5, markersize=9, color=method_colors['Repair'], label='ML + ILP Repair')
    ax.plot(board_labels, greedy_times, '^-', lw=2.5, markersize=9, color=method_colors['Greedy'], label='Greedy')
    ax.plot(board_labels, ml_times, 's-', lw=2.5, markersize=9, color=method_colors['ML'], label='Pure ML', alpha=0.7)
    ax.set_xlabel('Board Size')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Runtime Comparison at Level 5 (Highest Complexity)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/level5_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Figure 2 saved: figures/level5_comparison.png")

    # ---------------------------
    # FIGURE 3: quality_comparison (2x2 subplots)
    # ---------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Solution Quality Across Complexity Levels', fontsize=16, y=0.98)

    # (a) ILP Quality (always 100%)
    ax = axes[0, 0]
    for idx, level_name in enumerate(level_names):
        quality = [extract_quality(level_name, b, 'ilp') for b in board_labels]
        ax.plot(board_labels, quality, 'o-', lw=2, markersize=6, color=level_colors[idx], label=level_labels[idx])
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.7, label='Optimal (100%)')
    ax.set_xlabel('Board Size')
    ax.set_ylabel('Quality (% of Optimal)')
    ax.set_title('(a) ILP Quality')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(95, 105)  # ILP is always exactly 100%

    # (b) Greedy Quality
    ax = axes[0, 1]
    for idx, level_name in enumerate(level_names):
        quality = [extract_quality(level_name, b, 'greedy') for b in board_labels]
        ax.plot(board_labels, quality, 'o-', lw=2, markersize=6, color=level_colors[idx], label=level_labels[idx])
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.7, label='Optimal (100%)')
    ax.set_xlabel('Board Size')
    ax.set_ylabel('Quality (% of Optimal)')
    ax.set_title('(b) Greedy Quality')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    # (c) ML Quality
    ax = axes[1, 0]
    for idx, level_name in enumerate(level_names):
        quality = [extract_quality(level_name, b, 'ml') for b in board_labels]
        ax.plot(board_labels, quality, 'o-', lw=2, markersize=6, color=level_colors[idx], label=level_labels[idx])
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.7, label='Optimal (100%)')
    ax.set_xlabel('Board Size')
    ax.set_ylabel('Quality (% of Optimal)')
    ax.set_title('(c) Pure ML Quality')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 150)

    # (d) Repair Quality
    ax = axes[1, 1]
    for idx, level_name in enumerate(level_names):
        quality = [extract_quality(level_name, b, 'repair') for b in board_labels]
        ax.plot(board_labels, quality, 'o-', lw=2, markersize=6, color=level_colors[idx], label=level_labels[idx])
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.7, label='Optimal (100%)')
    ax.set_xlabel('Board Size')
    ax.set_ylabel('Quality (% of Optimal)')
    ax.set_title('(d) ML+ILP Repair Quality')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig('figures/quality_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Figure 3 saved: figures/quality_comparison.png")

    print("\n✅ All figures generated in 'figures/' directory:")
    print("  - time_comparison.png  (2x2: ILP, Greedy, ML, Repair)")
    print("  - level5_comparison.png (single: 4 methods at Level 5)")
    print("  - quality_comparison.png (2x2: ILP, Greedy, ML, Repair)")