"""
Hearthstone Benchmark: COMPLEXITY SCALING (5 Levels) - EACH LEVEL TRAINED
FIX: Repair quality evaluated using FULL ILP model (no >100% issue)
FIX: Deathrattle summon indices correctly mapped in repair_ilp
FIX: Plot y-axis for quality set to 0..105
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
NUM_TEST_PER_SIZE = 30
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
    If fixed_action is provided (list of length m*n_all), x variables are fixed,
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

    # If fixed_action is provided, fix x
    if fixed_action is not None:
        for i in range(m):
            for j in range(n_all):
                idx = i * n_all + j
                if idx < len(fixed_action):
                    prob += x[i, j] == fixed_action[idx]
                else:
                    prob += x[i, j] == 0

    # Shield variables for Level 2+
    if complexity_level >= 2:
        shield_break = pulp.LpVariable.dicts("sb", (j for j in range(n_all)), cat='Binary')

    # Damage variables for Level 4+
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

    # Effective health
    eff_health = {}
    for j in range(n_all):
        base_hp = all_defenders[j]
        shield_bonus = 1 if complexity_level >= 2 else 0
        healing_bonus = 1 if (complexity_level >= 5 and healer[j]) else 0
        eff_health[j] = base_hp + shield_bonus + healing_bonus

    # Objective
    prob += pulp.lpSum(weights[j] * e[j] for j in range(min(n_orig, n_all)))

    # Attack constraints
    for i in range(m):
        max_attacks = 2 if (complexity_level >= 2 and windfury[i]) else 1
        prob += pulp.lpSum(x[i, j] for j in range(n_all)) <= max_attacks

    # Damage & Elimination
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
        # Prevent damage without elimination
        if complexity_level >= 4:
            prob += pulp.lpSum(d[i, j] for i in range(m)) <= 1000 * e[j]
        else:
            prob += pulp.lpSum(attackers[i] * x[i, j] for i in range(m)) <= 1000 * e[j]

    # Deathrattle
    if complexity_level >= 3:
        for j in range(min(n_orig, n_all)):
            if summon_indices[j] != -1 and summon_indices[j] < n_all:
                prob += e[j] <= e[summon_indices[j]]

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if prob.status != 1:
        return None, None, None

    # Extract solution
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

        # Abilities based on complexity level
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
# PHASE 4: REPAIR (FIXED: CORRECT DEATHRATTLE MAPPING)
# =============================================================
def repair_ilp(attackers, defenders, weights, deathrattle, windfury,
               summon_health, aura_buff, retaliation, healer,
               m, n_orig, ml_action, complexity_level):
    n_all = n_orig + sum(1 for h in summon_health if h > 0)
    if n_all > MAX_N:
        n_all = MAX_N

    # 正确构建 summon_indices 映射
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

    # 亡语约束 — 使用正确的 summon_indices
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

def evaluate_action_with_full_ilp(action, attackers, defenders, weights, deathrattle,
                                  windfury, summon_health, aura_buff, retaliation,
                                  healer, m, n_orig, complexity_level):
    """
    Evaluate a given action using the FULL ILP model (same as optimal).
    Returns the objective value (guaranteed <= ilp_val).
    """
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
# MAIN: TRAIN ON EACH LEVEL, TEST ON SAME LEVEL
# =============================================================
if __name__ == "__main__":
    print("=" * 80)
    print("Hearthstone Benchmark: COMPLEXITY SCALING (5 Levels)")
    print("TRAIN & TEST: each level separately")
    print("FIX: Repair quality evaluated using FULL ILP model (no >100%)")
    print("FIX: Deathrattle mapping in repair is now correct")
    print("=" * 80)

    complexity_levels = [
        (1, "Level 1 (Basic)"),
        (2, "Level 2 (+Shield+Windfury)"),
        (3, "Level 3 (+Deathrattle+Aura)"),
        (4, "Level 4 (+Retaliation)"),
        (5, "Level 5 (+Healing Aura)"),
    ]

    test_configs = [
        (2, 2, "2x2"),
        (3, 3, "3x3"),
        (4, 4, "4x4"),
        (5, 5, "5x5"),
        (6, 6, "6x6"),
        (7, 7, "7x7"),
    ]

    all_results = {}

    for level, level_name in complexity_levels:
        print(f"\n{'='*80}")
        print(f"Running {level_name}...")
        print(f"{'='*80}")

        print("\n[Phase 2] Generating training data...")
        X_train, y_train = generate_training_data(level)
        print(f"  Training samples: {len(X_train)}, features: {X_train.shape[1] if len(X_train)>0 else 0}")

        print("\n[Phase 3] Training Neural Network...")
        model = train_nn(X_train, y_train)
        if model is None:
            print(f"  No training data for level {level}. Skipping...")
            continue
        print("  Model trained!")

        print("\n[Phase 4] Evaluating...")
        print("-" * 100)
        print(f"{'Board':<8} {'ILP(ms)':<12} {'Repair(ms)':<12} {'ILP Feas':<12} {'Rep Feas':<12} {'Rep Quality':<12}")
        print("-" * 100)

        level_results = {'boards': [], 'ilp_times': [], 'repair_times': [], 'quality': []}

        for m, n_orig, label in test_configs:
            ilp_times, rep_times = [], []
            ilp_vals, rep_vals = [], []
            rep_feas = []

            for _ in range(NUM_TEST_PER_SIZE):
                attackers, defenders, weights, dr, sh, wf, aura, ret, healer = generate_random_board(m, n_orig, level)
                n_all = n_orig + sum(1 for h in sh if h > 0)
                if n_all > MAX_N:
                    n_all = MAX_N

                # ILP (gold standard)
                start = time.perf_counter()
                ilp_val, _, _ = solve_hearthstone_ilp(attackers, defenders, weights, dr, wf, sh, aura, ret, healer, m, n_orig, level)
                ilp_t = (time.perf_counter() - start) * 1000
                if ilp_val is None:
                    continue
                ilp_times.append(ilp_t)
                ilp_vals.append(ilp_val)

                # ML prediction
                ml_action = nn_predict(model, attackers, defenders, weights, dr, wf, sh, aura, ret, healer, m, n_orig, level)

                # Repair (with fixed deathrattle mapping)
                start = time.perf_counter()
                rep_action, _ = repair_ilp(attackers, defenders, weights, dr, wf, sh, aura, ret, healer, m, n_orig, ml_action, level)
                rep_t = (time.perf_counter() - start) * 1000
                rep_times.append(rep_t)
                rep_feas.append(1 if check_feasibility(rep_action, attackers, wf, m, n_all) else 0)

                # Evaluate repair with full ILP
                rep_val = evaluate_action_with_full_ilp(
                    rep_action, attackers, defenders, weights,
                    dr, wf, sh, aura, ret, healer,
                    m, n_orig, level
                )
                rep_vals.append(rep_val)

            if not ilp_times:
                continue

            avg_ilp = np.mean(ilp_times)
            avg_rep = np.mean(rep_times)
            avg_ilp_val = np.mean(ilp_vals)
            avg_rep_val = np.mean(rep_vals)
            avg_quality = 100 * avg_rep_val / avg_ilp_val if avg_ilp_val > 0 else 0

            level_results['boards'].append(label)
            level_results['ilp_times'].append(avg_ilp)
            level_results['repair_times'].append(avg_rep)
            level_results['quality'].append(avg_quality)

            print(f"{label:<8} {avg_ilp:<12.2f} {avg_rep:<12.2f} "
                  f"{'100.0%':<12} {np.mean(rep_feas)*100:<12.1f}% {avg_quality:<12.1f}%")

        all_results[level_name] = level_results
        print("=" * 100)

    # =============================================================
    # PLOTTING
    # =============================================================
    if all_results:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442']
        markers = ['o', 's', 'D', '^', 'p']

        # (a) ILP Time
        ax1 = axes[0, 0]
        for idx, (level_name, data) in enumerate(all_results.items()):
            ax1.plot(data['boards'], data['ilp_times'],
                     marker=markers[idx], linestyle='-', linewidth=2, markersize=8,
                     label=f'{level_name}', color=colors[idx])
        ax1.set_xlabel('Board Size', fontsize=12)
        ax1.set_ylabel('ILP Time (ms)', fontsize=12)
        ax1.set_title('(a) ILP Time by Complexity Level', fontsize=14)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        # (b) Repair Time
        ax2 = axes[0, 1]
        for idx, (level_name, data) in enumerate(all_results.items()):
            ax2.plot(data['boards'], data['repair_times'],
                     marker=markers[idx], linestyle='-', linewidth=2, markersize=8,
                     label=f'{level_name}', color=colors[idx])
        ax2.set_xlabel('Board Size', fontsize=12)
        ax2.set_ylabel('Repair Time (ms)', fontsize=12)
        ax2.set_title('(b) ML+ILP Repair Time by Complexity Level', fontsize=14)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        # (c) ILP vs Repair (Level 5)
        ax3 = axes[1, 0]
        if 'Level 5 (+Healing Aura)' in all_results:
            data = all_results['Level 5 (+Healing Aura)']
            ax3.plot(data['boards'], data['ilp_times'], 'o-', linewidth=2, markersize=8,
                     label='ILP Exact (Level 5)', color='#D55E00')
            ax3.plot(data['boards'], data['repair_times'], 'D-', linewidth=2, markersize=8,
                     label='ML + ILP Repair (Level 5)', color='#009E73')
            ax3.set_xlabel('Board Size', fontsize=12)
            ax3.set_ylabel('Time (ms)', fontsize=12)
            ax3.set_title('(c) ILP vs Repair (Highest Complexity)', fontsize=14)
            ax3.legend(fontsize=10)
            ax3.grid(True, alpha=0.3)

        # (d) Repair Quality (full y-axis 0..105)
        ax4 = axes[1, 1]
        for idx, (level_name, data) in enumerate(all_results.items()):
            ax4.plot(data['boards'], data['quality'],
                     marker=markers[idx], linestyle='-', linewidth=2, markersize=8,
                     label=f'{level_name}', color=colors[idx])
        ax4.axhline(y=100, color='black', linestyle='--', alpha=0.5, label='Optimal (100%)')
        ax4.set_xlabel('Board Size', fontsize=12)
        ax4.set_ylabel('Repair Quality (% of Opt)', fontsize=12)
        ax4.set_title('(d) Repair Quality (Full ILP Evaluation)', fontsize=14)
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 105)   # Now shows all levels including low quality

        plt.tight_layout()
        plt.savefig('complexity_scaling_5levels_fixed.png', dpi=300)
        print("\n✅ Plot saved as 'complexity_scaling_5levels_fixed.png'")