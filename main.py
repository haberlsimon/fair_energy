import pulp
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations

#Anzahl Prosumer
P = 4
#Anzahl Zeitschritte
T = 5
#Export and Curtail, 0 bedeutet deaktiviert
EC = 0

p_DA = [2, 16, 1, 10, 1]   # Day-Ahead Preise wie in fairness by design von Zoe Fornier
p_B  = [6, 25, 5, 15, 5]   # Balancing Preise

q_DA_min_trade = 11
q_min = [0, 5, 0, 2]
q_max = [5, 5, 4, 3]

#Acceptability Verstärker Alpha
alpha = 1

# Eigenbedarf (über alle T Perioden, z. B. MWh)
own_demand_total = [10, 25,  8, 15]

#Erzeugerprofile (Erste Dimension Prosumer, Zweite Dimension Zeitschritte)
pv     = [[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]]

p_feed = [1, 12, 0, 8, 0]          # Einspeisetarif [€/MWh]
r_curt = [1.1, 13, 0.1, 8.5, 0.1]        # Vergütung für Verzicht (feed-in + kleiner Bonus)
K_export = [10, 10, 8, 10, 10]  # Export-Kappe je Periode (Beispiel)

# Individuelle Optimierung (Pi)
def solve_individual(i):
    m = pulp.LpProblem(f"Indiv_{i}", pulp.LpMinimize)
    qB  = pulp.LpVariable.dicts("qB",  range(T), lowBound=0)
    use_pv = pulp.LpVariable.dicts("use_pv", range(T), lowBound=0)
    export   = pulp.LpVariable.dicts("export",   range(T), lowBound=0)
    curtail  = pulp.LpVariable.dicts("curtail",  (range(T)), lowBound=0)

    # Zielfunktion: nur Einkaufskosten (PV hat hier keine direkten Kosten)
    m += pulp.lpSum(
        p_B[t]*qB[t]
        - p_feed[t]*export[t]*EC
        - r_curt[t]*curtail[t]*EC
        for t in range(T)
        )   
    
    m += pulp.lpSum((qB[t] + use_pv[t]) for t in range (T)) == own_demand_total[i]

    for t in range(T):
        
        m += use_pv[t]  <= pv[i][t]  # PV-Nutzung limitiert durch vorhandene PV

        # Per-Perioden-Grenzen
        m += qB[t]  >= q_min[i]
        m += qB[t]  <= q_max[i]

        

    # (Gesamtbedarfsgleichung ist implizit erfüllt durch die Summation der Periodenbilanzen)
    status = m.solve(pulp.PULP_CBC_CMD(msg=0))
    v_total = pulp.value(m.objective)
    v_stage = [pulp.value(
        p_B[t]*qB[t] 
        - p_feed[t]*export[t]*EC
        - r_curt[t]*curtail[t]*EC
        ) for t in range(T)
    ]
    allocations = {
        "qB": [pulp.value(qB[t]) for t in range(T)],
        "use_pv": [pulp.value(use_pv[t]) for t in range(T)],
        "export": [pulp.value(export[t]) for t in range(T)],
    }
    return v_total, v_stage, allocations

# Individuelle Kosten
v_individual, v_stagewise = [], []
for i in range(P):
    v_total, v_stage, _ = solve_individual(i)
    v_individual.append(v_total)
    v_stagewise.append(v_stage)

def solve_aggregation(mode="utilitarian", accept_type=None):
    m = pulp.LpProblem(f"Agg_{mode}", pulp.LpMinimize)

    qDA = pulp.LpVariable.dicts("qDA", (range(P), range(T)), lowBound=0)
    qB  = pulp.LpVariable.dicts("qB",  (range(P), range(T)), lowBound=0)
    use_pv = pulp.LpVariable.dicts("use_pv", (range(P), range(T)), lowBound=0)
    export   = pulp.LpVariable.dicts("export",   (range(P), range(T)), lowBound=0)
    curtail  = pulp.LpVariable.dicts("curtail",  (range(P), range(T)), lowBound=0)
    qDA_total = pulp.LpVariable.dicts("qDA_total", range(T), lowBound=0)
    bDA = pulp.LpVariable.dicts("bDA", range(T), cat="Binary")

    # Kosten pro Prosumer (nur Einkauf)
    costs = {}
    for i in range (P):
        costs[i] = pulp.lpSum(
        (p_DA[t]*qDA[i][t] + p_B[t]*qB[i][t]
        #- p_feed[t]*export[i][t]*EC
        #- r_curt[t]*curtail[i][t]*EC
        )
        for t in range(T)
        )
    costs_stage = {
        (i,t): p_DA[t]*qDA[i][t] + p_B[t]*qB[i][t]
              #- p_feed[t]*export[i][t]*EC
              #- r_curt[t]*curtail[i][t]*EC
        for i in range(P) for t in range(T)
    }

    # zentrale Day-Ahead-Entscheidung (Gruppenentscheidung)
    for t in range(T):
        m += pulp.lpSum(qDA[i][t] for i in range(P)) == qDA_total[t]
        m += qDA_total[t] >= q_DA_min_trade * bDA[t]
        for i in range(P):
            m += qDA[i][t] <= q_max[i] * bDA[t]  # wenn Markt "zu", dann 0

    # Per-Perioden-Bilanzen inkl. PV
    for i in range(P):
        m += pulp.lpSum((qDA[i][t] + qB[i][t] + use_pv[i][t]) for t in range(T)) == own_demand_total[i]
        for t in range(T):
            
            m += use_pv[i][t] + export[i][t]*EC + curtail[i][t]*EC <= pv[i][t]
            
            m += qDA[i][t] + qB[i][t] >= q_min[i]
            m += qDA[i][t] + qB[i][t] <= q_max[i]

    # Zielfunktion
    if mode == "utilitarian":
        m += pulp.lpSum(costs[i] for i in range(P))
        if accept_type == "Aav":
            for i in range(P):
                m += pulp.lpSum(costs_stage[(i, t)] for t in range(T)) <= alpha * sum(v_stagewise[i])
        elif accept_type == "Ap":
            for i in range(P):
                pref = 0.0
                for t in range(T):
                    pref += v_stagewise[i][t]
                    m += pulp.lpSum(costs_stage[(i, tt)] for tt in range(t+1)) <= alpha * pref
        elif accept_type == "As":
            for i in range(P):
                for t in range(T):
                    m += costs_stage[(i, t)] <= alpha * v_stagewise[i][t]

    elif mode == "scaled_minimax":
        # ---------- PHASE 1: min gamma ----------
        gamma = pulp.LpVariable("gamma", lowBound=0)

        # SM-Constraints
        for i in range(P):
            denom = max(sum(v_stagewise[i]), 1e-9)  # Schutz gegen 0
            m += pulp.lpSum(costs_stage[(i, t)] for t in range(T)) <= gamma * denom

        # (Optional) Acceptability-Constraints – nutzen dieselbe Baseline:
        if accept_type == "Aav":
            for i in range(P):
                m += pulp.lpSum(costs_stage[(i, t)] for t in range(T)) <= alpha * sum(v_stagewise[i])
        elif accept_type == "Ap":
            for i in range(P):
                pref = 0.0
                for t in range(T):
                    pref += v_stagewise[i][t]
                    m += pulp.lpSum(costs_stage[(i, tt)] for tt in range(t+1)) <= alpha * pref
        elif accept_type == "As":
            for i in range(P):
                for t in range(T):
                    m += costs_stage[(i, t)] <= alpha * v_stagewise[i][t]

        # Ziel für Phase 1 (wichtig: NICHT 'm += gamma', sondern Objektiv setzen)
        m.setObjective(gamma)

        # Solve Phase 1
        _ = m.solve(pulp.PULP_CBC_CMD(msg=0))
        gamma_star = pulp.value(gamma)

        # ---------- PHASE 2: min sekundäres Ziel bei gamma <= gamma* ----------
        # Kappe gamma auf Optimum von Phase 1
        m += gamma <= gamma_star + 1e-9

        # Sekundärziel: utilitaristisch (Summe der Kosten)
        secondary_obj = pulp.lpSum(costs[i] for i in range(P))
        m.objective = secondary_obj

    

    

    status = m.solve(pulp.PULP_CBC_CMD(msg=0))
    total_cost = pulp.value(m.objective)
    indiv_costs = [pulp.value(costs[i]) for i in range(P)]
    savings_pct = [((v_individual[i] - indiv_costs[i]) / v_individual[i] * 100) if v_individual[i] > 0 else 0 for i in range(P)]
    allocations = {
    i: {
        "qDA": [pulp.value(qDA[i][t]) for t in range(T)],
        "qB": [pulp.value(qB[i][t]) for t in range(T)],
        "use_pv": [pulp.value(use_pv[i][t]) for t in range(T)],
        "export": [pulp.value(export[i][t]) for t in range(T)],
        "curtail": [pulp.value(curtail[i][t]) for t in range(T)],
    }
    for i in range(P)
    }
    return total_cost, indiv_costs, savings_pct, allocations

# Szenarien definieren
scenarios = []
costs_by_agent = []  # Liste von [c1, c2, c3, c4] je Szenario

# Independent (Einzellösungen)
scenarios.append("independent")
costs_by_agent.append(v_individual)  # schon berechnet

# Aggregationsfälle sammeln: (mode, accept_type, label)
cases = [
    ("utilitarian", None,   "U / None"),
    ("utilitarian", "Aav",  "U / Aav"),
    ("utilitarian", "Ap",   "U / Ap"),
    ("utilitarian", "As",   "U / As"),
    ("scaled_minimax", None,  "SM / None"),
    ("scaled_minimax", "Aav", "SM / Aav"),
    ("scaled_minimax", "Ap",  "SM / Ap"),
    ("scaled_minimax", "As",  "SM / As"),
]

for mode, acc, label in cases:
    total, indiv, _, alloc = solve_aggregation(mode, accept_type=acc)
    scenarios.append(label)
    costs_by_agent.append(indiv)
    for i in alloc:
        print(label, i+1, alloc[i], "\n")

N = len(scenarios)
x = np.arange(N)
width = 0.2

# Daten in Arrays
costs = np.array(costs_by_agent) 
colors = ["#4e79a7", "#f28e2c", "#e15759", "#76b7b2"]
agent_labels = ["P1", "P2", "P3", "P4"]

fig, ax = plt.subplots(figsize=(10, 6))

bottom = np.zeros(N)
for j in range(4):  # je Prosumer
    ax.bar(x, costs[:, j], width, bottom=bottom, label=agent_labels[j], color=colors[j])
    bottom += costs[:, j]

# Summen als Text über den Balken
totals = costs.sum(axis=1)
for i, tot in enumerate(totals):
    ax.text(x[i], tot + max(totals)*0.01, f"{tot:.0f}", ha="center", va="bottom", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(scenarios, rotation=35, ha="right")
ax.set_ylabel("Kosten (Geldeinheiten)")
ax.set_title("Gesamtkosten je Modell (gestapelt nach Prosumer)")
ax.legend(ncol=4, loc="upper left", bbox_to_anchor=(0,1.15))

plt.tight_layout()
plt.savefig("fairness_results_stacked.png", dpi=300)

SOLVER = pulp.PULP_CBC_CMD(msg=0, timeLimit=300)

def coalition_cost(S, accept_type=None, alpha=alpha):
    """
    C(S) unter denselben Physik-Constraints wie Aggregation.
    Wenn accept_type in {"Aav","Ap","As"}, dann auch diese Acceptability 
    (nur bezogen auf die Spieler i ∈ S) hinzufügen.
    """
    m = pulp.LpProblem("", pulp.LpMinimize)

    qDA = pulp.LpVariable.dicts("qDA", (range(P), range(T)), lowBound=0)
    qB  = pulp.LpVariable.dicts("qB",  (range(P), range(T)), lowBound=0)
    use_pv = pulp.LpVariable.dicts("use_pv", (range(P), range(T)), lowBound=0)
    export   = pulp.LpVariable.dicts("export",   (range(P), range(T)), lowBound=0)
    curtail  = pulp.LpVariable.dicts("curtail",  (range(P), range(T)), lowBound=0)
    qDA_total = pulp.LpVariable.dicts("qDA_total", range(T), lowBound=0)
    bDA = pulp.LpVariable.dicts("bDA", range(T), cat="Binary")

    # Kosten je i,t:
    costs_stage = {
        (i,t): p_DA[t]*qDA[i][t] + p_B[t]*qB[i][t]
              #- p_feed[t]*export[i][t] - r_curt[t]*curtail[i][t]
        for i in S for t in range(T)
    }
    costs_i = { i: pulp.lpSum(costs_stage[(i,t)] for t in range(T)) for i in S }
    m += pulp.lpSum(costs_i[i] for i in S)   # Zielfunktion: C(S)

    # Physik: per t Bilanz, PV-Split, q_min/q_max, DA-Block wie in Aggregation (schon drin)
    # --- Physik-Constraints für Koalition S ---
    for i in S:
        for t in range(T):
            # Per-Periode Energiebilanz
            m += qDA[i][t] + qB[i][t] + use_pv[i][t] == (own_demand_total[i]/T)
            # PV-Split (EC schaltet Export/Curtail aus)
            m += use_pv[i][t]  <= pv[i][t]
            # Einkaufsgrenzen
            m += qDA[i][t] + qB[i][t] >= q_min[i]
            m += qDA[i][t] + qB[i][t] <= q_max[i]
    
    # Day-Ahead-Block (pro Periode!)
    for t in range(T):
        m += pulp.lpSum(qDA[i][t] for i in S) == qDA_total[t]
        m += qDA_total[t] >= q_DA_min_trade * bDA[t]
        for i in S:
            m += qDA[i][t] <= q_max[i] * bDA[t]
        # --- Optional: Acceptability-Variante für Koalition S ---
        if accept_type is not None:
            # Solo-Baseline nur über Spieler in S:
            v_solo_S = { i: v_stagewise[i] for i in S }   # v_stagewise[i][t] MUSS aus solve_individual konsistent kommen
        if accept_type == "As":
            for i in S:
                for t in range(T):
                    m += costs_stage[(i,t)] <= alpha * v_solo_S[i][t]
        elif accept_type == "Ap":
            for i in S:
                pref = 0.0
                for t in range(T):
                    pref += v_solo_S[i][t]
                    m += pulp.lpSum(costs_stage[(i,tt)] for tt in range(t+1)) <= alpha * pref
        elif accept_type == "Aav":
            for i in S:
                m += pulp.lpSum(costs_stage[(i,t)] for t in range(T)) <= alpha * sum(v_solo_S[i])

    status = m.solve(SOLVER)
    return float(sum(pulp.value(costs_i[i]) for i in S))

def shapley_values(players, coalition_cost_fn, mode="costs", accept_type=None, alpha=alpha):
    """
    mode="costs":  charakteristische Funktion c(S) = C(S)
    mode="savings": s(S) = sum(v_i^solo, i in S) - C(S)
    """
    P = list(players)
    n = len(P)
    # Precompute C(S) für alle S
    C = {}
    for k in range(n+1):
        for S in combinations(P, k):
            C[S] = coalition_cost_fn(S, accept_type=accept_type, alpha=alpha)

    # Solo-Summen für savings-Modus:
    if mode == "savings":
        v_solo_sum = { S: sum(v_individual[i] for i in S) for S in C.keys() }
        v_fun = lambda S: v_solo_sum[S] - C[S]
    else:
        v_fun = lambda S: -C[S]  # Shapley auf "utility" ist üblich; äquivalent: Beiträge auf Kosten mit Minus

    # Shapley
    import math
    phi = {i: 0.0 for i in P}
    for S in C.keys():
        Sset = set(S)
        for i in P:
            if i in Sset: 
                continue
            S_with = tuple(sorted(Sset | {i}))
            S_without = tuple(sorted(S))
            weight = math.factorial(len(S)) * math.factorial(n-len(S)-1) / math.factorial(n)
            phi[i] += weight * (v_fun(S_with) - v_fun(S_without))

    # Aus den φ die Zahlungen bauen:
    if mode == "costs":

        charges = { i: -phi[i] for i in P }
        
        total = sum(charges.values())
        gap = C[tuple(P)] - total
        if abs(gap) > 1e-6:
            # verteil' die Korrektur proportional
            s = sum(abs(charges[i]) for i in P) or 1.0
            for i in P:
                charges[i] += gap * (abs(charges[i]) / s)
        return charges
    else:
        # savings-Shapley → Ersparnisanteile; Zahlung = v_solo - φ^s
        savings = phi
        charges = { i: v_individual[i] - savings[i] for i in P }
        return charges, savings

if __name__ == "__main__":
    # Aggregationskosten der Großkoalition (zur Kontrolle)
    C_N = coalition_cost(tuple(range(P)))
    print(f"[INFO] C(N) (utilitarian): {C_N:.2f}")

    # 1) Kosten-Shapley je Variante
    charges_None = shapley_values(range(P), coalition_cost, mode="costs", accept_type=None)
    charges_Aav  = shapley_values(range(P), coalition_cost, mode="costs", accept_type="Aav", alpha=alpha)
    charges_Ap   = shapley_values(range(P), coalition_cost, mode="costs", accept_type="Ap",  alpha=alpha)
    charges_As   = shapley_values(range(P), coalition_cost, mode="costs", accept_type="As",  alpha=alpha)

    # 2) Matrix für den Stack (Reihenfolge wie im Paper-Chart)
    variants = ["None","Aav","Ap","As"]
    stacks = np.array([
        [charges_None[i] for i in range(P)],
        [charges_Aav[i]  for i in range(P)],
        [charges_Ap[i]   for i in range(P)],
        [charges_As[i]   for i in range(P)],
    ])  # shape (4, P)

    # 3) Plot
    x = np.arange(len(variants))
    width = 0.65
    colors = ["#4e79a7", "#f28e2c", "#e15759", "#76b7b2"]
    agent_labels = ["P1","P2","P3","P4"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(variants))
    for j in range(P):
        ax.bar(x, stacks[:, j], width, bottom=bottom, label=agent_labels[j], color=colors[j])
        bottom += stacks[:, j]

    totals = stacks.sum(axis=1)
    for i, tot in enumerate(totals):
        ax.text(x[i], tot + max(totals)*0.01, f"{tot:.0f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(variants)
    ax.set_ylabel("Kosten-Zahlung (Shapley)")
    ax.set_title("Kosten-Shapley je Acceptability-Variante")
    ax.legend(ncol=4, loc="upper left", bbox_to_anchor=(0,1.15))
    plt.tight_layout()
    plt.savefig("fairness_shapley_costs_by_variant.png", dpi=300)
    print("Plot gespeichert als fairness_shapley_costs_by_variant.png")