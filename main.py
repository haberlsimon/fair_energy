import pulp
import matplotlib.pyplot as plt
import numpy as np
import itertools
from math import factorial

#Anzahl Prosumer
P = 4
#Anzahl Zeitschritte
T = 5

p_DA = [2, 16, 1, 10, 1]   # Day-Ahead Preise wie in fairness by design von Zoe Fornier
p_B  = [6, 25, 5, 15, 5]   # Balancing Preise

q_DA_min_trade = 11
q_min = [0, 5, 0, 2]
q_max = [5, 5, 4, 3]

alpha = 0.98

# Eigenproduktion (über alle T Perioden, z. B. MWh)
own_prod_total   = [ 1,  0, 0,  2]
# Eigenbedarf (über alle T Perioden, z. B. MWh)
own_demand_total = [10, 25,  8, 15]

pv     = [([own_prod_total[i]/T]*T) for i in range(P)]

p_feed = [1, 12, 0, 8, 0]          # Einspeisetarif [€/MWh]
r_curt = [1.1, 13, 0.1, 0.1]*T                      # Vergütung für Verzicht (feed-in + kleiner Bonus)
K_export = [10, 10, 8, 10, 10]  # Export-Kappe je Periode (Beispiel)

# Individuelle Optimierung (Pi)
def solve_individual(i):
    m = pulp.LpProblem(f"Indiv_{i}", pulp.LpMinimize)
    qDA = pulp.LpVariable.dicts("qDA", (range(P), range(T)), lowBound=0)
    qB  = pulp.LpVariable.dicts("qB",  range(T), lowBound=0)
    use_pv = pulp.LpVariable.dicts("use_pv", range(T), lowBound=0)
    export   = pulp.LpVariable.dicts("export",   range(T), lowBound=0)
    curtail  = pulp.LpVariable.dicts("curtail",  (range(T)), lowBound=0)

    # Zielfunktion: nur Einkaufskosten (PV hat hier keine direkten Kosten)
    m += pulp.lpSum(
        p_DA[t]*qDA[i][t] + p_B[t]*qB[t]
        - p_feed[t]*export[t]
        - r_curt[t]*curtail[t]
        for t in range(T)
        )   
    
    m += pulp.lpSum((qB[t] + use_pv[t]) for t in range (T)) == own_demand_total[i]

    for t in range(T):
        
        m += use_pv[t] + export[t] <= pv[i][t]  # PV-Nutzung limitiert durch vorhandene PV

        # Per-Perioden-Grenzen
        m += qB[t] >= q_min[i]
        m += qB[t] <= q_max[i]

        

    # (Gesamtbedarfsgleichung ist implizit erfüllt durch die Summation der Periodenbilanzen)
    status = m.solve(pulp.PULP_CBC_CMD(msg=0))
    v_total = pulp.value(m.objective)
    v_stage = [pulp.value(
        p_B[t]*qB[t]
        - p_feed[t]*export[t]
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
        - p_feed[t]*export[i][t]
        - r_curt[t]*curtail[i][t])
        for t in range(T)
        )
    costs_stage = {
        (i,t): p_DA[t]*qDA[i][t] + p_B[t]*qB[i][t]
              - p_feed[t]*export[i][t]
              - r_curt[t]*curtail[i][t]
        for i in range(P) for t in range(T)
    }

    # Zielfunktion
    if mode == "utilitarian":
        m += pulp.lpSum(costs[i] for i in range(P))
    elif mode == "scaled_minimax":
        z = pulp.LpVariable("z", lowBound=0)
        EPS = 1e-9
        for i in range(P):
            denom = max(v_individual[i], EPS)
            m += costs[i] / denom <= z
        m += z

    # Acceptability
    if accept_type == "As":
        for i in range(P):
            for t in range(T):
                m += costs_stage[(i,t)] <= alpha * v_stagewise[i][t]
    elif accept_type == "Ap":
        for i in range(P):
            for t in range(T):
                m += pulp.lpSum(costs_stage[(i,tau)] for tau in range(t+1)) <= alpha * pulp.lpSum(v_stagewise[i][:t+1])
    elif accept_type == "Aav":
        for i in range(P):
            m += pulp.lpSum(costs_stage[(i,t)] for t in range(T)) <= alpha * pulp.lpSum(v_stagewise[i])
    
    # Per-Perioden-Bilanzen inkl. PV
    for i in range(P):
        m += pulp.lpSum((qDA[i][t] + qB[i][t] + use_pv[i][t]) for t in range(T)) == own_demand_total[i]
        for t in range(T):
            
            m += use_pv[i][t] + export[i][t] + curtail[i][t] <= pv[i][t]
            
            m += qDA[i][t] + qB[i][t] >= q_min[i]
            m += qDA[i][t] + qB[i][t] <= q_max[i]

    # zentrale Day-Ahead-Entscheidung (Gruppenentscheidung)
    for t in range(T):
        m += pulp.lpSum(qDA[i][t] for i in range(P)) == qDA_total[t]
        m += qDA_total[t] >= q_DA_min_trade * bDA[t]
        for i in range(P):
            m += qDA[i][t] <= q_max[i] * bDA[t]  # wenn Markt "zu", dann 0
    
    #print(m.constraints)
    

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

# 1) Szenarien definieren (ein Koordinatensystem, Balken nebeneinander)
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

# 2) Gestapelte Balken zeichnen
N = len(scenarios)
x = np.arange(N)
width = 0.2

# Daten in Arrays
costs = np.array(costs_by_agent)  # shape (N, 4)
colors = ["#4e79a7", "#f28e2c", "#e15759", "#76b7b2"]  # P1..P4
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

SOLVER = pulp.PULP_CBC_CMD(msg=0, timeLimit=30)

def coalition_cost(S):
    """Aggregationsgesamtkosten C(S) für Koalition S (utilitarian, ohne Acceptability)."""
    S = tuple(sorted(S))
    if len(S) == 0:
        return 0.0
    if len(S) == 1:
        i = S[0]
        return float(v_individual[i])  # 1er Koalition = Solo

    m = pulp.LpProblem(f"Coalition_util_{'_'.join(map(str,S))}", pulp.LpMinimize)
    qDA = pulp.LpVariable.dicts("qDA", (S, range(T)), lowBound=0)
    qB  = pulp.LpVariable.dicts("qB",  (S, range(T)), lowBound=0)
    qDA_total = pulp.LpVariable.dicts("qDA_total", range(T), lowBound=0)
    bDA = pulp.LpVariable.dicts("bDA", range(T), cat="Binary")
    use_pv = pulp.LpVariable.dicts("use_pv", (S, range(T)), lowBound=0)
    export  = pulp.LpVariable.dicts("export",  (S, range(T)), lowBound=0)
    curtail = pulp.LpVariable.dicts("curtail", (S, range(T)), lowBound=0)

    costs_S = {
        i: pulp.lpSum( p_DA[t]*qDA[i][t] + p_B[t]*qB[i][t]
            - p_feed[t]*export[i][t]
            - r_curt[t]*curtail[i][t]
            for t in range(T)) for i in S
    }
    m += pulp.lpSum(costs_S[i] for i in S)

    for i in S:
        
        m += pulp.lpSum((qDA[i][t] + qB[i][t] + use_pv[i][t]) for t in range(T)) == own_demand_total[i]

        for t in range(T):
            
            
            m += use_pv[i][t] + export[i][t] + curtail[i][t] <= pv[i][t]

            m += qDA[i][t] + qB[i][t] >= q_min[i]
            m += qDA[i][t] + qB[i][t] <= q_max[i]

    for t in range(T):
        m += pulp.lpSum(qDA[i][t] for i in S) == qDA_total[t]
        m += qDA_total[t] >= q_DA_min_trade * bDA[t]
        for i in S:
            m += qDA[i][t] <= q_max[i] * bDA[t]

    status = m.solve(SOLVER)
    if pulp.LpStatus[status] != "Optimal":
        print(f"[WARN] Koalition {S}: Status={pulp.LpStatus[status]}")
        return float("inf")  # klar signalisiert: unlösbar oder abgebrochen

    return float(sum(pulp.value(costs_S[i]) for i in S))

def shapley_savings(I_count):
    players = list(range(I_count))
    # Vorberechnungen
    solo_sum = {tuple(sorted(S)): sum(v_individual[i] for i in S)
                for r in range(I_count+1) for S in itertools.combinations(players, r)}
    coal_cost = {}
    # Progress
    all_S = [S for r in range(I_count+1) for S in itertools.combinations(players, r)]
    print(f"[INFO] Berechne C(S) für {len(all_S)} Koalitionen...")
    for S in all_S:
        coal_cost[tuple(sorted(S))] = coalition_cost(S)
    value = {S: solo_sum[S] - coal_cost[S] for S in coal_cost}  # Ersparnis v(S)

    # Shapley
    phi = [0.0]*I_count
    n = I_count
    for i in players:
        shap_i = 0.0
        for r in range(n):
            for S in itertools.combinations([p for p in players if p != i], r):
                S_key = tuple(sorted(S))
                S_i_key = tuple(sorted(S + (i,)))
                weight = factorial(r) * factorial(n - r - 1) / factorial(n)
                marginal = value[S_i_key] - value[S_key]
                shap_i += weight * marginal
        phi[i] = float(shap_i)
    return phi, value, coal_cost

if __name__ == "__main__":
    # Aggregationskosten der Großkoalition (zur Kontrolle)
    C_N = coalition_cost(tuple(range(P)))
    print(f"[INFO] C(N) (utilitarian): {C_N:.2f}")

    # Shapley berechnen
    phi, value_by_S, coal_cost_by_S = shapley_savings(P)

    print("\n=== Shapley-Ergebnisse (Ersparnisanteile) ===")
    for i, s in enumerate(phi, start=1):
        print(f"P{i}: φ_i = {s:.2f}")

    assigned_costs = [float(v_individual[i] - phi[i]) for i in range(P)]
    print("\nZugewiesene Kosten nach Shapley:", [round(c,2) for c in assigned_costs])
    print("Kontrolle: Sum assigned =", round(sum(assigned_costs),2), " | C(N) =", round(C_N,2))

    # Vergleichsplot (gestapelt)
    scenarios = ["independent", "utilitarian (C(N))", "Shapley assigned"]
    util_indiv = solve_aggregation("utilitarian", accept_type=None)[1]  # Kosten je Prosumer ohne Verteilung
    stacked = np.array([v_individual, util_indiv, assigned_costs])

    x = np.arange(len(scenarios))
    width = 0.65
    colors = ["#4e79a7", "#f28e2c", "#e15759", "#76b7b2"]
    agent_labels = ["P1","P2","P3","P4"]

    fig, ax = plt.subplots(figsize=(12,6))
    bottom = np.zeros(len(scenarios))
    for j in range(P):
        ax.bar(x, stacked[:, j], width, bottom=bottom, label=agent_labels[j], color=colors[j])
        bottom += stacked[:, j]
    totals = stacked.sum(axis=1)
    for i, tot in enumerate(totals):
        ax.text(x[i], tot + max(totals)*0.01, f"{tot:.0f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel("Kosten")
    ax.set_title("Kostenvergleich: Solo vs. reine Aggregation vs. Shapley-Verteilung")
    ax.legend(ncol=4, loc="upper left", bbox_to_anchor=(0,1.15))
    plt.tight_layout()
    plt.savefig("fairness_shapley_stacked.png", dpi=300)
    print("Plot gespeichert als fairness_shapley_stacked.png")