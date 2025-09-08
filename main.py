import pulp
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
import math

#Anzahl Prosumer
P = 4
#Anzahl Zeitschritte
T = 5

# Zahlungslogiken
EC                 = 0   # Export and Curtail ist möglich, 0 bedeutet deaktiviert
FEED_IN_PAYMENTS   = 0   # 1 = Export wird vergütet (Erlös), 0 = kein Erlös
CURTAIL_PAYMENTS   = 0   # 1 = Curtail wird vergütet, 0 = keine Vergütung
INTERNAL_SHARING   = 0   # 1 = interne PV-Verteilung (send/recv) aktiv; 0 = aus

# Interner Transferpreis π_t (€/MWh) – nur Verrechnung innerhalb des Systems
PI = [2, 16, 1, 10, 1]

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
p_feed = [1.5, 12, 0.75, 7.5, 0.75]            # Einspeisetarif [€/MWh]
r_curt = [1, 8, 0.5, 5, 0.5]              # Vergütung für Verzicht (feed-in + kleiner Bonus)
K_export = [10]*T

# Individuelle Optimierung (Pi)
def solve_individual(i):
    m = pulp.LpProblem(f"Indiv_{i}", pulp.LpMinimize)
    qB  = pulp.LpVariable.dicts("qB",  range(T), lowBound=0.0)
    use_pv = pulp.LpVariable.dicts("use_pv", range(T), lowBound=0.0)
    export   = pulp.LpVariable.dicts("export",   range(T), lowBound=0.0)
    curtail  = pulp.LpVariable.dicts("curtail",  (range(T)), lowBound=0.0)

    # Zielfunktion
    expr = pulp.lpSum(p_B[t]*qB[t] for t in range(T))

    # Export-Vergütung
    if FEED_IN_PAYMENTS and EC == 1:
            expr += pulp.lpSum((-p_feed[t]) * export[t] for t in range(T))
    # Curtail-Vergütung
    if CURTAIL_PAYMENTS and EC == 1:
        expr += pulp.lpSum((-r_curt[t]) * curtail[t] for t in range(T))
    m += expr
    
    m += pulp.lpSum((qB[t] + use_pv[t]) for t in range (T)) == own_demand_total[i]

    for t in range(T):
        
        m += use_pv[t] + export[t]*EC + curtail[t]*EC == pv[i][t]  # PV-Nutzung limitiert durch vorhandene PV

        # Per-Perioden-Grenzen
        m += qB[t]  >= q_min[i]
        m += qB[t]  <= q_max[i]

    status = m.solve(pulp.PULP_CBC_CMD(msg=0))

    v_total = pulp.value(m.objective)
    v_stage = []
    for t in range(T):
        val = p_B[t]*pulp.value(qB[t])

        if FEED_IN_PAYMENTS and EC == 1:
            val += -p_feed[t] * pulp.value(export[t])
        if CURTAIL_PAYMENTS and EC == 1:
            val += -r_curt[t] * pulp.value(curtail[t])

        v_stage.append(val)
    allocations = {
        "qB": [round(pulp.value(qB[t]), 2) for t in range(T)],
        "use_pv": [round(pulp.value(use_pv[t]), 2) for t in range(T)],
        "export": [round(pulp.value(export[t]), 2) for t in range(T)] if EC else 0.0,
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

    qDA = pulp.LpVariable.dicts("qDA", (range(P), range(T)), lowBound=0.0)
    qB  = pulp.LpVariable.dicts("qB",  (range(P), range(T)), lowBound=0.0)
    use_pv = pulp.LpVariable.dicts("use_pv", (range(P), range(T)), lowBound=0.0)
    export   = pulp.LpVariable.dicts("export",   (range(P), range(T)), lowBound=0.0)
    curtail  = pulp.LpVariable.dicts("curtail",  (range(P), range(T)), lowBound=0.0)
    qDA_total = pulp.LpVariable.dicts("qDA_total", range(T), lowBound=0.0)
    bDA = pulp.LpVariable.dicts("bDA", range(T), cat="Binary")
    if INTERNAL_SHARING:
        send = pulp.LpVariable.dicts("send", (range(P), range(T)), lowBound=0.0)
        recv = pulp.LpVariable.dicts("recv", (range(P), range(T)), lowBound=0.0)

    # Kosten pro Prosumer (nur Einkauf)
    costs = {}
    for i in range(P):
        # Externer Einkauf
        expr = pulp.lpSum(p_DA[t]*qDA[i][t] + p_B[t]*qB[i][t] for t in range(T))

        if FEED_IN_PAYMENTS and EC == 1:
            expr += pulp.lpSum((-p_feed[t]) * export[i][t] for t in range(T))  # Erlös mindert Kosten

        if CURTAIL_PAYMENTS and EC == 1:
            expr += pulp.lpSum((-r_curt[t]) * curtail[i][t] for t in range(T))  # Vergütung mindert Kosten

        if INTERNAL_SHARING:
            expr += pulp.lpSum( PI[t] * (recv[i][t] - send[i][t]) for t in range(T) )

        costs[i] = expr

    costs_stage = {
        (i,t):
            (p_DA[t]*qDA[i][t] + p_B[t]*qB[i][t])
            + ( (-p_feed[t]) * export[i][t] if (FEED_IN_PAYMENTS and EC==1) else 0 )
            + ( (-r_curt[t]) * curtail[i][t] if (CURTAIL_PAYMENTS and EC==1) else 0 )
            + ( PI[t]*(recv[i][t] - send[i][t]) if INTERNAL_SHARING else 0 )
        for i in range(P) for t in range(T)
    }

    # zentrale Day-Ahead-Entscheidung (Gruppenentscheidung)
    for t in range(T):
        m += pulp.lpSum(qDA[i][t] for i in range(P)) == qDA_total[t]
        m += qDA_total[t] >= q_DA_min_trade * bDA[t]
        for i in range(P):
            m += qDA[i][t] <= q_max[i] * bDA[t]  # wenn Markt "zu", dann 0

    
    for t in range(T):
        # Kappe gegen Netzlimit (z. B. Trafo-/Leitungskapazität)
        m += pulp.lpSum(export[i][t] for i in range(P)) <= K_export[t]

    # Per-Perioden-Bilanzen inkl. PV
    for i in range(P):

        if INTERNAL_SHARING:
            m += pulp.lpSum(qDA[i][t] + qB[i][t] + use_pv[i][t] + recv[i][t] for t in range(T)) == own_demand_total[i]
        else:
            m += pulp.lpSum(qDA[i][t] + qB[i][t] + use_pv[i][t]               for t in range(T)) == own_demand_total[i]

        for t in range(T):
            
            if INTERNAL_SHARING:
                if EC == 0:
                    m += use_pv[i][t] + send[i][t] == pv[i][t]
                    m += export[i][t] == 0
                    m += curtail[i][t] == 0
                else:
                    m += use_pv[i][t] + export[i][t] + curtail[i][t] + send[i][t] == pv[i][t]
            else:
                if EC == 0:
                    m += use_pv[i][t] == pv[i][t]
                    m += export[i][t] == 0
                    m += curtail[i][t] == 0
                else:
                    m += use_pv[i][t] + export[i][t] + curtail[i][t] == pv[i][t]
            
            m += qDA[i][t] + qB[i][t] >= q_min[i]
            m += qDA[i][t] + qB[i][t] <= q_max[i]
    
    if INTERNAL_SHARING:
        for t in range(T):
            m += pulp.lpSum(send[i][t] for i in range(P)) == pulp.lpSum(recv[i][t] for i in range(P))

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
        # PHASE 1: min gamma
        gamma = pulp.LpVariable("gamma", lowBound=0)

        # SM-Constraints
        for i in range(P):
            denom = max(sum(v_stagewise[i]), 1e-9)  # Schutz gegen 0
            m += pulp.lpSum(costs_stage[(i, t)] for t in range(T)) <= gamma * denom

        # Acceptability-Constraints
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

        # Ziel für Phase 1
        m.setObjective(gamma)

        # Solve Phase 1
        _ = m.solve(pulp.PULP_CBC_CMD(msg=0))
        gamma_star = pulp.value(gamma)

        # PHASE 2: min sekundäres Ziel bei gamma <= gamma*
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
        "qDA": [round(pulp.value(qDA[i][t]), 2) for t in range(T)],
        "qB": [round(pulp.value(qB[i][t]), 2) for t in range(T)],
        "use_pv": [round(pulp.value(use_pv[i][t]), 2) for t in range(T)],
        "export": [round(pulp.value(export[i][t]), 2) for t in range(T)] if EC else None,
        "curtail": [round(pulp.value(curtail[i][t]), 2) for t in range(T)] if EC else None,
        "send":   [round(pulp.value(send[i][t]), 2) for t in range(T)] if INTERNAL_SHARING else None,
        "recv":   [round(pulp.value(recv[i][t]), 2) for t in range(T)] if INTERNAL_SHARING else None,
    }
    for i in range(P)
    }
    return total_cost, indiv_costs, savings_pct, allocations

# Szenarien definieren
scenarios = []
costs_by_agent = []  # Liste von [c1, c2, c3, c4] je Szenario

# Independent
scenarios.append("independent")
costs_by_agent.append(v_individual)

# Aggregationsfälle
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
    total, indiv, _, alloc = solve_aggregation(mode, acc)
    scenarios.append(label)
    costs_by_agent.append(indiv)
    for i in alloc:
        print(label, i+1, alloc[i], "\n")

# Plot
N = len(scenarios)
x = np.arange(N)
bar_width = 0.25

costs = np.array(costs_by_agent)
colors = ["#4e79a7", "#f28e2c", "#e15759", "#76b7b2"]
agent_labels = [f"P{i+1}" for i in range(costs.shape[1])]

fig, ax = plt.subplots(figsize=(12, 7))

# Positive/Negative Anteile getrennt stapeln
pos = np.where(costs > 0, costs, 0.0)
neg = np.where(costs < 0, costs, 0.0)

pos_bottom = np.zeros(N)
neg_bottom = np.zeros(N)

for j in range(costs.shape[1]):
    # Positive Balken
    ax.bar(x, pos[:, j], bar_width, bottom=pos_bottom, color=colors[j],
           label=agent_labels[j] if j == 0 else None)
    pos_bottom += pos[:, j]

    # Negative Balken
    ax.bar(x, neg[:, j], bar_width, bottom=neg_bottom, color=colors[j])
    neg_bottom += neg[:, j]

# Totals (Summen über alle Prosumer)
totals = costs.sum(axis=1)

# Ober- und Unterkante der Balken
top_edge = pos_bottom
bot_edge = neg_bottom

ymax = float(np.max(top_edge)) if N else 0.0
ymin = float(np.min(bot_edge)) if N else 0.0
yrange = max(1.0, ymax - ymin)
offset = 0.02 * yrange

# Summenbeschriftung außerhalb der Balken
for i, tot in enumerate(totals):
    if tot >= 0:
        y = top_edge[i] + offset
        va = "bottom"
    else:
        y = bot_edge[i] - offset
        va = "top"
    ax.text(x[i], y, f"{tot:.1f}", ha="center", va=va, fontsize=9)

# Prosumer-Kosten-Box rechts neben jedem Balken
x_side_offset = 0.35
for i in range(N):
    lines = [f"{agent_labels[p]}: {costs[i, p]:.1f}" for p in range(costs.shape[1])]
    text = "\n".join(lines)
    y_mid = 0.5 * (top_edge[i] + bot_edge[i])
    ax.text(x[i] + x_side_offset, y_mid, text,
            ha="left", va="center", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85))

# Null-Linie für Orientierung
ax.axhline(0, linewidth=1, color="black")

# Achsen und Titel
ax.set_xticks(x)
ax.set_xticklabels(scenarios, rotation=35, ha="right")
ax.set_ylabel("Kosten (Geldeinheiten)")
ax.set_title("Gesamtkosten je Modell (gestapelt) – mit Netto-Produzenten (negative Kosten)")

# Legende
handles = [plt.Rectangle((0,0),1,1,color=colors[j]) for j in range(costs.shape[1])]
ax.legend(handles, agent_labels, ncol=4, loc="upper left", bbox_to_anchor=(0,1.15))

ax.set_xlim(-0.5, N - 0.5 + x_side_offset + 0.8)
ax.set_ylim(ymin - 0.08*yrange, ymax + 0.10*yrange)

plt.tight_layout()
plt.savefig("fairness_results_stacked_signed.png", dpi=300)

SOLVER = pulp.PULP_CBC_CMD(msg=0, timeLimit=300)

def coalition_cost(S, accept_type=None, alpha=alpha):
    """
    C(S): Minimale Koalitionskosten der Spieler in S unter denselben Physik-
    und Preis-Constraints wie im Aggregationsmodell.
    accept_type in {None, "Aav", "Ap", "As"} wird analog angewandt (nur auf i ∈ S).
    """
    m = pulp.LpProblem("coalition_cost", pulp.LpMinimize)

    # Variablen
    qDA = pulp.LpVariable.dicts("qDA", (range(P), range(T)), lowBound=0.0)
    qB  = pulp.LpVariable.dicts("qB",  (range(P), range(T)), lowBound=0.0)
    use_pv  = pulp.LpVariable.dicts("use_pv",  (range(P), range(T)), lowBound=0.0)
    export  = pulp.LpVariable.dicts("export",  (range(P), range(T)), lowBound=0.0)
    curtail = pulp.LpVariable.dicts("curtail", (range(P), range(T)), lowBound=0.0)
    qDA_total    = pulp.LpVariable.dicts("qDA_total", range(T), lowBound=0.0)
    export_total = pulp.LpVariable.dicts("export_total", range(T), lowBound=0.0)
    bDA = pulp.LpVariable.dicts("bDA", range(T), cat="Binary")

    if INTERNAL_SHARING:
        send = pulp.LpVariable.dicts("send", (range(P), range(T)), lowBound=0.0)
        recv = pulp.LpVariable.dicts("recv", (range(P), range(T)), lowBound=0.0)

    costs_stage = {}
    for i in S:
        for t in range(T):
            expr = p_DA[t]*qDA[i][t] + p_B[t]*qB[i][t]
            if FEED_IN_PAYMENTS and EC == 1:
                expr += (-p_feed[t]) * export[i][t]
            if CURTAIL_PAYMENTS and EC == 1:
                expr += (-r_curt[t]) * curtail[i][t]
            if INTERNAL_SHARING:
                expr += PI[t] * (recv[i][t] - send[i][t])
            costs_stage[(i,t)] = expr

    # Zielfunktion: Summe der Koalitionskosten
    m += pulp.lpSum(costs_stage[(i,t)] for i in S for t in range(T))

    # Physik-Constraints
    for i in S:
        # Pro t: PV-Accounting + Einkaufsgrenzen
        for t in range(T):
            if INTERNAL_SHARING:
                if EC == 0:
                    m += use_pv[i][t] + send[i][t] == pv[i][t]
                    m += export[i][t] == 0
                    m += curtail[i][t] == 0
                else:
                    m += use_pv[i][t] + export[i][t] + curtail[i][t] + send[i][t] == pv[i][t]
            else:
                if EC == 0:
                    m += use_pv[i][t] == pv[i][t]
                    m += export[i][t] == 0
                    m += curtail[i][t] == 0
                else:
                    m += use_pv[i][t] + export[i][t] + curtail[i][t] == pv[i][t]

            m += qDA[i][t] + qB[i][t] >= q_min[i]
            m += qDA[i][t] + qB[i][t] <= q_max[i]

        # Summenbilanz über den Horizont
        if INTERNAL_SHARING:
            m += pulp.lpSum(qDA[i][t] + qB[i][t] + use_pv[i][t] + recv[i][t] for t in range(T)) == own_demand_total[i]
        else:
            m += pulp.lpSum(qDA[i][t] + qB[i][t] + use_pv[i][t]               for t in range(T)) == own_demand_total[i]

    # Day-Ahead-Block (Aggregation in S)
    for t in range(T):
        m += pulp.lpSum(qDA[i][t] for i in S) == qDA_total[t]
        m += qDA_total[t] >= q_DA_min_trade * bDA[t]
        for i in S:
            m += qDA[i][t] <= q_max[i] * bDA[t]

    # Export-Kappe
    if EC == 1:
        for t in range(T):
            m += pulp.lpSum(export[i][t] for i in S) == export_total[t]
            m += export_total[t] <= K_export[t]

    # interne Transfers budgetneutral (nur innerhalb S)
    if INTERNAL_SHARING:
        for t in range(T):
            m += pulp.lpSum(send[i][t] for i in S) == pulp.lpSum(recv[i][t] for i in S)

    # Acceptability (nur auf i ∈ S anwenden)
    if accept_type == "As":
        for i in S:
            for t in range(T):
                m += costs_stage[(i,t)] <= alpha * v_stagewise[i][t]
    elif accept_type == "Ap":
        for i in S:
            pref = 0.0
            for tt in range(T):
                pref += v_stagewise[i][tt]
                m += pulp.lpSum(costs_stage[(i,tt2)] for tt2 in range(tt+1)) <= alpha * pref
    elif accept_type == "Aav":
        for i in S:
            m += pulp.lpSum(costs_stage[(i,t)] for t in range(T)) <= alpha * sum(v_stagewise[i])

    # Solve
    _ = m.solve(SOLVER)

    # Rückgabe: C(S)
    return float(sum(pulp.value(costs_stage[(i,t)]) for i in S for t in range(T)))

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
        v_fun = lambda S: -C[S]

    # Shapley
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

    # Kosten-Shapley je Variante
    charges_None, savings_None = shapley_values(range(P), coalition_cost, mode="savings", accept_type=None)
    charges_Aav, savings_Aav  = shapley_values(range(P), coalition_cost, mode="savings", accept_type="Aav", alpha=alpha)
    charges_Ap, savings_Ap   = shapley_values(range(P), coalition_cost, mode="savings", accept_type="Ap",  alpha=alpha)
    charges_As, savings_As   = shapley_values(range(P), coalition_cost, mode="savings", accept_type="As",  alpha=alpha)

    # Matrix für den Stack
    variants = ["None","Aav","Ap","As"]
    stacks = np.array([
        [charges_None[i] for i in range(P)],
        [charges_Aav[i]  for i in range(P)],
        [charges_Ap[i]   for i in range(P)],
        [charges_As[i]   for i in range(P)],
    ])
    stacks_savings = np.array([
        [savings_None[i] for i in range(P)],
        [savings_Aav[i]  for i in range(P)],
        [savings_Ap[i]   for i in range(P)],
        [savings_As[i]   for i in range(P)],
    ]) 
    for s in stacks_savings:
        values = [float(s[i]) for i in range(P)]
        print("Shapley φ (Einsparungen):", [round(values[i], 1) for i in range(P)])
        print("Zahlungen v_i - φ_i:", [round(v_individual[i] - values[i], 1) for i in range(P)])


    # Plot
    #x = np.arange(len(variants))
    #width = 0.65
    #colors = ["#4e79a7", "#f28e2c", "#e15759", "#76b7b2"]
    #agent_labels = ["P1","P2","P3","P4"]
#
    #fig, ax = plt.subplots(figsize=(10, 6))
    #bottom = np.zeros(len(variants))
    #for j in range(P):
    #    ax.bar(x, stacks[:, j], width, bottom=bottom, label=agent_labels[j], color=colors[j])
    #    bottom += stacks[:, j]
#
    #totals = stacks.sum(axis=1)
    #for i, tot in enumerate(totals):
    #    ax.text(x[i], tot + max(totals)*0.01, f"{tot:.0f}", ha="center", va="bottom", fontsize=9)
#
    #ax.set_xticks(x)
    #ax.set_xticklabels(variants)
    #ax.set_ylabel("Kosten-Zahlung (Shapley)")
    #ax.set_title("Kosten-Shapley je Acceptability-Variante")
    #ax.legend(ncol=4, loc="upper left", bbox_to_anchor=(0,1.15))
    #plt.tight_layout()
    #plt.savefig("fairness_shapley_costs_by_variant.png", dpi=300)