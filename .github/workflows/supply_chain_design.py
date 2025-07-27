import pulp

PLANTS = ['P1', 'P2', 'P3']
DCS = ['D1', 'D2', 'D3', 'D4']
CUSTOMERS = [f'C{i}' for i in range(1, 11)]

fixed_costs_plant = {'P1': 650000, 'P2': 800000, 'P3': 720000}
fixed_costs_dc = {'D1': 200000, 'D2': 250000, 'D3': 220000, 'D4': 180000}
transport_cost_plant_dc = {
    ('P1', 'D1'): 2.5, ('P1', 'D2'): 3.1, ('P1', 'D3'): 4.0, ('P1', 'D4'): 5.0,
    ('P2', 'D1'): 3.5, ('P2', 'D2'): 2.8, ('P2', 'D3'): 3.6, ('P2', 'D4'): 4.2,
    ('P3', 'D1'): 4.2, ('P3', 'D2'): 3.9, ('P3', 'D3'): 2.9, ('P3', 'D4'): 3.3,
}
transport_cost_dc_customer = { (d, c): 1.5 + (i*0.1) + (j*0.2) for i, d in enumerate(DCS) for j, c in enumerate(CUSTOMERS) }

emissions_production = {'P1': 1.8, 'P2': 1.2, 'P3': 1.5} # kg CO2 per unit
emissions_transport_plant_dc = { k: v * 0.1 for k, v in transport_cost_plant_dc.items() } # Proportional to cost
emissions_transport_dc_customer = { k: v * 0.1 for k, v in transport_cost_dc_customer.items() }

jobs_plant = {'P1': 100, 'P2': 150, 'P3': 120}
jobs_dc = {'D1': 40, 'D2': 55, 'D3': 50, 'D4': 35}

capacity_plant = {'P1': 50000, 'P2': 70000, 'P3': 60000}
capacity_dc = {'D1': 80000, 'D2': 90000, 'D3': 85000, 'D4': 75000}
demand_customer = {c: 8000 + (i*500) for i, c in enumerate(CUSTOMERS)}

def solve_supply_chain(method='cost_optimal', params=None):
    
    model = pulp.LpProblem("Sustainable_Supply_Chain_Design", pulp.LpMinimize)

    open_plant = pulp.LpVariable.dicts("OpenPlant", PLANTS, cat='Binary')
    open_dc = pulp.LpVariable.dicts("OpenDC", DCS, cat='Binary')
    flow_plant_dc = pulp.LpVariable.dicts("Flow_P_D", (PLANTS, DCS), lowBound=0)
    flow_dc_cust = pulp.LpVariable.dicts("Flow_D_C", (DCS, CUSTOMERS), lowBound=0)

    cost_obj = (pulp.lpSum(fixed_costs_plant[p] * open_plant[p] for p in PLANTS) +
                pulp.lpSum(fixed_costs_dc[d] * open_dc[d] for d in DCS) +
                pulp.lpSum(transport_cost_plant_dc[p, d] * flow_plant_dc[p][d] for p in PLANTS for d in DCS) +
                pulp.lpSum(transport_cost_dc_customer[d, c] * flow_dc_cust[d][c] for d in DCS for c in CUSTOMERS))

    emissions_obj = (pulp.lpSum(emissions_production[p] * flow_plant_dc[p][d] for p in PLANTS for d in DCS) +
                     pulp.lpSum(emissions_transport_plant_dc[p, d] * flow_plant_dc[p][d] for p in PLANTS for d in DCS) +
                     pulp.lpSum(emissions_transport_dc_customer[d, c] * flow_dc_cust[d][c] for d in DCS for c in CUSTOMERS))

    jobs_obj = (pulp.lpSum(jobs_plant[p] * open_plant[p] for p in PLANTS) +
                pulp.lpSum(jobs_dc[d] * open_dc[d] for d in DCS))

    for c in CUSTOMERS:
        model += pulp.lpSum(flow_dc_cust[d][c] for d in DCS) == demand_customer[c], f"Demand_{c}"

    for d in DCS:
        model += pulp.lpSum(flow_plant_dc[p][d] for p in PLANTS) == pulp.lpSum(flow_dc_cust[d][c] for c in CUSTOMERS), f"Flow_Conservation_{d}"

    for p in PLANTS:
        model += pulp.lpSum(flow_plant_dc[p][d] for d in DCS) <= capacity_plant[p] * open_plant[p], f"Capacity_{p}"

    for d in DCS:
        model += pulp.lpSum(flow_dc_cust[d][c] for c in CUSTOMERS) <= capacity_dc[d] * open_dc[d], f"Capacity_{d}"

    print(f"\n--- Solving with method: {method} ---")
    
    if method == 'cost_optimal':
        model.setObjective(cost_obj)
    
    elif method == 'emission_optimal':
        model.setObjective(emissions_obj)
        
    elif method == 'jobs_optimal':
        model.setObjective(jobs_obj)
        model.sense = pulp.LpMaximize

    elif method == 'weighted_sum':
        if not params: params = {'w1': 0.5, 'w2': 0.3, 'w3': 0.2}
        model.setObjective(params['w1'] * cost_obj + params['w2'] * emissions_obj - params['w3'] * jobs_obj)

    elif method == 'epsilon_constrained':
        if not params: params = {'epsilon_emissions': 75000, 'epsilon_jobs': 150}
        model.setObjective(cost_obj)
        model += emissions_obj <= params['epsilon_emissions'], "Epsilon_Emissions_Constraint"
        model += jobs_obj >= params['epsilon_jobs'], "Epsilon_Jobs_Constraint"

    elif method == 'lexicographic':
        model.setObjective(cost_obj)
        model.solve()
        z1_star = pulp.value(model.objective)
        print(f"Lexicographic Step 1: Optimal Cost = {z1_star:,.2f}")
        model += cost_obj <= z1_star # Add as constraint
        
        model.setObjective(emissions_obj)
        model.solve()
        z2_star = pulp.value(model.objective)
        print(f"Lexicographic Step 2: Optimal Emissions = {z2_star:,.2f}")
        model += emissions_obj <= z2_star # Add as constraint

        model.setObjective(jobs_obj)
        model.sense = pulp.LpMaximize
        
    elif method == 'goal_programming':
        if not params: params = {'goal_cost': 2000000, 'goal_emissions': 80000, 'goal_jobs': 180, 'p1': 1, 'p2': 1, 'p3': 1}
        dev_cost_plus = pulp.LpVariable("dev_cost_plus", lowBound=0)
        dev_emis_plus = pulp.LpVariable("dev_emis_plus", lowBound=0)
        dev_jobs_minus = pulp.LpVariable("dev_jobs_minus", lowBound=0)
        
        model += cost_obj - dev_cost_plus <= params['goal_cost']
        model += emissions_obj - dev_emis_plus <= params['goal_emissions']
        model += jobs_obj + dev_jobs_minus >= params['goal_jobs']
        
        model.setObjective(params['p1']*dev_cost_plus + params['p2']*dev_emis_plus + params['p3']*dev_jobs_minus)

    model.solve()
    
    print(f"Status: {pulp.LpStatus[model.status]}")
    print("\n--- Results ---")
    print(f"Total Cost (Z1)      = €{pulp.value(cost_obj):,.2f}")
    print(f"Total Emissions (Z2) = {pulp.value(emissions_obj):,.2f} kg CO2")
    print(f"Total Jobs (Z3)      = {pulp.value(jobs_obj):.0f}")
    
    print("\nNetwork Configuration:")
    for p in PLANTS:
        if open_plant[p].varValue > 0.5:
            print(f"- Plant {p} is OPEN")
    for d in DCS:
        if open_dc[d].varValue > 0.5:
            print(f"- DC {d} is OPEN")
    print("-" * 20)


if __name__ == "__main__":
    
    solve_supply_chain(method='cost_optimal')
    solve_supply_chain(method='emission_optimal')
    solve_supply_chain(method='jobs_optimal')
    
    solve_supply_chain(method='weighted_sum', params={'w1': 0.5, 'w2': 0.3, 'w3': 0.2})
    solve_supply_chain(method='epsilon_constrained', params={'epsilon_emissions': 75000, 'epsilon_jobs': 150})
    solve_supply_chain(method='lexicographic')
    solve_supply_chain(method='goal_programming', params={'goal_cost': 2000000, 'goal_emissions': 80000, 'goal_jobs': 180, 'p1': 1, 'p2': 1, 'p3': 1})
    n=input()
