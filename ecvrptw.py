import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from gurobipy import *
data = {
    'Node': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'X': [82, 96, 50, 48, 13, 29, 63, 84, 14, 2],
    'Y': [76, 44, 5, 20, 7, 89, 41, 39, 24, 39],
    'Demand': [0, 6, 9, 6, 8, 7, 6, 8, 0, 0],
    # 'Demand': [0, 19, 21, 6, 19, 7, 12, 16, 0, 0],
    'time': [0, 10, 10, 10, 10, 10, 10, 10, 20, 20],
    'start_time': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'end_time': [50,50, 50, 50, 50, 50, 50, 50, 50, 50],
    'Type': ['Depot', 'Customer', 'Customer', 'Customer', 'Customer', 'Customer', 'Customer', 'Customer', 'Charging', 'Charging']
}
df = pd.DataFrame(data)
plt.switch_backend('TkAgg')
depot_index = df.index[df['Type'] == 'Depot'].tolist()
customer_indices = df.index[df['Type'] == 'Customer'].tolist()
charging_stations_indices = df.index[df['Type'] == 'Charging'].tolist()
# 车辆参数

no_of_customers = len(customer_indices)
no_of_stations = len(charging_stations_indices)

# ?????????????????????????????????
Q = 20  # 车辆容量
no_of_vehicles = 3  # 车辆数量
initial_battery = 150
battery_consumption_rate = 1.3  # 电量消耗率

df = pd.DataFrame(data)

# 定义坐标和需求
coordinates = df[['X', 'Y']].values
Demand = df['Demand'].values
Service_time = df['time'].values
start_time = df['start_time'].values
end_time = df['end_time'].values

n = len(coordinates)
depot = coordinates[depot_index[0], :] if depot_index else None
customers = coordinates[customer_indices, :]
stations = coordinates[charging_stations_indices, :]
M = 1000  # big number
# 创建Gurobi模型
m = Model("MVRP")
x, y, z, t = {}, {}, {}, {}  # t 是时间变量

# 距离矩阵
dist_matrix = np.empty([n, n])
for i in range(len(coordinates)):
    for j in range(len(coordinates)):
        x[i, j] = m.addVar(vtype=GRB.BINARY, name="x%d,%d" % (i, j))
        dist_matrix[i, j] = np.sqrt((coordinates[i, 0] - coordinates[j, 0]) ** 2 + (coordinates[i, 1] - coordinates[j, 1]) ** 2)
        if i == j:
            dist_matrix[i, j] = M  # big 'M'
m.update()

# 累积需求和时间变量
for j in range(n):
    y[j] = m.addVar(vtype=GRB.INTEGER, name="y%d" % j)
    z[j] = m.addVar(vtype=GRB.INTEGER, name="z%d" % j)
    t[j] = m.addVar(vtype=GRB.CONTINUOUS, name="t%d" % j)  # 时间变量
m.update()

# 添加约束
# 约束1：每个节点的车辆出发次数必须为1（除充电站）
for i in range(n - 1):
    if i + 1 not in charging_stations_indices:
        m.addConstr(quicksum(x[(i + 1, j)] for j in range(n)) == 1)
m.update()

# 约束2：每个节点的车辆到达次数必须为1（除充电站）
for j in range(n - 1):
    if j + 1 not in charging_stations_indices:
        m.addConstr(quicksum(x[(i, j + 1)] for i in range(n)) == 1)
m.update()

# 约束3：充电站只能被访问一次
for idx in charging_stations_indices:
    m.addConstr(quicksum(x[i, idx] for i in range(n)) <= 1, name=f"max_one_incoming_{idx}")
    m.addConstr(quicksum(x[idx, j] for j in range(n)) == quicksum(x[i, idx] for i in range(n)), name=f"out_equals_in_{idx}")

# 约束4：车辆从仓库出发的次数必须等于车辆数量
m.addConstr(quicksum(x[(0, j)] for j in range(n)) == no_of_vehicles)
m.update()

# 约束5：车辆返回仓库的次数必须等于车辆数量
m.addConstr(quicksum(x[(i, 0)] for i in range(n)) == no_of_vehicles)
m.update()

# 约束6：累积需求不超过车辆容量
for j in range(n - 1):
    m.addConstr(y[j + 1] <= Q)
    m.addConstr(y[j + 1] >= Demand[j + 1])
    for i in range(n - 1):
        m.addConstr(y[j + 1] >= y[i + 1] + Demand[j + 1] * (x[i + 1, j + 1]) - Q * (1 - (x[i + 1, j + 1])))
m.update()

# 约束7：累积需求满足条件
for j in range(1, n):
    m.addConstr(y[j] == quicksum((y[i] + Demand[j]) * x[i, j] for i in range(n) if i != j), "Cumulative_Demand_%d" % j)
m.update()

# 约束8：初始化电量变量
b = {}
for i in range(n):
    b[i] = m.addVar(0, initial_battery, 0.0, vtype=GRB.CONTINUOUS, name=f"battery_{i}")
m.update()
# 约束9：起始节点电量为初始电量
m.addConstr(b[0] == initial_battery, name="initial_battery")
# 约束10：电量不为负
m.addConstrs((b[i] >= 0 for i in range(n)), name="non_negative_battery")
# 约束11：充电站访问次数限制
for idx in charging_stations_indices:
    m.addConstr(quicksum(x[i, idx] for i in range(n)) <= 5, name=f"charging_station_visit_limit_{idx}")
# 约束12：辅助变量，表示充电站是否被访问
visited = {}
for idx in charging_stations_indices:
    visited[idx] = m.addVar(vtype=GRB.BINARY, name=f"visited_{idx}")
    m.addConstr(visited[idx] == quicksum(x[i, idx] for i in range(n)), name=f"visited_constr_{idx}")
# 约束13：充电站电量更新
for idx in charging_stations_indices:
    m.addConstr(b[idx] == initial_battery * visited[idx], name=f"recharge_at_station_{idx}")
# 约束14：从仓库到其他节点的电量约束
for j in range(1, n):
    m.addConstr(b[j] - b[0] + battery_consumption_rate * dist_matrix[0, j] * x[0, j] - initial_battery * (1 - x[0, j]) <= 0, name=f"battery_from_depot_to_{j}")
# 约束15：其他节点间的电量约束
for i in range(1, n-2):
    for j in range(1, n-2):
        if i != j:
            m.addConstr(b[j] - b[i] + battery_consumption_rate * dist_matrix[i, j] * x[i, j] - initial_battery * (1 - x[i, j]) <= 0, name=f"battery_from_{i}_to_{j}")
# 约束16：充电站电量更新
for idx in charging_stations_indices:
    for j in range(n):
        if idx != j:
            m.addConstr(b[j] >= b[i] - battery_consumption_rate * dist_matrix[i, j] * x[i, j], name=f"battery_update_from_{i}_to_{j}")
# 约束17：充电站电量恢复
for idx in charging_stations_indices:
    m.addConstr(b[idx] == initial_battery * visited[idx], name=f"recharge_at_station_{idx}")
# 约束18：电量足够从i到j
for i in range(n):
    for j in range(n):
        if i != j:
            m.addConstr(b[i] - battery_consumption_rate * dist_matrix[i, j] * x[i, j] >= 0, name=f"battery_sufficient_from_{i}_to_{j}")
m.update()

# 添加时间窗口约束
for i in range(1, no_of_customers + 1):
    m.addConstr(t[i] >= start_time[i], name=f"start_time_window_{i}")
    m.addConstr(t[i] <= end_time[i], name=f"end_time_window_{i}")
    m.addConstr(t[i] + Service_time[i] <= end_time[i], name=f"service_time_window_{i}")


for j in range(1, n):
    m.addConstr(t[j] == quicksum((t[i] + Service_time[j]) * x[i, j] for i in range(n) if i != j), "Cumulative_Demand_%d" % j)
m.update()

# 目标函数
m.setObjective(quicksum(quicksum(x[(i, j)] * dist_matrix[(i, j)] for j in range(n)) for i in range(n)), GRB.MINIMIZE)
m.update()

# 优化模型
m.optimize()

def print_battery_changes(m):
    if m.status == GRB.OPTIMAL or m.status == GRB.SUBOPTIMAL:
        sol_x = m.getAttr('x', x)
        sol_b = m.getAttr('x', b)
        print("\nDetailed battery usage and changes on routes:")
        for i in range(n):
            for j in range(n):
                if sol_x[i, j] >= 0.5:
                    battery_used = battery_consumption_rate * dist_matrix[i, j]
                    battery_left = sol_b[i] - battery_used
                    print(f"Route from Node {i} to Node {j}:")
                    print(f"  Initial Battery at Node {i}: {sol_b[i]:.2f}")
                    print(f"  Battery Used: {battery_used:.2f}")
                    print(f"  Remaining Battery at Node {j}: {battery_left:.2f}")

print_battery_changes(m)

if m.status == GRB.OPTIMAL:
    sol_y, sol_x, sol_z, sol_t = m.getAttr('x', y), m.getAttr('x', x), m.getAttr('x', z), m.getAttr('x', t)
    X, Y, Z, T = np.empty([n, n]), np.empty([n]), np.empty([n]), np.empty([n])
    for i in range(n):
        Y[i] = sol_y[i]
        Z[i] = sol_z[i]
        T[i] = sol_t[i]
        for j in range(n):
            X[i, j] = int(sol_x[i, j])
    print('\nObjective is:', m.objVal)
    print('\nDecision variable X (binary decision of travelling from one node to another):\n', X.astype('int32'))
    print('\nDecision variable z:(service start time of every customers in minutes)\n', Z.astype('int32')[1:])
    print('\nDecision variable y (cumulative demand collected at every customer node):\n', Y.astype('int32')[1:])
    print('\nDecision variable t (arrival time at every node):\n', T.astype('int32'))

    sol_b = m.getAttr('x', b)
    B = np.empty([n])
    for i in range(n):
        B[i] = sol_b[i]
    print('\nDecision variable b (battery levels at each node):\n', B.astype('int32'))

    def plot_tours(solution_x, battery_levels, demands, arrival_times):
        tours = [[i, j] for i in range(solution_x.shape[0]) for j in range(solution_x.shape[1]) if solution_x[i, j] == 1]
        plt.figure(figsize=(8, 6))
        for t, tour in enumerate(tours):
            plt.plot([df["X"][tour[0]], df["X"][tour[1]]], [df["Y"][tour[0]], df["Y"][tour[1]]], color="black", linewidth=0.5)
        plt.scatter(df["X"][1:no_of_customers + 1], df["Y"][1:no_of_customers + 1], marker='x', color='g', label="Customers")
        plt.scatter(df["X"][0], df["Y"][0], marker='o', color='b', label="Depot")
        plt.scatter(df["X"][no_of_customers + 1:], df["Y"][no_of_customers + 1:], marker='s', color='r', label="Charging Stations")
        for i in range(len(df["X"])):
            plt.annotate(f'Node {i}\nBattery: {battery_levels[i]:.2f}\nDemand: {demands[i]}\nArrival: {arrival_times[i]:.2f}', (df["X"][i], df["Y"][i]),
                         textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)
        plt.xlabel("X"), plt.ylabel("Y"), plt.title("Vehicle Routes")
        plt.legend(loc="best")
        plt.grid(True)
        plt.show()

    plot_tours(X, B, Demand, T)
else:
    print("No feasible solution found.")
