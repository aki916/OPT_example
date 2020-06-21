import pulp
import random
import networkx as nx
import math,time
import matplotlib.pyplot as plt

def make_random_graph(n):
    G = nx.DiGraph()
    for i in range(n):
        G.add_node(i,x=random.randint(0,10),y=random.randint(0,10))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist = math.sqrt((G.nodes()[i]['x']-G.nodes()[j]['x'])**2 + (G.nodes()[i]['y']-G.nodes()[j]['y'])**2)
                G.add_edge(i,j,dist=dist)
    return G

def get_random_sequential_order(num_node,m):
    box = set()
    # 選好順序(i,j)をm個まで取得 (i < j)
    while len(box) < m:
        i = random.randint(0,num_node-2)
        j = random.randint(i+1,num_node-1)
        if (i,j) not in box:
            box.add((i,j))
    return box

def solve_SOP(G,precedense,num_node,ss):
    problem = pulp.LpProblem(name='SOP',sense=pulp.LpMinimize)
    x = {(i,j):pulp.LpVariable(cat="Binary",name=f"x_{i}_{j}") for (i,j) in G.edges()}
    u = {i:pulp.LpVariable(cat="Integer",name=f"u_{i}",lowBound=1,upBound=num_node-1) for i in G.nodes()}
    cost = {(i,j):G.adj[i][j]['dist'] for (i,j) in G.edges()}

    problem += pulp.lpSum([x[(i,j)]*cost[(i,j)] for (i,j) in G.edges()])


    for i in G.nodes():
        if i != num_node-1:
            problem.addConstraint(pulp.lpSum([x[(i,j)] for j in range(num_node) if j != i]) == 1, f'outflow_{i}')
        if i != 0:
            problem.addConstraint(pulp.lpSum([x[(j,i)] for j in range(num_node) if j != i]) == 1, f'inflow_{i}')

    for i,j in G.edges():
        if i != ss and j != ss:
            problem.addConstraint(u[i]-u[j]+(num_node-1)*x[i,j] <= num_node-2, f'up_{i}_{j}')

    for i,j in precedense:
        problem.addConstraint(u[i]+1 <= u[j], f'sequential_{i}_{j}')

    u[ss] = 0

    print('start solving')
    start = time.time()
    status = problem.solve(pulp.CPLEX())
    # status = problem.solve()
    print(pulp.LpStatus[status])

    duartion = time.time()-start
    print(duartion)

    if pulp.LpStatus[status] != 'Optimal':
        print('Infeasible!')
        exit()

    return x,u,duartion

def plot(G,x,u,precedense,ss):
    pos = {i: (G.nodes()[i]['x'], G.nodes()[i]['y']) for i in G.nodes()}
    nx.draw_networkx_nodes(G, pos, node_size=100, alpha=1, node_color='skyblue')
    edgelist = [e for e in G.edges() if x[e].value() > 0]
    nx.draw_networkx_edges(G, pos, edgelist=edgelist,width=3)
    precedense = [e for e in precedense]
    nx.draw_networkx_edges(G, pos, edgelist=precedense,edge_color='red')
    for i in G.nodes():
        if i != ss:
            plt.text(G.nodes()[i]['x'],G.nodes()[i]['y'],int(u[i].value()))
        else:
            plt.text(G.nodes()[i]['x'],G.nodes()[i]['y'],u[i])


    plt.show()

def main():
    # node数
    num_node = 10
    # 選好順序の個数
    num_precedence = 5
    # 始点
    ss = 0
    # 選好順序リストの取得
    precedense = get_random_sequential_order(num_node,num_precedence)
    print(precedense)
    # random graphの獲得
    G = make_random_graph(num_node)
    # SOP
    x,u = solve_SOP(G,precedense,num_node,ss)
    # 描画
    plot(G,x,u,precedense,ss)





if __name__ == '__main__':
    main()