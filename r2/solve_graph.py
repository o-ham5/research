import pulp
from pulp import CPLEX, lpSum, GUROBI
import time
import networkx as nx
from networkx.algorithms.approximation import maximum_independent_set, min_weighted_vertex_cover, max_clique, clique_removal, ramsey
import matplotlib.pyplot as plt
from itertools import combinations
# from ortoolpy import min_node_cover

from plot import plot_gc


class MyCPLEX(CPLEX):
    def __init__(self, mip=True, msg=True, timeLimit=None,
                 epgap=None, logfilename=None, threads=None):
        super().__init__(
            mip, msg, timeLimit, epgap, logfilename
        )
        self.threads = threads
    
    def actualSolve(self, lp, callback=None):
        self.buildSolverModel(lp)
        # set threads
        if self.threads is not None:
            self.solverModel.parameters.threads.set(self.threads)
        self.callSolver(lp)
        solutionStatus = self.findSolutionValues(LookupError)
        for var in lp.variables():
            var.modified = False
        for constraint in lp.constraints.values():
            constraint.modified = False     
        return solutionStatus
    

def solve_gc_IP(G, k, solver, threads, plot, pos):
    """
    k-彩色問題を通常の定式化で解く
    入力
        G: グラフ
        k(int): 色数
        solver: "cbc" or "cplex" or "gurobi"
        threads: gurobiの場合指定
        plot: trueの場合plotする
    出力
        

    # ><><><><><><><><><><><><><><><><><><><><><
    # ><><><><><><><><><><><><><><><><><><><><><
    # ><><><><><><><><><><><><><><><><><><><><><
    # ><><><><><><><><><><><><><><><><><><><><><

    """

    colors = list(range(1, k+1))

    nodes = G.nodes
    edges = G.edges

    problem = pulp.LpProblem(f'{k}-Coloring Problem', pulp.LpMinimize)

    problem += 0

    var = {
        node: pulp.LpVariable.dicts(str(node), colors, cat = pulp.LpBinary) for node in nodes
    }


    # １つの領域は一色でしか塗らない
    for node in nodes:
        problem += lpSum(var[node].values()) == 1

    # 隣り合った領域は同色で塗らない
    for node1, node2 in edges:
        for color in colors:
            problem += var[node1][color] + var[node2][color] <= 1

    start = time.time()

    if solver == 'cbc':
        result = problem.solve()
    elif solver == 'cplex':
        result = problem.solve(CPLEX(msg=True, mip=True))
    elif solver == 'gurobi':
        result = problem.solve(GUROBI(msg=True, mip=True, Threads=threads))

    elapsed_time = time.time() - start

    print('elapsed_time :', elapsed_time)

    print('result : '+ pulp.LpStatus[result])
    
    if pulp.LpStatus[result] == 'Optimal':
        print('value =', pulp.value(problem.objective))

        output = {}
        for node, v in var.items():
            for color, flg in v.items():
                if flg.value() == 1:
                    output[node] = color
                    break
        

        if plot:
            plot_gc(G, output, pos)
    
def solve_vc_approximation(G):
    """
    最小頂点被覆問題を近似解法で解く
    (Reference: A local-ratio theorem for approximating the weighted vertex cover problem)
    入力
        G: グラフ
    出力

    # ><><><><><><><><><><><><><><><><><><><><><
    # ><><><><><><><><><><><><><><><><><><><><><

    """
    start = time.time()

    # vc_nodes = min_weighted_vertex_cover(G)
    
    cost = dict(G.nodes(data=None, default=1))
    for u, v in G.edges():
        min_cost = min(cost[u], cost[v])
        cost[v] -= min_cost
    vc_nodes = {u for u, c in cost.items() if c == 0}

    elapsed_time = time.time() - start
    
    print('elapsed_time :', elapsed_time)
    print(f'covered vertex number: {len(vc_nodes)}')
    print(f'covered vertex: {vc_nodes}')

def solve_vc_IP(G, solver, threads):
    """
    最小頂点被覆問題を通常の定式化で解く
    入力
        G: グラフ
        solver: "cbc" or "cplex" or "gurobi"
        threads: gurobiの場合指定
    出力
        xの値(0 or 1)のリスト(G.nodesの順)

    # ><><><><><><><><><><><><><><><><><><><><><
    # ><><><><><><><><><><><><><><><><><><><><><

    """

    problem = pulp.LpProblem(sense=pulp.LpMinimize)

    var_node = {
        node: pulp.LpVariable(str(node), cat = pulp.LpBinary) for node in G.nodes
    }
    for node1, node2 in G.edges():
        problem += var_node[node1] + var_node[node2] >= 1
    problem += lpSum([var_node[node] for node in G.nodes])
    
    start = time.time()

    if solver == 'cbc':
        result = problem.solve()
    elif solver == 'cplex':
        result = problem.solve(CPLEX(msg=True, mip=True))
    elif solver == 'gurobi':
        result = problem.solve(GUROBI(msg=True, mip=True, Threads=threads))

    elapsed_time = time.time() - start
    
    print('elapsed_time :', elapsed_time)

    print('result : '+ pulp.LpStatus[result])

    if pulp.LpStatus[result] == 'Optimal':
        
        # answer = [i for i, x in enumerate(var_node) if value(x) > 0.5]
        answers = [pulp.value(var_node[node]) for node in G.nodes]

        print('value =', pulp.value(problem.objective))

        return answers

    else:
        return None

def solve_mis(G):
    """
    最大独立集合問題を既存の近似解法で解く
    (Reference: Approximating Maximum Independent Sets by Excluding Subgraphs)
    補グラフの最大クリークを近似的に求めることで元グラフの最大独立集合を近似値として求めている
    入力
        G: 無向グラフ
    出力
        なし
        

    # ><><><><><><><><><><><><><><><><><><><><><
    # ><><><><><><><><><><><><><><><><><><><><><

    """
    
    start = time.time()

    indep_nodes = maximum_independent_set(G)

    elapsed_time = time.time() - start
    
    print('elapsed_time :', elapsed_time)
    print(f'independent set number: {len(indep_nodes)}')
    print(f'independent set: {indep_nodes}')

def solve_mcq_approximation(G):
    """
    最大クリーク問題を既存の近似解法で解く
    (Reference: Approximating Maximum Independent Sets by Excluding Subgraphs)
    入力
        G: 無向グラフ
    出力
        なし
        

    # ><><><><><><><><><><><><><><><><><><><><><
    # ><><><><><><><><><><><><><><><><><><><><><
    # ><><><><><><><><><><><><><><><><><><><><><

    """
    

    if G is None:
        raise ValueError("Expected NetworkX graph!")

    st1 = time.time()
    cgraph = nx.complement(G)
    et1 = time.time()

    print('elapsed_time (complement) :', et1-st1)

    st2 = time.time()
    iset, _ = clique_removal(cgraph)
    et2 = time.time()

    print('elapsed_time (clique) :', et2-st2)
    print(f'maximum clique number: {len(iset)}')
    print(f'maximum clique: {iset}')

def solve_mcq_IP(G, solver, threads):
    """
    最大クリーク問題を通常の定式化で解く
    入力
        G: グラフ
        solver: "cbc" or "cplex" or "gurobi"
        threads: gurobiの場合指定
    出力
        xの値(0 or 1)のリスト(G.nodesの順)

    # ><><><><><><><><><><><><><><><><><><><><><
    # ><><><><><><><><><><><><><><><><><><><><><
    # ><><><><><><><><><><><><><><><><><><><><><
    # ><><><><><><><><><><><><><><><><><><><><><
    # ><><><><><><><><><><><><><><><><><><><><><
    # ><><><><><><><><><><><><><><><><><><><><><

    """

    problem = pulp.LpProblem(sense=pulp.LpMaximize)

    var_node = {
        node: pulp.LpVariable(str(node), cat = pulp.LpBinary) for node in G.nodes
    }
    for node1, node2 in combinations(G.nodes, 2):
        if not (node1, node2) in G.edges:
            problem += var_node[node1] + var_node[node2] <= 1
    
    problem += lpSum([var_node[node] for node in G.nodes])
    
    start = time.time()

    if solver == 'cbc':
        result = problem.solve()
    elif solver == 'cplex':
        result = problem.solve(CPLEX(msg=True, mip=True))
    elif solver == 'gurobi':
        result = problem.solve(GUROBI(msg=True, mip=True, Threads=threads))

    elapsed_time = time.time() - start
    
    print('elapsed_time :', elapsed_time)

    print('result : '+ pulp.LpStatus[result])

    if pulp.LpStatus[result] == 'Optimal':
        
        # answer = [i for i, x in enumerate(var_node) if value(x) > 0.5]
        answers = [pulp.value(var_node[node]) for node in G.nodes]

        print('value =', pulp.value(problem.objective))

        return answers

    else:
        return None

    

# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><


def solve_gc_Welsh_Powell(G):
    st = time.time()
    #sorting the nodes based on it's valency
    node_list = sorted(G.nodes(), key =lambda x:len(list(G.neighbors(x))))
    col_val = {} #dictionary to store the colors assigned to each node
    col_val[node_list[0]] = 0 #assign the first color to the first node
    # Assign colors to remaining N-1 nodes
    for node in node_list[1:]:
        available = [True] * len(G.nodes()) #boolean list[i] contains false if the node color 'i' is not available

        #iterates through all the adjacent nodes and marks it's color as unavailable, if it's color has been set already
        for adj_node in G.neighbors(node): 
            if adj_node in col_val.keys():
                col = col_val[adj_node]
                available[col] = False
        clr = 0
        for clr in range(len(available)):
            if available[clr] == True:
                break
        col_val[node] = clr
    print('chromatic number = ', max(col_val.values()) + 1)
    print('elapsed time =', time.time() - st)
    return col_val

