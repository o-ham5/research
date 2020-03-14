import networkx as nx
import matplotlib.pyplot as plt
import japanmap
import random
from itertools import combinations, chain


def grid_graph(ROW, COL, diagonal=True, plot=False):
    G = nx.grid_graph([COL, ROW])

    if diagonal:
        for row in range(ROW-1):
            for col in range(COL-1):
                G.add_edge((row, col), (row+1, col+1))
                G.add_edge((row, col+1), (row+1, col))

    if plot:
        nx.draw_networkx(G, pos={n:n for n in G}, with_labels=False, font_weight='bold', node_size=500)
        plt.axis('off')
        plt.show()
    
    return G


def japan_graph():
    pref_codes = tuple(range(1, 48))
    G_japan = nx.Graph()
    G_japan.add_nodes_from(pref_codes)
    for code in pref_codes:
        G_japan.add_edges_from([(code, ncode) for ncode in japanmap.adjacent(code) if code < ncode])
    
    return G_japan


def broom_graph(length, cols, plot=False):

    G = nx.Graph()
    for i in range(length):
        for j in range(cols):
            G.add_edge((i, j/(cols-1)), (i+0.33, 0.5))
            G.add_edge((i+1, j/(cols-1)), (i+0.66, 0.5))
        G.add_edge((i+0.33, 0.5), (i+0.66, 0.5))

    if plot:
        nx.draw_networkx(G, pos={n:n for n in G}, with_labels=False, font_weight='bold', node_size=500)
        plt.axis('off')
        plt.show()
        
    return G

def one_line_graph(n):

    G = nx.Graph()
    for i in range(n):
        if i != 0:
            G.add_edge(i-1, i)

    return G

def complete_graph(n):

    return nx.complete_graph(n)

def random_graph(n, m):
    
    if n*(n-1)/4 > m:
        return nx.gnm_random_graph(n, m)
    else:
        return nx.dense_gnm_random_graph(n, m)

def random_clique_graph(n, m, K):

    while True:

        G = nx.gnm_random_graph(n, m)

        max_clique_num = len(max(list(nx.find_cliques(G)), key=lambda x:len(x)))
        if max_clique_num <= K:
            break


    return G

def random_planar_graph(n, m):
    
    while True:
        G = nx.gnm_random_graph(n, m)
        if nx.check_planarity(G)[0]:
        # if nx.is_connected(G):
            break

    return G

def random_connect_graph(n, m):
    
    while True:
        G = nx.gnm_random_graph(n, m)
        if nx.is_connected(G):
            break

    return G


def complete_bipartite_graph(n1, n2):

    return nx.complete_multipartite_graph(n1, n2)

def complete_multipartite_graph(nn):

    return nx.complete_multipartite_graph(*nn)

def cycle_graph(n, plot=False):

    G = nx.cycle_graph(n)

    if plot:
        nx.draw_networkx(G, with_labels=False, font_weight='bold', node_size=500)
        plt.axis('off')
        plt.show()

    return G

def subset_graph(n, subset_num):
    
    def powerset(iterable):
        
        s = list(iterable)
        return list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))

    main_set = list(range(1, n+1))
    use_sets = []

    p_set_num = pow(2, n)
    if p_set_num / 2 >= subset_num:
        while len(use_sets) < subset_num:

            bool_list = [random.randint(0, 1) for i in main_set]
            subset = tuple([main_set[i] for i, p in enumerate(bool_list) if p])

            if not subset in use_sets:
                use_sets.append(subset)
            
    else:
        power_set = powerset(main_set)
        use_sets = random.sample(power_set, subset_num)
        
    print(f'use subsets: {use_sets}')

    G = nx.Graph()
    G.add_nodes_from(use_sets)
    for set1, set2 in combinations(use_sets, 2):
        if set(set1) & set(set2):
            G.add_edge(set1, set2)

    return G



    




