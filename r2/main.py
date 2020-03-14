from make_graph import *
from solve_graph import *
# ><><><><><><><><><><><><><><><><><><><><><
# ><><><><><><><><><><><><><><><><><><><><><
import networkx as nx

import argparse, argcomplete
import os
import pickle
from collections import defaultdict

from timeout_decorator import timeout, TimeoutError
import subprocess
import sys

def main():

    sys.setrecursionlimit(100000)

    parser = argparse.ArgumentParser()

    # グラフ作成時のパラメータ色々
    parser.add_argument('-r', '--rows', type=int, default=1, help='')
    parser.add_argument('-c', '--cols', type=int, default=1, help='')
    parser.add_argument('-l', '--length', type=int, default=1, help='')
    parser.add_argument('-n', '--nodes', type=int, default=1, help='')
    parser.add_argument('-n2', '--nodes2', type=int, default=1, help='')
    parser.add_argument('-m', '--edges', type=int, default=1, help='')
    parser.add_argument('-k', '--param', type=int, default=4, help='')
    parser.add_argument('-K', '--clique_size', type=int, default=4, help='')
    parser.add_argument('-S', '--subset_num', default=1, help='')
    parser.add_argument('--nn', nargs="*", help='')

    # ><><><><><><><><><><><><><><><><><><><><><
    parser.add_argument('--A', type=int, default=None, help='')
    parser.add_argument('--B', type=int, default=None, help='')
    parser.add_argument('--C', type=int, default=None, help='')

    # グラフ・問題の設定
    parser.add_argument('-g', '--graph', choices=['grid', 'grid_diag', 'random', 'complete', 'one_line', 'broom', 'cycle', 'japan', 'complete_bi', 'complete_multi', 'random_planar', 'random_clique', 'random_connect', 'subset'], default='grid', help='')
    parser.add_argument('-p', '--problem', choices=['gc', 'vc', 'mmm', 'cj', 'gcut', 'mis', 'mc'], default='vc', help='')

    # ><><><><><><><><><><><><><><><><><><><><><
    # ><><><><><><><><><><><><><><><><><><><><><

    # pulpで解く際のソルバー・並列数指定
    parser.add_argument('-s', '--solver', choices=['gurobi', 'cplex', 'cbc'], default='cplex', help='')
    parser.add_argument('--cpu', type=int, default=1, help='')

    # plotがあるものは指定することで描写
    parser.add_argument('--plot', action='store_true', help='')
    # ><><><><><><><><><><><><><><><><><><><><><
    parser.add_argument('-w', '--write', action='store_true', help='')

    parser.add_argument('--src', nargs="*", default=[], help='source dir path')

    # argcomplete.autocomplete(parser)
    args = parser.parse_args()


    pos = False
    output = args.graph

    # 元グラフ生成
    if args.src:
        G = read_srcGraph(args.src[0])
        output = '_'.join(args.src[1:])
    else:
        if args.graph == 'grid':
            G = grid_graph(args.rows, args.cols, diagonal=False, plot=args.plot)
            pos = True
            output += f'_r{args.rows}c{args.cols}'
        elif args.graph == 'grid_diag':
            G = grid_graph(args.rows, args.cols, diagonal=True, plot=args.plot)
            pos = True
            output += f'_r{args.rows}c{args.cols}'
        elif args.graph == 'random':
            G = random_graph(args.nodes, args.edges)
            output += f'_n{args.nodes}m{args.edges}'
        elif args.graph == 'complete':
            G = complete_graph(args.nodes)
            output += f'_n{args.nodes}'
        elif args.graph == 'one_line':
            G = one_line_graph(args.nodes)
            output += f'_n{args.nodes}'
        elif args.graph == 'broom':
            G = broom_graph(args.length, args.cols, plot=args.plot)
            pos = True
            output += f'_l{args.length}c{args.cols}'
        elif args.graph == 'cycle':
            G = cycle_graph(args.nodes, plot=args.plot)
            output += f'_n{args.nodes}'
        elif args.graph == 'japan':
            G = japan_graph()
        elif args.graph == 'complete_bi':
            G = complete_bipartite_graph(args.nodes, args.nodes2)
            output += f'_n{args.nodes}n{args.nodes2}'
        elif args.graph == 'complete_multi':
            nn = tuple(map(int, args.nn))
            G = complete_multipartite_graph(nn)
            output += f'_nn{"x".join(args.nn)}'
        elif args.graph == 'random_planar':
            G = random_planar_graph(args.nodes, args.edges)
            output += f'_n{args.nodes}m{args.edges}'
        elif args.graph == 'random_clique':
            G = random_clique_graph(args.nodes, args.edges, args.param)
            output += f'_n{args.nodes}m{args.edges}K{args.param}'
        elif args.graph == 'random_connect':
            G = random_connect_graph(args.nodes, args.edges)
            output += f'_n{args.nodes}m{args.edges}'
        elif args.graph == 'subset':
            if args.subset_num == 'all':
                subset_num = pow(2, args.nodes)
            else:
                try:
                    subset_num = int(args.subset_num)
                except:
                    print(f'subset number error: {args.subset_num}')
                    exit()
            G = subset_graph(args.nodes, subset_num)
            output += f'_n{args.nodes}S{args.subset_num}'

    

    output += f'_{args.problem}'
    

    # ><><><><><><><><><><><><><><><><><><><><><

    # ><><><><><><><><><><><><><><><><><><><><><
    if args.problem == 'gc':
        # 参考のため最大クリークを求める
        try:
            max_clique_num = find_max_clique(G)
            print('max clique :', max_clique_num)
            
        except TimeoutError:
            print('timeout error: cannot find max clique.')
        # 参考のため平面グラフかどうか求める
        print('planarity :', nx.check_planarity(G)[0])
        ___, output = make_gc_graph(G, args.A, args.param, output)
    elif args.problem == 'vc':
        ___, output = make_vc_graph(G, args.A, args.B, output)
    elif args.problem == 'mmm':
        ___, output = make_mmm_graph(G, args.A, args.B, args.C, output)
    elif args.problem == 'cj':
        ___, output = make_cj_graph(G, args.A, args.B, args.clique_size, output)
    elif args.problem == 'gcut':
        ___, output = make_gcut_graph(G, args.A, args.B, output)
        # ><><><><><><><><><><><><><><><><><><><><><
    elif args.problem == 'mis':
        ___, output = make_mis_graph(G, args.A, args.B, output)
    elif args.problem == 'mc':
        ___, output = make_mc_graph(G, args.A, output)
        # ><><><><><><><><><><><><><><><><><><><><><


    output_dir = '../output/' + output
    if len(args.src) != 1:
        if os.path.exists(output_dir):
            while True:
                print(f"すでに{output_dir}は存在しますが続けますか？(y/n) : ", end="")
                inp = input()
                if inp == "y":
                    print("input: yes")
                    break
                elif inp == "n":
                    print("input: no")
                    exit()
                else:
                    print("'y' または 'n' をタイプしてください。")
            
        else:
                os.mkdir(output_dir)

        # "cj" , "gcut" , "mc"の場合はdsjフォーマットへ変換して保存
        write_srcGraph(G, output_dir)
        if args.problem == "cj" or args.problem == "gcut" or args.problem == "mc":
            write_dsj(G, output_dir)

    src_dir = args.src[0] if args.src else output_dir


    # ><><><><><><><><><><><><><><><><><><><><><
    # ><><><><><><><><><><><><><><><><><><><><><
    # ><><><><><><><><><><><><><><><><><><><><><
    # ><><><><><><><><><><><><><><><><><><><><><
    # ><><><><><><><><><><><><><><><><><><><><><


    # ><><><><><><><><><><><><><><><><><><><><><
    if args.write:
        # ><><><><><><><><><><><><><><><><><><><><><
        # ><><><><><><><><><><><><><><><><><><><><><
        # ><><><><><><><><><><><><><><><><><><><><><
        # ><><><><><><><><><><><><><><><><><><><><><


    # 既存手法で解く
    if args.opt == "all" or args.opt == "conv":

        # ><><><><><><><><><><><><><><><><><><><><><

        if args.problem == 'gc':
            print("< welsh_powell >")
            solve_gc_Welsh_Powell(G)
            print("< IP >")
            solve_gc_IP(G, args.param, args.solver, args.cpu, args.plot, pos)

        elif args.problem == 'vc':
            print("< approximation >")
            solve_vc_approximation(G)
            print("< IP >")
            solve_vc_IP(G, args.solver, args.cpu)

        # elif args.problem == 'mmm':
        
        elif args.problem == 'cj':
            # c_file = "mcqpsolve_tabu.c"
            # cmd1 = f"icc -O2 {c_dir}{c_file} -o {c_file[:-2]}".split(" ")
            # cmd2 = f"./{c_file[:-2]} {src_dir}/dsj_file.txt".split(" ")

            # f = open(f"log_{args.opt}.txt", "a")

            # print(f"running {' '.join(cmd1)}")
            # subprocess.call(cmd1, encoding='utf-8')
            # print(f"running {' '.join(cmd2)}")
            # subprocess.run(cmd2, encoding='utf-8', stdout=f)
        
            # f.close()

            print("< approximation >")
            solve_mcq_approximation(G)
            print("< IP >")
            solve_mcq_IP(G, args.solver, args.cpu)
            
        elif args.problem == 'gcut':
            """
            # ><><><><><><><><><><><><><><><><><><><><><
            """
            c_file = "gppsolve_tabu.c"
            cmd1 = f"icc -O2 {c_dir}{c_file} -o {c_file[:-2]}".split(" ")
            cmd2 = f"./{c_file[:-2]} {src_dir}/dsj_file.txt".split(" ")

            f = open(f"log_{args.opt}.txt", "a")

            print(f"running {' '.join(cmd1)}")
            subprocess.call(cmd1, encoding='utf-8')
            print(f"running {' '.join(cmd2)}")
            subprocess.run(cmd2, encoding='utf-8', stdout=f)
        
            f.close()
            
        elif args.problem == 'mis':
            solve_mis(G)

        elif args.problem == 'mc':
            """
            # ><><><><><><><><><><><><><><><><><><><><><
            # ><><><><><><><><><><><><><><><><><><><><><
            # ><><><><><><><><><><><><><><><><><><><><><
            # ><><><><><><><><><><><><><><><><><><><><><
            # ><><><><><><><><><><><><><><><><><><><><><
            # ><><><><><><><><><><><><><><><><><><><><><

            """
            c_file = "mcpsolve_tabu.c"
            cmd1 = f"icc -O2 {c_dir}{c_file} -o {c_file[:-2]}".split(" ")
            cmd2 = f"./{c_file[:-2]} {src_dir}/dsj_file.txt".split(" ")

            f = open(f"log_{args.opt}.txt", "a")

            print(f"running {' '.join(cmd1)}")
            subprocess.call(cmd1, encoding='utf-8')
            print(f"running {' '.join(cmd2)}")
            subprocess.run(cmd2, encoding='utf-8', stdout=f)

    # ><><><><><><><><><><><><><><><><><><><><><
    # ><><><><><><><><><><><><><><><><><><><><><
    if ___:

        # ><><><><><><><><><><><><><><><><><><><><><
        # ><><><><><><><><><><><><><><><><><><><><><
        # ><><><><><><><><><><><><><><><><><><><><><
        # ><><><><><><><><><><><><><><><><><><><><><


        # プロット
        if args.plot:
            if args.problem == 'gc' and args.graph == 'japan':
                # ><><><><><><><><><><><><><><><><><><><><><
            elif args.problem == 'gc':
                # ><><><><><><><><><><><><><><><><><><><><><
            elif args.problem == 'vc':
                # ><><><><><><><><><><><><><><><><><><><><><

    

@timeout(30)
def find_max_clique(G):
    max_clique = len(max(list(nx.find_cliques(G)), key=lambda x:len(x)))

    return max_clique

def write_srcGraph(G, output):
    with open(f"{output}/srcGraph.pkl", "wb") as f:
        pickle.dump(G, f)

def read_srcGraph(src_path):
    with open(f"{src_path}/srcGraph.pkl", "rb") as f:
        return pickle.load(f)

def write_dsj(G, output):

    G = nx.convert_node_labels_to_integers(G, first_label=1)

    conv_dsj = defaultdict(list)
    for node1, node2 in G.edges:
        conv_dsj[node1].append(node2)
        conv_dsj[node2].append(node1)

    conv_dsj = dict(sorted(conv_dsj.items()))
    for i in conv_dsj:
        conv_dsj[i] = sorted(conv_dsj[i])

    with open(f"{output}/dsj_file.txt", 'w') as f:
        for i in conv_dsj:
            f.write('{0:4d}'.format(i) + ' ' + '{0:4d}'.format(len(conv_dsj[i])) + ' ')
            for j in conv_dsj[i]:
                f.write(' ' + str(j))
            f.write('\n')


if __name__ == '__main__':
    main()