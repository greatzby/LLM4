# data/simple_graph/create_graph_composition.py
import networkx as nx
import random
import os
import argparse
import numpy as np

def generate_composition_graph(num_nodes_per_stage, edge_prob_within, edge_prob_between, num_stages=3):
    """
    创建一个分层的有向图，用于测试组合能力
    
    Args:
        num_nodes_per_stage: 每个阶段的节点数
        edge_prob_within: 组内边的概率
        edge_prob_between: 组间边的概率
        num_stages: 阶段数（默认3）
    """
    G = nx.DiGraph()
    total_nodes = num_nodes_per_stage * num_stages
    
    # 添加所有节点
    for i in range(total_nodes):
        G.add_node(i)
    
    # 定义每个阶段的节点集合
    stages = []
    for s in range(num_stages):
        stage_nodes = list(range(s * num_nodes_per_stage, (s + 1) * num_nodes_per_stage))
        stages.append(stage_nodes)
    
    # 添加组内边（同一阶段内的边）
    for stage_idx, stage_nodes in enumerate(stages):
        for i in stage_nodes:
            for j in stage_nodes:
                if i < j and random.random() < edge_prob_within:
                    G.add_edge(i, j)
    
    # 添加组间边（只从Si到Si+1）
    for s in range(num_stages - 1):
        current_stage = stages[s]
        next_stage = stages[s + 1]
        for i in current_stage:
            for j in next_stage:
                if random.random() < edge_prob_between:
                    G.add_edge(i, j)
    
    return G, stages

def get_reachable_nodes(G, target_node):
    """获取可以到达目标节点的所有节点"""
    TC = nx.transitive_closure(G)
    reachable_from = TC.predecessors(target_node)
    return list(reachable_from)

def obtain_reachability(G):
    """计算所有节点的可达性"""
    reachability = {}
    pairs = 0
    for node in G.nodes():
        reachability[node] = get_reachable_nodes(G, node)
        pairs += len(reachability[node])
    return reachability, pairs

def random_walk(G, source_node, target_node, reachability, max_attempts=10):
    """在图中执行随机游走"""
    for _ in range(max_attempts):
        path = [source_node]
        current = source_node
        visited = set([source_node])
        
        while current != target_node and len(path) < 100:
            neighbors = list(G.successors(current))
            # 只选择能到达目标的邻居
            valid_neighbors = [n for n in neighbors if n in reachability[target_node] or n == target_node]
            valid_neighbors = [n for n in valid_neighbors if n not in visited]
            
            if not valid_neighbors:
                break
                
            current = random.choice(valid_neighbors)
            path.append(current)
            visited.add(current)
            
            if current == target_node:
                return path
    
    return None

def create_composition_dataset(G, stages, reachability, train_paths_per_pair=10):
    """
    创建组合数据集
    训练集：S1→S2 和 S2→S3 的路径
    测试集：S1→S3 的路径
    """
    train_set = []
    test_set = []
    
    S1, S2, S3 = stages[0], stages[1], stages[2]
    
    # 训练集：S1→S2 的路径
    print("Generating S1→S2 training paths...")
    for source in S1:
        for target in S2:
            if target in reachability[target] and source in reachability[target]:
                # 添加多条路径
                for _ in range(train_paths_per_pair):
                    path = random_walk(G, source, target, reachability)
                    if path:
                        train_set.append([source, target] + path)
    
    # 训练集：S2→S3 的路径
    print("Generating S2→S3 training paths...")
    for source in S2:
        for target in S3:
            if target in reachability[target] and source in reachability[target]:
                # 添加多条路径
                for _ in range(train_paths_per_pair):
                    path = random_walk(G, source, target, reachability)
                    if path:
                        train_set.append([source, target] + path)
    
    # 测试集：S1→S3 的路径（测试组合能力）
    print("Generating S1→S3 test paths...")
    for source in S1:
        for target in S3:
            if target in reachability[target] and source in reachability[target]:
                path = random_walk(G, source, target, reachability)
                if path:
                    test_set.append([source, target] + path)
    
    # 可选：添加一些S1→S2和S2→S3的测试路径以验证基础能力
    print("Adding some basic test paths...")
    # S1→S2测试路径（少量）
    for _ in range(5):
        source = random.choice(S1)
        target = random.choice(S2)
        if target in reachability[target] and source in reachability[target]:
            path = random_walk(G, source, target, reachability)
            if path:
                test_set.append([source, target] + path)
    
    # S2→S3测试路径（少量）
    for _ in range(5):
        source = random.choice(S2)
        target = random.choice(S3)
        if target in reachability[target] and source in reachability[target]:
            path = random_walk(G, source, target, reachability)
            if path:
                test_set.append([source, target] + path)
    
    return train_set, test_set

def obtain_stats(dataset, stage_info=None):
    """统计数据集信息"""
    max_len = 0
    pairs = set()
    stage_transitions = {'S1->S2': 0, 'S2->S3': 0, 'S1->S3': 0}
    
    for data in dataset:
        max_len = max(max_len, len(data))
        source, target = data[0], data[1]
        pairs.add((source, target))
        
        # 统计阶段转换
        if stage_info:
            S1, S2, S3 = stage_info
            if source in S1 and target in S2:
                stage_transitions['S1->S2'] += 1
            elif source in S2 and target in S3:
                stage_transitions['S2->S3'] += 1
            elif source in S1 and target in S3:
                stage_transitions['S1->S3'] += 1
    
    len_stats = [0] * (max_len + 1)
    for data in dataset:
        length = len(data)
        len_stats[length] += 1
    
    print(f'Number of source-target pairs: {len(pairs)}')
    print(f'Stage transitions: {stage_transitions}')
    for ii in range(3, len(len_stats)):
        if len_stats[ii] > 0:
            print(f'  Paths with length {ii-2}: {len_stats[ii]}')

def format_data(data):
    """格式化数据为字符串"""
    return f"{data[0]} {data[1]} " + ' '.join(str(num) for num in data[2:]) + '\n'

def write_dataset(dataset, file_name):
    """写入数据集到文件"""
    with open(file_name, "w") as file:
        for data in dataset:
            file.write(format_data(data))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate composition graph for testing multi-hop reasoning')
    parser.add_argument('--nodes_per_stage', type=int, default=30, help='Number of nodes per stage')
    parser.add_argument('--edge_prob_within', type=float, default=0.1, help='Edge probability within stages')
    parser.add_argument('--edge_prob_between', type=float, default=0.3, help='Edge probability between stages')
    parser.add_argument('--num_stages', type=int, default=3, help='Number of stages (default: 3)')
    parser.add_argument('--train_paths_per_pair', type=int, default=10, help='Number of training paths per node pair')
    parser.add_argument('--experiment_name', type=str, default='composition', help='Experiment name for folder')
    
    args = parser.parse_args()
    
    # 总节点数
    total_nodes = args.nodes_per_stage * args.num_stages
    print(f"Creating composition graph with {total_nodes} nodes ({args.nodes_per_stage} per stage)")
    
    # 生成图
    G, stages = generate_composition_graph(
        args.nodes_per_stage, 
        args.edge_prob_within, 
        args.edge_prob_between,
        args.num_stages
    )
    
    print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    print(f"Stages: S1={stages[0][:5]}..., S2={stages[1][:5]}..., S3={stages[2][:5]}...")
    
    # 计算可达性
    reachability, feasible_pairs = obtain_reachability(G)
    print(f"Total feasible pairs: {feasible_pairs}")
    
    # 创建数据集
    train_set, test_set = create_composition_dataset(G, stages, reachability, args.train_paths_per_pair)
    
    # 创建输出目录
    folder_name = os.path.join(os.path.dirname(__file__), f'{args.experiment_name}_{total_nodes}')
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    # 统计信息
    print("\nTraining set statistics:")
    obtain_stats(train_set, stages)
    
    print("\nTest set statistics:")
    obtain_stats(test_set, stages)
    
    # 写入文件
    write_dataset(train_set, os.path.join(folder_name, f'train_{args.train_paths_per_pair}.txt'))
    write_dataset(test_set, os.path.join(folder_name, 'test.txt'))
    nx.write_graphml(G, os.path.join(folder_name, 'composition_graph.graphml'))
    
    # 保存阶段信息
    import pickle
    stage_info = {
        'stages': stages,
        'nodes_per_stage': args.nodes_per_stage,
        'num_stages': args.num_stages
    }
    with open(os.path.join(folder_name, 'stage_info.pkl'), 'wb') as f:
        pickle.dump(stage_info, f)
    
    print(f"\nDataset created successfully in {folder_name}/")