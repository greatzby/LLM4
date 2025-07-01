# create_graph_composition.py (修复版本)
import networkx as nx
import random
import os
import argparse
import numpy as np

def generate_composition_graph(num_nodes_per_stage, edge_prob_within, edge_prob_between, num_stages=3):
    """
    创建一个分层的有向图，用于测试组合能力
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
    # 使用反向图来找到所有能到达target的节点
    G_reverse = G.reverse()
    reachable = set(nx.descendants(G_reverse, target_node))
    return list(reachable)

def obtain_reachability(G):
    """计算所有节点的可达性"""
    reachability = {}
    pairs = 0
    for node in G.nodes():
        reachable_nodes = get_reachable_nodes(G, node)
        reachability[node] = reachable_nodes
        pairs += len(reachable_nodes)
    return reachability, pairs

def can_reach(G, source, target):
    """检查source是否能到达target"""
    try:
        nx.shortest_path(G, source, target)
        return True
    except nx.NetworkXNoPath:
        return False

def random_walk(G, source_node, target_node, max_attempts=10):
    """在图中执行随机游走生成路径"""
    for _ in range(max_attempts):
        try:
            # 使用最短路径作为基础
            shortest_path = nx.shortest_path(G, source_node, target_node)
            
            # 可以在这里添加一些随机性，比如偶尔偏离最短路径
            # 但为了确保能生成有效路径，我们先使用最短路径
            return shortest_path
        except nx.NetworkXNoPath:
            return None
    
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
    s1_s2_count = 0
    for source in S1:
        for target in S2:
            if can_reach(G, source, target):
                # 添加多条路径
                for _ in range(train_paths_per_pair):
                    path = random_walk(G, source, target)
                    if path:
                        train_set.append([source, target] + path)
                        s1_s2_count += 1
    print(f"  Generated {s1_s2_count} S1→S2 paths")
    
    # 训练集：S2→S3 的路径
    print("Generating S2→S3 training paths...")
    s2_s3_count = 0
    for source in S2:
        for target in S3:
            if can_reach(G, source, target):
                # 添加多条路径
                for _ in range(train_paths_per_pair):
                    path = random_walk(G, source, target)
                    if path:
                        train_set.append([source, target] + path)
                        s2_s3_count += 1
    print(f"  Generated {s2_s3_count} S2→S3 paths")
    
    # 测试集：S1→S3 的路径（测试组合能力）
    print("Generating S1→S3 test paths...")
    s1_s3_count = 0
    for source in S1:
        for target in S3:
            if can_reach(G, source, target):
                path = random_walk(G, source, target)
                if path:
                    # 验证路径确实经过S2
                    has_s2 = any(node in S2 for node in path[1:-1])
                    if has_s2:
                        test_set.append([source, target] + path)
                        s1_s3_count += 1
    print(f"  Generated {s1_s3_count} S1→S3 paths")
    
    # 可选：添加一些S1→S2和S2→S3的测试路径以验证基础能力
    print("Adding some basic test paths...")
    basic_count = 0
    
    # S1→S2测试路径（少量）
    for _ in range(5):
        source = random.choice(S1)
        target = random.choice(S2)
        if can_reach(G, source, target):
            path = random_walk(G, source, target)
            if path:
                test_set.append([source, target] + path)
                basic_count += 1
    
    # S2→S3测试路径（少量）
    for _ in range(5):
        source = random.choice(S2)
        target = random.choice(S3)
        if can_reach(G, source, target):
            path = random_walk(G, source, target)
            if path:
                test_set.append([source, target] + path)
                basic_count += 1
    
    print(f"  Added {basic_count} basic test paths")
    
    return train_set, test_set

def obtain_stats(dataset, stage_info=None):
    """统计数据集信息"""
    if not dataset:
        print("  Dataset is empty!")
        return
        
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
    
    print(f'  Number of unique source-target pairs: {len(pairs)}')
    print(f'  Total paths: {len(dataset)}')
    print(f'  Stage transitions: {stage_transitions}')
    
    # 打印路径长度分布
    print('  Path length distribution:')
    for ii in range(3, len(len_stats)):
        if len_stats[ii] > 0:
            print(f'    Length {ii-2}: {len_stats[ii]} paths')

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
    
    # 验证图的连通性
    s1_to_s2_pairs = sum(1 for s1 in stages[0] for s2 in stages[1] if can_reach(G, s1, s2))
    s2_to_s3_pairs = sum(1 for s2 in stages[1] for s3 in stages[2] if can_reach(G, s2, s3))
    s1_to_s3_pairs = sum(1 for s1 in stages[0] for s3 in stages[2] if can_reach(G, s1, s3))
    
    print(f"\nConnectivity check:")
    print(f"  S1→S2 pairs: {s1_to_s2_pairs}")
    print(f"  S2→S3 pairs: {s2_to_s3_pairs}")
    print(f"  S1→S3 pairs: {s1_to_s3_pairs}")
    
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
    
    # 打印一些示例路径
    if train_set:
        print("\nSample training paths:")
        for i in range(min(3, len(train_set))):
            print(f"  {train_set[i][:10]}..." if len(train_set[i]) > 10 else f"  {train_set[i]}")
    
    if test_set:
        print("\nSample test paths:")
        for i in range(min(3, len(test_set))):
            print(f"  {test_set[i][:10]}..." if len(test_set[i]) > 10 else f"  {test_set[i]}")