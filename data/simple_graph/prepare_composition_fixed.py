# prepare_composition_fixed.py
import os
import random
import pickle
import numpy as np
import networkx as nx
from collections import defaultdict

def prepare_composition_data_fixed(
    graph_file='data/simple_graph/composition_graph.graphml',
    output_dir='data/simple_graph/composition_90_fixed',
    train_paths_per_pair=10,
    val_ratio=0.15,
    test_samples_per_type=50,
    seed=42
):
    """准备平衡的组合能力训练数据"""
    
    random.seed(seed)
    np.random.seed(seed)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载图
    G = nx.read_graphml(graph_file)
    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # 定义阶段
    total_nodes = G.number_of_nodes()
    nodes_per_stage = total_nodes // 3
    
    S1 = list(range(0, nodes_per_stage))
    S2 = list(range(nodes_per_stage, 2 * nodes_per_stage))
    S3 = list(range(2 * nodes_per_stage, total_nodes))
    
    stages = (S1, S2, S3)
    
    # 保存阶段信息
    stage_info = {'stages': stages, 'S1': S1, 'S2': S2, 'S3': S3}
    with open(os.path.join(output_dir, 'stage_info.pkl'), 'wb') as f:
        pickle.dump(stage_info, f)
    
    print(f"\nStage info:")
    print(f"  S1: {len(S1)} nodes ({S1[0]}-{S1[-1]})")
    print(f"  S2: {len(S2)} nodes ({S2[0]}-{S2[-1]})")
    print(f"  S3: {len(S3)} nodes ({S3[0]}-{S3[-1]})")
    
    # 收集训练路径（只包含S1->S2和S2->S3）
    train_paths = []
    
    # 1. S1->S2路径
    s1_s2_count = 0
    for s1 in S1:
        for s2 in S2:
            if G.has_edge(str(s1), str(s2)):
                for _ in range(train_paths_per_pair):
                    train_paths.append([s1, s2])
                    s1_s2_count += 1
    
    # 2. S2->S3路径
    s2_s3_count = 0
    for s2 in S2:
        for s3 in S3:
            if G.has_edge(str(s2), str(s3)):
                for _ in range(train_paths_per_pair):
                    train_paths.append([s2, s3])
                    s2_s3_count += 1
    
    print(f"\nTraining paths collected:")
    print(f"  S1->S2: {s1_s2_count} paths")
    print(f"  S2->S3: {s2_s3_count} paths")
    print(f"  Total: {len(train_paths)} paths")
    
    # 打乱训练数据
    random.shuffle(train_paths)
    
    # 分割训练和验证集
    val_size = int(len(train_paths) * val_ratio)
    val_paths = train_paths[:val_size]
    train_paths = train_paths[val_size:]
    
    # 统计验证集分布
    val_stats = {'S1->S2': 0, 'S2->S3': 0}
    for path in val_paths:
        if path[0] in S1 and path[1] in S2:
            val_stats['S1->S2'] += 1
        elif path[0] in S2 and path[1] in S3:
            val_stats['S2->S3'] += 1
    
    print(f"\nValidation set:")
    print(f"  Total: {len(val_paths)} paths")
    print(f"  S1->S2: {val_stats['S1->S2']}")
    print(f"  S2->S3: {val_stats['S2->S3']}")
    
    # 创建平衡的测试集（包含所有三种类型）
    test_paths = []
    
    # S1->S2测试样本
    s1s2_pairs = [(s1, s2) for s1 in S1 for s2 in S2 if G.has_edge(str(s1), str(s2))]
    random.shuffle(s1s2_pairs)
    for i in range(min(test_samples_per_type, len(s1s2_pairs))):
        s1, s2 = s1s2_pairs[i]
        test_paths.append([s1, s2])
    
    # S2->S3测试样本
    s2s3_pairs = [(s2, s3) for s2 in S2 for s3 in S3 if G.has_edge(str(s2), str(s3))]
    random.shuffle(s2s3_pairs)
    for i in range(min(test_samples_per_type, len(s2s3_pairs))):
        s2, s3 = s2s3_pairs[i]
        test_paths.append([s2, s3])
    
    # S1->S3测试样本（需要找经过S2的路径）
    s1s3_count = 0
    attempts = 0
    max_attempts = test_samples_per_type * 10
    
    while s1s3_count < test_samples_per_type and attempts < max_attempts:
        attempts += 1
        s1 = random.choice(S1)
        s3 = random.choice(S3)
        
        # 尝试找一条S1->S2->S3的路径
        path_found = False
        for s2 in random.sample(S2, min(10, len(S2))):
            if G.has_edge(str(s1), str(s2)) and G.has_edge(str(s2), str(s3)):
                test_paths.append([s1, s2, s3])
                s1s3_count += 1
                path_found = True
                break
    
    # 统计测试集分布
    test_stats = {'S1->S2': 0, 'S2->S3': 0, 'S1->S3': 0}
    for path in test_paths:
        if len(path) == 2:
            if path[0] in S1 and path[1] in S2:
                test_stats['S1->S2'] += 1
            elif path[0] in S2 and path[1] in S3:
                test_stats['S2->S3'] += 1
        elif len(path) >= 3:
            if path[0] in S1 and path[-1] in S3:
                test_stats['S1->S3'] += 1
    
    print(f"\nTest set:")
    print(f"  Total: {len(test_paths)} paths")
    print(f"  S1->S2: {test_stats['S1->S2']}")
    print(f"  S2->S3: {test_stats['S2->S3']}")
    print(f"  S1->S3: {test_stats['S1->S3']}")
    
    # 格式化数据（source target path格式）
    def format_path(path):
        source, target = path[0], path[-1]
        return f"{source} {target} " + " ".join(map(str, path))
    
    # 准备文本数据
    train_text = '\n'.join([format_path(p) for p in train_paths])
    val_text = '\n'.join([format_path(p) for p in val_paths])
    test_text = '\n'.join([format_path(p) for p in test_paths])
    
    # 保存文本文件
    with open(os.path.join(output_dir, f'train_{train_paths_per_pair}.txt'), 'w') as f:
        f.write(train_text)
    
    with open(os.path.join(output_dir, 'val.txt'), 'w') as f:
        f.write(val_text)
    
    with open(os.path.join(output_dir, 'test.txt'), 'w') as f:
        f.write(test_text)
    
    # 准备词汇表
    all_text = train_text + '\n' + val_text + '\n' + test_text
    chars = sorted(list(set(all_text)))
    vocab_size = len(chars) + 2  # +2 for [PAD] and newline
    
    # 创建编码映射
    stoi = {}
    stoi['[PAD]'] = 0
    stoi['\n'] = 1
    for i, ch in enumerate(chars):
        if ch != '\n':  # 已经分配给1了
            stoi[ch] = i + 2
    
    itos = {i: ch for ch, i in stoi.items()}
    
    # 编码函数
    def encode(s):
        return [stoi.get(c, 0) for c in s]
    
    # 确定block_size
    max_path_length = max(
        max(len(p) for p in train_paths),
        max(len(p) for p in val_paths),
        max(len(p) for p in test_paths)
    )
    # source + target + path + spaces + buffer
    block_size = (max_path_length + 2) * 3 + 10
    block_size = ((block_size + 31) // 32) * 32  # 对齐到32
    
    print(f"\nVocabulary size: {vocab_size}")
    print(f"Block size: {block_size}")
    
    # 保存元信息
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
        'block_size': block_size
    }
    
    with open(os.path.join(output_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    
    # 处理为二进制格式
    for split_name, text in [('train', train_text), ('val', val_text)]:
        lines = text.strip().split('\n')
        data = []
        
        for line in lines:
            tokens = encode(line + '\n')
            # Pad到block_size + 1（为了创建输入和目标）
            while len(tokens) < block_size + 1:
                tokens.append(0)  # PAD
            tokens = tokens[:block_size + 1]
            data.extend(tokens)
        
        arr = np.array(data, dtype=np.uint16)
        
        filename = f'{split_name}_{train_paths_per_pair}.bin' if split_name == 'train' else f'{split_name}.bin'
        arr.tofile(os.path.join(output_dir, filename))
        
        print(f"\n{split_name.capitalize()} binary data:")
        print(f"  Total tokens: {len(arr)}")
        print(f"  Sequences: {len(arr) // (block_size + 1)}")
    
    # 保存图
    nx.write_graphml(G, os.path.join(output_dir, 'composition_graph.graphml'))
    
    print(f"\n✅ Dataset preparation complete!")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    prepare_composition_data_fixed()