# fixed_evaluation_final.py
import torch
import numpy as np
from collections import defaultdict

@torch.no_grad()
def evaluate_ar_fixed(model, test_file, stages, stoi, itos, device, G, max_eval_per_type=50):
    """修复版的AR评估函数"""
    model.eval()
    
    S1, S2, S3 = stages
    
    # 读取测试数据
    with open(test_file, 'r') as f:
        test_lines = [line.strip() for line in f.readlines() if line.strip()]
    
    # 分类测试数据
    test_by_type = {
        'S1->S2': [],
        'S2->S3': [],
        'S1->S3': []
    }
    
    for line in test_lines:
        parts = line.split()
        if len(parts) >= 2:
            try:
                source, target = int(parts[0]), int(parts[1])
                if source in S1 and target in S2:
                    test_by_type['S1->S2'].append((source, target, parts))
                elif source in S2 and target in S3:
                    test_by_type['S2->S3'].append((source, target, parts))
                elif source in S1 and target in S3:
                    test_by_type['S1->S3'].append((source, target, parts))
            except:
                continue
    
    # 结果统计
    results = {}
    
    for path_type, test_cases in test_by_type.items():
        n_eval = min(len(test_cases), max_eval_per_type)
        results[path_type] = {
            'correct': 0,
            'total': n_eval,
            'errors': defaultdict(int)
        }
        
        # 评估
        for idx in range(n_eval):
            source, target, true_path_parts = test_cases[idx]
            
            # 生成prompt
            prompt = f"{source} {target} {source}"
            
            # 编码（直接构建，避免潜在bug）
            prompt_ids = []
            for token in [str(source), str(target), str(source)]:
                if token in stoi:
                    prompt_ids.append(stoi[token])
            
            if len(prompt_ids) != 3:
                results[path_type]['errors']['encoding_error'] += 1
                continue
            
            x = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
            
            # 生成（使用低温度）
            with torch.no_grad():
                y = model.generate(x, max_new_tokens=30, temperature=0.1, top_k=10)
            
            # 解码
            generated = []
            for token_id in y[0].tolist():
                if token_id == 1:  # newline
                    break
                if token_id in itos and token_id > 1:  # 跳过PAD
                    token_str = itos[token_id]
                    if token_str.isdigit():
                        generated.append(int(token_str))
            
            # 验证
            if len(generated) >= 2:
                if generated[0] == source and generated[-1] == target:
                    # 对S1->S3，检查是否经过S2
                    if path_type == 'S1->S3':
                        has_s2 = any(node in S2 for node in generated[1:-1])
                        if has_s2:
                            results[path_type]['correct'] += 1
                        else:
                            results[path_type]['errors']['no_s2'] += 1
                    else:
                        # S1->S2和S2->S3只需要端点正确
                        results[path_type]['correct'] += 1
                else:
                    results[path_type]['errors']['wrong_endpoints'] += 1
            else:
                results[path_type]['errors']['too_short'] += 1
        
        # 计算准确率
        results[path_type]['accuracy'] = results[path_type]['correct'] / results[path_type]['total'] if results[path_type]['total'] > 0 else 0.0
    
    model.train()
    return results

# 测试修复的函数
if __name__ == "__main__":
    import pickle
    import os
    import networkx as nx
    from model import GPTConfig, GPT
    
    # 加载模型
    ckpt_path = 'out/composition_standard_20250702_023335/ckpt_35000.pt'
    checkpoint = torch.load(ckpt_path)
    
    data_dir = 'data/simple_graph/composition_90'
    with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    
    with open(os.path.join(data_dir, 'stage_info.pkl'), 'rb') as f:
        stage_info = pickle.load(f)
    
    stoi, itos = meta['stoi'], meta['itos']
    stages = stage_info['stages']
    
    G = nx.read_graphml(os.path.join(data_dir, 'composition_graph.graphml'))
    
    # 创建模型
    model_args = checkpoint['model_args']
    config = GPTConfig(**model_args)
    model = GPT(config).cuda()
    model.load_state_dict(checkpoint['model'])
    
    # 测试
    test_file = os.path.join(data_dir, 'test.txt')
    results = evaluate_ar_fixed(model, test_file, stages, stoi, itos, 'cuda', G)
    
    print("\n=== Fixed AR Evaluation Results ===")
    for path_type, stats in results.items():
        print(f"\n{path_type}:")
        print(f"  Accuracy: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
        if stats['errors']:
            print(f"  Errors:")
            for err_type, count in stats['errors'].items():
                print(f"    {err_type}: {count}")