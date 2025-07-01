# simple_ar_test.py
import torch
import pickle
import os
import networkx as nx
from model import GPTConfig, GPT

def simple_ar_evaluation():
    # 加载模型和数据
    ckpt_path = 'out/composition_standard_20250702_023335/ckpt_35000.pt'
    checkpoint = torch.load(ckpt_path)
    
    data_dir = 'data/simple_graph/composition_90'
    with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    
    stoi, itos = meta['stoi'], meta['itos']
    
    # 加载图
    G = nx.read_graphml(os.path.join(data_dir, 'composition_graph.graphml'))
    
    # 创建模型
    model_args = checkpoint['model_args']
    config = GPTConfig(**model_args)
    model = GPT(config).cuda()
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # 读取测试数据
    test_file = os.path.join(data_dir, 'test.txt')
    with open(test_file, 'r') as f:
        test_lines = [line.strip() for line in f.readlines() if line.strip()]
    
    # 简单的评估
    results = {
        'S1->S2': {'correct': 0, 'total': 0},
        'S2->S3': {'correct': 0, 'total': 0},
        'S1->S3': {'correct': 0, 'total': 0}
    }
    
    for line in test_lines[:100]:  # 测试前100个
        parts = line.split()
        if len(parts) < 2:
            continue
            
        source, target = int(parts[0]), int(parts[1])
        
        # 判断类型
        if 0 <= source < 30 and 30 <= target < 60:
            path_type = 'S1->S2'
        elif 30 <= source < 60 and 60 <= target < 90:
            path_type = 'S2->S3'
        elif 0 <= source < 30 and 60 <= target < 90:
            path_type = 'S1->S3'
        else:
            continue
        
        results[path_type]['total'] += 1
        
        # 生成预测
        prompt = f"{source} {target} {source}"
        
        # 手动编码（避免encode函数的问题）
        prompt_ids = []
        for token in prompt.split():
            if token in stoi:
                prompt_ids.append(stoi[token])
        
        if not prompt_ids:
            continue
            
        x = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0).cuda()
        
        # 生成
        with torch.no_grad():
            y = model.generate(x, max_new_tokens=30, temperature=0.1, top_k=5)  # 更低的温度
        
        # 手动解码
        generated = []
        for token in y[0].tolist():
            if token == 1:  # newline
                break
            if token in itos and token > 1:  # 忽略PAD
                generated.append(itos[token])
        
        # 解析路径
        pred_path = []
        for t in generated:
            if t.isdigit():
                pred_path.append(int(t))
        
        # 检查正确性
        if len(pred_path) >= 2:
            if pred_path[0] == source and pred_path[-1] == target:
                # 检查路径有效性
                valid = True
                for i in range(len(pred_path) - 1):
                    if not G.has_edge(str(pred_path[i]), str(pred_path[i+1])):
                        valid = False
                        break
                
                if valid:
                    results[path_type]['correct'] += 1
                    if results[path_type]['correct'] <= 3:  # 打印前几个成功案例
                        print(f"✓ {path_type}: {source}->{target}, path: {pred_path}")
    
    # 打印结果
    print("\n=== Simple AR Evaluation Results ===")
    for path_type, stats in results.items():
        if stats['total'] > 0:
            accuracy = stats['correct'] / stats['total'] * 100
            print(f"{path_type}: {stats['correct']}/{stats['total']} = {accuracy:.1f}%")
        else:
            print(f"{path_type}: 0/0")

if __name__ == "__main__":
    simple_ar_evaluation()