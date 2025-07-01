# debug_eval_function.py
import torch
import pickle
import os
import networkx as nx
from model import GPTConfig, GPT

def debug_eval_function():
    """调试为什么evaluate_ar_by_type显示0%"""
    
    # 加载必要的数据
    ckpt_path = 'out/composition_standard_20250702_023335/ckpt_35000.pt'
    checkpoint = torch.load(ckpt_path)
    
    data_dir = 'data/simple_graph/composition_90'
    with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    
    with open(os.path.join(data_dir, 'stage_info.pkl'), 'rb') as f:
        stage_info = pickle.load(f)
    
    stoi, itos = meta['stoi'], meta['itos']
    S1, S2, S3 = stage_info['stages']
    
    # 加载图
    G = nx.read_graphml(os.path.join(data_dir, 'composition_graph.graphml'))
    
    # 创建模型
    model_args = checkpoint['model_args']
    config = GPTConfig(**model_args)
    model = GPT(config).cuda()
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # 模拟evaluate_ar_by_type的逻辑
    test_file = os.path.join(data_dir, 'test.txt')
    with open(test_file, 'r') as f:
        test_lines = [line.strip() for line in f.readlines() if line.strip()]
    
    # 测试第一个S1->S3案例
    for line in test_lines[:5]:
        parts = line.split()
        if len(parts) >= 2:
            source, target = int(parts[0]), int(parts[1])
            if 0 <= source < 30 and 60 <= target < 90:
                print(f"\nDebugging case: {source} -> {target}")
                print(f"True path: {parts}")
                
                # Step 1: 准备prompt（原始评估函数的方式）
                prompt = f"{source} {target} {source}"
                print(f"Prompt string: '{prompt}'")
                
                # Step 2: 使用原始encode函数
                def encode(s, stoi):
                    tokens = []
                    parts = s.strip().split()
                    for ch in parts:
                        if ch in stoi:
                            tokens.append(stoi[ch])
                    return tokens
                
                prompt_ids = encode(prompt, stoi)
                print(f"Encoded IDs: {prompt_ids}")
                
                # 手动检查编码
                manual_ids = []
                for token in prompt.split():
                    if token in stoi:
                        manual_ids.append(stoi[token])
                        print(f"  '{token}' -> {stoi[token]}")
                    else:
                        print(f"  '{token}' -> NOT IN VOCAB!")
                
                if not prompt_ids:
                    print("ERROR: Empty encoding!")
                    continue
                
                # Step 3: 生成
                x = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0).cuda()
                
                with torch.no_grad():
                    y = model.generate(x, max_new_tokens=30, temperature=0.8, top_k=50)
                
                print(f"Generated token IDs: {y[0].tolist()[:20]}...")
                
                # Step 4: 解码
                generated_tokens = y[0].tolist()
                pred_parts = []
                for token in generated_tokens:
                    if token == 1:  # newline
                        break
                    if token in itos and token > 1:  # 忽略PAD
                        pred_parts.append(itos[token])
                
                print(f"Decoded parts: {pred_parts}")
                
                # Step 5: 提取数字路径
                pred_path = []
                for p in pred_parts:
                    if p.isdigit():
                        pred_path.append(int(p))
                
                print(f"Extracted path: {pred_path}")
                
                # Step 6: 检查有效性
                if len(pred_path) >= 2:
                    print(f"Path endpoints: {pred_path[0]} -> {pred_path[-1]}")
                    print(f"Expected: {source} -> {target}")
                    
                    if pred_path[0] == source and pred_path[-1] == target:
                        # 检查路径有效性
                        valid = True
                        for i in range(len(pred_path) - 1):
                            if not G.has_edge(str(pred_path[i]), str(pred_path[i+1])):
                                print(f"  Invalid edge: {pred_path[i]} -> {pred_path[i+1]}")
                                valid = False
                                break
                        
                        if valid:
                            print("✓ Valid path!")
                        else:
                            print("✗ Invalid path (edge check failed)")
                    else:
                        print("✗ Wrong endpoints")
                else:
                    print("✗ Path too short")
                
                break

if __name__ == "__main__":
    debug_eval_function()