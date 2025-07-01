# debug_ar_evaluation.py
import torch
import pickle
import os
import numpy as np
from model import GPTConfig, GPT
from collections import defaultdict

def debug_ar_evaluation():
    # 加载checkpoint
    ckpt_path = 'out/composition_standard_20250702_023335/ckpt_35000.pt'
    checkpoint = torch.load(ckpt_path)
    
    # 加载元信息
    data_dir = 'data/simple_graph/composition_90'
    with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    
    stoi, itos = meta['stoi'], meta['itos']
    
    # 加载模型
    model_args = checkpoint['model_args']
    config = GPTConfig(**model_args)
    model = GPT(config).cuda()
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # 读取测试文件
    test_file = os.path.join(data_dir, 'test.txt')
    with open(test_file, 'r') as f:
        test_lines = [line.strip() for line in f.readlines() if line.strip()]
    
    print(f"Total test cases: {len(test_lines)}")
    
    # 测试前几个S1→S2案例
    s1_s2_cases = []
    s2_s3_cases = []
    s1_s3_cases = []
    
    for line in test_lines:
        parts = line.split()
        if len(parts) >= 2:
            source, target = int(parts[0]), int(parts[1])
            if 0 <= source < 30 and 30 <= target < 60:
                s1_s2_cases.append(line)
            elif 30 <= source < 60 and 60 <= target < 90:
                s2_s3_cases.append(line)
            elif 0 <= source < 30 and 60 <= target < 90:
                s1_s3_cases.append(line)
    
    print(f"\nTest distribution:")
    print(f"S1->S2: {len(s1_s2_cases)}")
    print(f"S2->S3: {len(s2_s3_cases)}")
    print(f"S1->S3: {len(s1_s3_cases)}")
    
    # 测试S1->S2生成（应该能工作）
    print("\n=== Testing S1->S2 Generation ===")
    for i, test_line in enumerate(s1_s2_cases[:3]):
        parts = test_line.split()
        source, target = parts[0], parts[1]
        true_path = ' '.join(parts)
        
        prompt = f"{source} {target} {source}"
        print(f"\nTest case {i}:")
        print(f"  True path: {true_path}")
        print(f"  Prompt: {prompt}")
        
        # 编码
        prompt_ids = []
        for token in prompt.split():
            if token in stoi:
                prompt_ids.append(stoi[token])
            else:
                print(f"  ERROR: Token '{token}' not in vocabulary!")
        
        if not prompt_ids:
            print("  ERROR: Failed to encode prompt")
            continue
            
        x = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0).cuda()
        
        # 生成多次尝试
        for attempt in range(3):
            with torch.no_grad():
                y = model.generate(x, max_new_tokens=30, temperature=0.8, top_k=50)
            
            # 解码
            generated_tokens = y[0].tolist()
            decoded = []
            for token in generated_tokens:
                if token in itos:
                    decoded.append(itos[token])
            
            pred_str = ' '.join(decoded)
            print(f"  Attempt {attempt}: {pred_str}")
            
            # 解析路径
            pred_parts = pred_str.split()
            pred_path = []
            for p in pred_parts:
                if p == '\n':
                    break
                if p.isdigit():
                    pred_path.append(int(p))
            
            # 检查正确性
            if len(pred_path) >= 2:
                if pred_path[0] == int(source) and pred_path[-1] == int(target):
                    print(f"  ✓ Success! Path: {pred_path}")
                else:
                    print(f"  ✗ Wrong endpoints: {pred_path[0]} -> {pred_path[-1]}")
            else:
                print(f"  ✗ Path too short: {pred_path}")
    
    # 测试原始evaluate_ar_by_type的逻辑
    print("\n=== Debugging Original Evaluation Logic ===")
    
    # 模拟evaluate_ar_by_type的一个案例
    test_line = s1_s2_cases[0] if s1_s2_cases else s1_s3_cases[0]
    parts = test_line.split()
    source, target = parts[0], parts[1]
    
    print(f"\nSimulating evaluation for: {source} -> {target}")
    
    # Step 1: 编码
    prompt = f"{source} {target} {source}"
    prompt_ids = []
    for token in prompt.split():
        if token in stoi:
            prompt_ids.append(stoi[token])
    
    print(f"Prompt: {prompt}")
    print(f"Encoded IDs: {prompt_ids}")
    
    # Step 2: 生成
    x = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0).cuda()
    with torch.no_grad():
        y = model.generate(x, max_new_tokens=30, temperature=0.8, top_k=50)
    
    print(f"Generated token IDs: {y[0].tolist()[:20]}...")
    
    # Step 3: 解码
    pred_str = ""
    for token in y[0].tolist():
        if token in itos:
            pred_str += itos[token] + " "
    
    print(f"Decoded string: {pred_str.strip()}")
    
    # Step 4: 解析（这可能是问题所在）
    pred_parts = pred_str.split()
    pred_path = []
    for p in pred_parts:
        if p == '\n':
            break
        if p.isdigit():
            pred_path.append(int(p))
    
    print(f"Parsed path: {pred_path}")

if __name__ == "__main__":
    debug_ar_evaluation()