# quick_eval.py - 快速评估模型的真实能力
import torch
import pickle
import os
import random
from model import GPTConfig, GPT

def quick_comprehensive_eval():
    """全面评估模型在各种路径上的能力"""
    
    # 加载模型
    ckpt_path = 'out/composition_standard_20250702_023335/ckpt_35000.pt'
    checkpoint = torch.load(ckpt_path)
    
    data_dir = 'data/simple_graph/composition_90'
    with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    
    with open(os.path.join(data_dir, 'stage_info.pkl'), 'rb') as f:
        stage_info = pickle.load(f)
    
    stoi, itos = meta['stoi'], meta['itos']
    S1, S2, S3 = stage_info['stages']
    
    # 创建模型
    model_args = checkpoint['model_args']
    config = GPTConfig(**model_args)
    model = GPT(config).cuda()
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # 测试各种路径
    results = {
        'S1->S2': {'correct': 0, 'total': 0},
        'S2->S3': {'correct': 0, 'total': 0},
        'S1->S3': {'correct': 0, 'total': 0}
    }
    
    # 每种类型测试20个随机样本
    n_samples = 20
    
    print("=== Comprehensive Model Evaluation ===\n")
    
    # 1. 测试S1->S2
    print("Testing S1->S2:")
    for _ in range(n_samples):
        s1 = random.choice(S1)
        s2 = random.choice(S2)
        
        prompt = f"{s1} {s2} {s1}"
        prompt_ids = []
        for token in prompt.split():
            if token in stoi:
                prompt_ids.append(stoi[token])
        
        if prompt_ids:
            x = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0).cuda()
            
            with torch.no_grad():
                y = model.generate(x, max_new_tokens=10, temperature=0.1, top_k=5)
            
            generated = []
            for token in y[0].tolist():
                if token == 1:  # newline
                    break
                if token in itos and token > 1:
                    generated.append(itos[token])
            
            pred_path = [int(t) for t in generated if t.isdigit()]
            
            results['S1->S2']['total'] += 1
            if len(pred_path) >= 2 and pred_path[0] == s1 and pred_path[-1] == s2:
                results['S1->S2']['correct'] += 1
                if results['S1->S2']['correct'] <= 3:
                    print(f"  ✓ {s1}->{s2}: {pred_path}")
    
    # 2. 测试S2->S3
    print("\nTesting S2->S3:")
    for _ in range(n_samples):
        s2 = random.choice(S2)
        s3 = random.choice(S3)
        
        prompt = f"{s2} {s3} {s2}"
        prompt_ids = []
        for token in prompt.split():
            if token in stoi:
                prompt_ids.append(stoi[token])
        
        if prompt_ids:
            x = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0).cuda()
            
            with torch.no_grad():
                y = model.generate(x, max_new_tokens=10, temperature=0.1, top_k=5)
            
            generated = []
            for token in y[0].tolist():
                if token == 1:  # newline
                    break
                if token in itos and token > 1:
                    generated.append(itos[token])
            
            pred_path = [int(t) for t in generated if t.isdigit()]
            
            results['S2->S3']['total'] += 1
            if len(pred_path) >= 2 and pred_path[0] == s2 and pred_path[-1] == s3:
                results['S2->S3']['correct'] += 1
                if results['S2->S3']['correct'] <= 3:
                    print(f"  ✓ {s2}->{s3}: {pred_path}")
    
    # 3. 测试S1->S3
    print("\nTesting S1->S3:")
    for i in range(n_samples):
        s1 = random.choice(S1)
        s3 = random.choice(S3)
        
        prompt = f"{s1} {s3} {s1}"
        prompt_ids = []
        for token in prompt.split():
            if token in stoi:
                prompt_ids.append(stoi[token])
        
        if prompt_ids:
            x = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0).cuda()
            
            with torch.no_grad():
                y = model.generate(x, max_new_tokens=20, temperature=0.1, top_k=5)
            
            generated = []
            for token in y[0].tolist():
                if token == 1:  # newline
                    break
                if token in itos and token > 1:
                    generated.append(itos[token])
            
            pred_path = [int(t) for t in generated if t.isdigit()]
            
            results['S1->S3']['total'] += 1
            if len(pred_path) >= 2 and pred_path[0] == s1 and pred_path[-1] == s3:
                # 检查是否经过S2
                has_s2 = any(node in S2 for node in pred_path[1:-1])
                if has_s2:
                    results['S1->S3']['correct'] += 1
                    if results['S1->S3']['correct'] <= 3:
                        print(f"  ✓ {s1}->{s3}: {pred_path}")
                elif i < 5:  # 打印前几个失败案例
                    print(f"  ✗ {s1}->{s3}: {pred_path} (no S2)")
    
    # 打印结果
    print("\n=== Results ===")
    for path_type, stats in results.items():
        if stats['total'] > 0:
            acc = stats['correct'] / stats['total'] * 100
            print(f"{path_type}: {stats['correct']}/{stats['total']} = {acc:.1f}%")

if __name__ == "__main__":
    quick_comprehensive_eval()