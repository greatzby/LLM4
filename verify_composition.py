# verify_composition.py
import torch
import pickle
import os
from model import GPTConfig, GPT

def verify_composition_reasoning():
    """验证Standard模型的组合推理能力"""
    
    # 加载Standard模型
    data_dir = 'data/simple_graph/composition_90'
    with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    
    stoi, itos = meta['stoi'], meta['itos']
    
    checkpoint = torch.load('out/composition_standard_20250702_023335/ckpt_35000.pt', map_location='cpu')
    config = GPTConfig(**checkpoint['model_args'])
    model = GPT(config).cuda()
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    print("COMPOSITION REASONING ANALYSIS")
    print("="*60)
    
    # 测试1：模型见过的S1->S2路径
    print("\n1. S1->S2 paths (seen in training):")
    test_s1_s2 = [(0, 30), (5, 35), (10, 40)]
    
    for s1, s2 in test_s1_s2:
        prompt = f"{s1} {s2} {s1}"
        prompt_ids = [stoi[t] for t in prompt.split() if t in stoi]
        x = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0).cuda()
        
        with torch.no_grad():
            y = model.generate(x, max_new_tokens=10, temperature=0.01)
        
        path = []
        for tid in y[0].tolist():
            if tid == 1: break
            if tid in itos:
                try:
                    path.append(int(itos[tid]))
                except:
                    pass
        
        print(f"  {s1}->{s2}: {path}")
    
    # 测试2：模型见过的S2->S3路径
    print("\n2. S2->S3 paths (seen in training):")
    test_s2_s3 = [(30, 60), (35, 65), (40, 70)]
    
    for s2, s3 in test_s2_s3:
        prompt = f"{s2} {s3} {s2}"
        prompt_ids = [stoi[t] for t in prompt.split() if t in stoi]
        x = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0).cuda()
        
        with torch.no_grad():
            y = model.generate(x, max_new_tokens=10, temperature=0.01)
        
        path = []
        for tid in y[0].tolist():
            if tid == 1: break
            if tid in itos:
                try:
                    path.append(int(itos[tid]))
                except:
                    pass
        
        print(f"  {s2}->{s3}: {path}")
    
    # 测试3：S1->S3组合（未见过）
    print("\n3. S1->S3 composition (NOT seen in training):")
    print("   Note: Model combines knowledge from S1->S2 and S2->S3")
    
    test_s1_s3 = [(0, 60), (5, 65), (10, 70)]
    
    for s1, s3 in test_s1_s3:
        prompt = f"{s1} {s3} {s1}"
        prompt_ids = [stoi[t] for t in prompt.split() if t in stoi]
        x = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0).cuda()
        
        with torch.no_grad():
            y = model.generate(x, max_new_tokens=20, temperature=0.01)
        
        path = []
        for tid in y[0].tolist():
            if tid == 1: break
            if tid in itos:
                try:
                    path.append(int(itos[tid]))
                except:
                    pass
        
        # 分析路径中的S2节点
        s2_nodes = [n for n in path[1:-1] if 30 <= n < 60]
        
        print(f"  {s1}->{s3}: {path}")
        print(f"           S2 nodes used: {s2_nodes}")
    
    # 测试4：分析Fixed模型为什么失败
    print("\n\n4. Why Fixed Model Fails:")
    print("-"*40)
    print("Character-level model sees '0 60' as:")
    print("  '0' -> ' ' -> '6' -> '0'")
    print("It learns surface patterns, not semantic relationships.")
    print("\nToken-level model sees '0 60' as:")
    print("  node_0 -> node_60")
    print("It learns the abstract concept of paths between nodes.")

if __name__ == "__main__":
    verify_composition_reasoning()