# debug_generation_difference.py
import torch
import pickle
import os
from model import GPTConfig, GPT

def debug_both_models():
    """深入对比两个模型的生成过程"""
    
    # 测试案例
    source, target = 0, 60
    
    print("="*80)
    print("DEBUGGING GENERATION DIFFERENCES")
    print("="*80)
    
    # 1. Standard Model
    print("\n1. STANDARD MODEL (Token-level)")
    print("-"*40)
    
    data_dir = 'data/simple_graph/composition_90'
    with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    
    stoi, itos = meta['stoi'], meta['itos']
    
    checkpoint = torch.load('out/composition_standard_20250702_023335/ckpt_35000.pt', map_location='cpu')
    config = GPTConfig(**checkpoint['model_args'])
    model = GPT(config).cuda()
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # 构建prompt
    prompt = f"{source} {target} {source}"
    prompt_tokens = prompt.split()
    prompt_ids = [stoi[t] for t in prompt_tokens if t in stoi]
    
    print(f"Prompt: '{prompt}'")
    print(f"Prompt tokens: {prompt_tokens}")
    print(f"Prompt IDs: {prompt_ids}")
    
    x = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0).cuda()
    
    # 逐步生成，看每一步
    print("\nStep-by-step generation:")
    current_ids = prompt_ids.copy()
    
    for step in range(10):
        x_current = torch.tensor(current_ids, dtype=torch.long).unsqueeze(0).cuda()
        
        with torch.no_grad():
            logits, _ = model(x_current)
            # 获取最后一个位置的预测
            next_logits = logits[0, -1, :]
            # 使用greedy decoding
            next_id = torch.argmax(next_logits).item()
        
        if next_id == 1:  # EOS
            print(f"  Step {step}: Generated EOS")
            break
            
        next_token = itos[next_id] if next_id in itos else '?'
        current_ids.append(next_id)
        
        # 解码当前序列
        current_tokens = [itos[tid] if tid in itos else '?' for tid in current_ids]
        print(f"  Step {step}: Next token='{next_token}' (id={next_id}), Sequence={current_tokens}")
    
    # 2. Fixed Model
    print("\n\n2. FIXED MODEL (Character-level)")
    print("-"*40)
    
    data_dir = 'data/simple_graph/composition_90/composition_90_fixed'
    with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    
    stoi, itos = meta['stoi'], meta['itos']
    
    checkpoint = torch.load('out/composition_fixed_20250702_044341/ckpt_5000.pt', map_location='cpu')
    config = GPTConfig(**checkpoint['model_args'])
    model = GPT(config).cuda()
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # 构建prompt（字符级）
    prompt_str = f"{source} {target}"
    prompt_ids = [stoi[c] for c in prompt_str if c in stoi]
    
    print(f"Prompt: '{prompt_str}'")
    print(f"Prompt chars: {list(prompt_str)}")
    print(f"Prompt IDs: {prompt_ids}")
    print(f"Char mapping: {[(c, stoi[c]) for c in prompt_str if c in stoi]}")
    
    # 逐步生成
    print("\nStep-by-step generation:")
    current_ids = prompt_ids.copy()
    
    for step in range(15):
        x_current = torch.tensor(current_ids, dtype=torch.long).unsqueeze(0).cuda()
        
        with torch.no_grad():
            logits, _ = model(x_current)
            next_logits = logits[0, -1, :]
            next_id = torch.argmax(next_logits).item()
        
        if next_id == 1:  # newline
            print(f"  Step {step}: Generated newline")
            break
            
        next_char = itos[next_id] if next_id in itos else '?'
        current_ids.append(next_id)
        
        # 解码当前序列
        current_chars = [itos[tid] if tid in itos else '?' for tid in current_ids]
        current_string = ''.join(current_chars)
        print(f"  Step {step}: Next char='{next_char}' (id={next_id}), String='{current_string}'")
    
    # 3. 分析训练数据
    print("\n\n3. TRAINING DATA ANALYSIS")
    print("-"*40)
    
    # 检查Fixed模型的训练数据中S1->S3的例子
    print("Checking if Fixed model saw S1->S3 examples in training:")
    with open('data/simple_graph/composition_90/composition_90_fixed/train_10.txt', 'r') as f:
        s1_s3_count = 0
        for i, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) >= 2:
                src, tgt = int(parts[0]), int(parts[1])
                if src < 30 and tgt >= 60:
                    s1_s3_count += 1
                    if s1_s3_count <= 3:
                        print(f"  Found S1->S3: {line.strip()}")
            if i > 1000:
                break
        
        print(f"  Total S1->S3 examples found in first 1000 lines: {s1_s3_count}")

if __name__ == "__main__":
    debug_both_models()