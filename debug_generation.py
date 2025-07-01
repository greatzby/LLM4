# debug_generation.py
import torch
import pickle
import os
import numpy as np
from model import GPTConfig, GPT

def debug_generation():
    # 加载模型和数据
    data_dir = 'data/simple_graph/composition_90/composition_90_fixed'
    checkpoint_path = 'out/composition_fixed_20250702_042854/ckpt_10000.pt'  # 使用你的checkpoint路径
    
    # 加载元信息
    with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    
    stoi = meta['stoi']
    itos = meta['itos']
    block_size = meta['block_size']
    vocab_size = meta['vocab_size']
    
    print("Vocabulary mapping:")
    for k, v in sorted(stoi.items(), key=lambda x: x[1]):
        print(f"  '{k}' -> {v}")
    
    # 加载模型
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_args = checkpoint['model_args']
    
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model = model.to('cuda:0')
    
    # 测试几个例子
    test_prompts = [
        "0 30 0",   # S1->S2
        "30 60 30", # S2->S3
        "0 60 0"    # S1->S3
    ]
    
    print("\nTesting generation:")
    for prompt_str in test_prompts:
        print(f"\n--- Prompt: '{prompt_str}' ---")
        
        # 编码
        prompt_ids = []
        for char in prompt_str:
            if char in stoi:
                prompt_ids.append(stoi[char])
                print(f"  '{char}' -> {stoi[char]}")
        
        print(f"  Encoded: {prompt_ids}")
        
        # 生成
        x = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0).to('cuda:0')
        
        with torch.no_grad():
            # 生成更多tokens看看
            y = model.generate(x, max_new_tokens=50, temperature=0.1, top_k=10)
        
        # 解码全部生成的内容
        full_generated = y[0].tolist()
        print(f"  Full generated tokens: {full_generated[:20]}...")  # 前20个
        
        # 解码生成的部分（跳过prompt）
        generated_tokens = full_generated[len(prompt_ids):]
        print(f"  Generated tokens (after prompt): {generated_tokens[:20]}...")
        
        # 转换为字符
        generated_chars = []
        for tid in generated_tokens:
            if tid == 1:  # newline
                generated_chars.append('\\n')
                break
            if tid in itos:
                generated_chars.append(itos[tid])
        
        print(f"  Generated chars: {''.join(generated_chars[:30])}")
        
        # 解析数字
        generated_str = ''.join([c for c in generated_chars if c != '\\n'])
        numbers = []
        current = ""
        
        for char in generated_str:
            if char == ' ':
                if current and current.isdigit():
                    numbers.append(int(current))
                current = ""
            elif char.isdigit():
                current += char
        
        if current and current.isdigit():
            numbers.append(int(current))
        
        print(f"  Parsed numbers: {numbers}")
        print(f"  Length: {len(numbers)}")

if __name__ == "__main__":
    debug_generation()