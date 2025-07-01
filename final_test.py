# final_test.py
import torch
import pickle
import os
from model import GPTConfig, GPT

def test_standard_model():
    """正确测试Standard模型"""
    print("Testing Standard Model (the one that should work)")
    
    # 加载数据
    data_dir = 'data/simple_graph/composition_90'
    with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    
    stoi, itos = meta['stoi'], meta['itos']
    
    # 加载模型
    checkpoint = torch.load('out/composition_standard_20250702_023335/ckpt_35000.pt', map_location='cpu')
    model_args = checkpoint['model_args']
    
    config = GPTConfig(**model_args)
    model = GPT(config).cuda()
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # 测试案例
    test_cases = [(0, 60), (5, 65), (10, 70)]
    
    print("\nDirect S1->S3 generation test:")
    for source, target in test_cases:
        # 构建prompt（关键！）
        prompt = f"{source} {target} {source}"
        prompt_tokens = prompt.split()  # ["0", "60", "0"]
        
        # 编码
        prompt_ids = []
        for token in prompt_tokens:
            if token in stoi:
                prompt_ids.append(stoi[token])
        
        print(f"\nPrompt: {prompt}")
        print(f"Token IDs: {prompt_ids}")
        
        x = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0).cuda()
        
        # 生成
        with torch.no_grad():
            y = model.generate(x, max_new_tokens=20, temperature=0.1, top_k=5)
        
        # 解码（正确方式）
        generated_ids = y[0].tolist()
        decoded_tokens = []
        
        for tid in generated_ids:
            if tid == 1:  # newline
                break
            if tid in itos:
                decoded_tokens.append(itos[tid])
        
        print(f"Decoded tokens: {decoded_tokens}")
        
        # 转换为数字路径（跳过prompt部分）
        path = []
        for i, token in enumerate(decoded_tokens):
            if i >= len(prompt_tokens):  # 跳过prompt
                try:
                    path.append(int(token))
                except:
                    pass
        
        print(f"Generated path: {path}")
        
        # 检查是否成功
        if len(path) >= 2 and path[0] == source and path[-1] == target:
            has_s2 = any(30 <= n < 60 for n in path[1:-1])
            print(f"Success: {'YES' if has_s2 else 'NO (no S2)'}")
        else:
            print("Success: NO")

def test_fixed_model_ar():
    """测试Fixed模型的AR生成（修复版）"""
    print("\n\n" + "="*60)
    print("Testing Fixed Model with corrected AR logic")
    
    # 加载数据
    data_dir = 'data/simple_graph/composition_90/composition_90_fixed'
    with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    
    stoi, itos = meta['stoi'], meta['itos']
    
    # 加载模型
    checkpoint = torch.load('out/composition_fixed_20250702_044341/ckpt_5000.pt', map_location='cpu')
    model_args = checkpoint['model_args']
    
    config = GPTConfig(**model_args)
    model = GPT(config).cuda()
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # 测试
    test_cases = [(0, 60), (5, 65)]
    
    print("\nS1->S3 generation test (character-level):")
    for source, target in test_cases:
        # 构建prompt：只有"source target"
        prompt_str = f"{source} {target}"
        
        # 字符级编码
        prompt_ids = []
        for char in prompt_str:
            if char in stoi:
                prompt_ids.append(stoi[char])
        
        print(f"\nPrompt: '{prompt_str}'")
        print(f"Character encoding: {[(c, stoi[c]) for c in prompt_str if c in stoi]}")
        
        x = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0).cuda()
        
        # 生成
        with torch.no_grad():
            y = model.generate(x, max_new_tokens=30, temperature=0.1, top_k=5)
        
        # 解码整个序列
        full_ids = y[0].tolist()
        chars = []
        for tid in full_ids:
            if tid == 1:  # newline
                break
            if tid in itos:
                chars.append(itos[tid])
        
        full_str = ''.join(chars)
        print(f"Full generated string: '{full_str}'")
        
        # 解析所有数字
        numbers = []
        current = ""
        for char in full_str:
            if char == ' ':
                if current.isdigit():
                    numbers.append(int(current))
                current = ""
            elif char.isdigit():
                current += char
        if current.isdigit():
            numbers.append(int(current))
        
        print(f"Parsed numbers: {numbers}")
        
        # 路径是从第3个数字开始（跳过source, target）
        if len(numbers) >= 3:
            path = numbers[2:]
            print(f"Generated path: {path}")
            
            # 检查
            if path and path[0] == source and path[-1] == target:
                has_s2 = any(30 <= n < 60 for n in path[1:-1])
                print(f"Valid path: {'YES with S2' if has_s2 else 'NO - missing S2'}")
            else:
                print("Valid path: NO")

if __name__ == "__main__":
    test_standard_model()
    test_fixed_model_ar()