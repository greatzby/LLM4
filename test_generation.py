# test_generation.py
import torch
import pickle
import os
from model import GPTConfig, GPT

def test_model_generation():
    # 加载最新的checkpoint
    ckpt_path = 'out/composition_standard_20250702_023335/ckpt_5000.pt'
    
    if not os.path.exists(ckpt_path):
        print("Checkpoint not found!")
        return
    
    # 加载模型
    checkpoint = torch.load(ckpt_path)
    model_args = checkpoint['model_args']
    
    # 加载元信息
    data_dir = 'data/simple_graph/composition_90'
    with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    
    stoi, itos = meta['stoi'], meta['itos']
    
    # 创建模型
    config = GPTConfig(**model_args)
    model = GPT(config).cuda()
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    print("Testing trained model generation...")
    
    # 测试不同类型的路径
    test_prompts = [
        ("0", "30"),   # S1->S2
        ("30", "60"),  # S2->S3
        ("0", "60"),   # S1->S3
        ("5", "35"),   # S1->S2
        ("35", "65"),  # S2->S3
    ]
    
    for source, target in test_prompts:
        prompt = f"{source} {target} {source}"
        
        # 编码
        tokens = []
        for t in prompt.split():
            if t in stoi:
                tokens.append(stoi[t])
        
        if not tokens:
            print(f"\nFailed to encode: {prompt}")
            continue
        
        x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).cuda()
        
        # 生成
        with torch.no_grad():
            y = model.generate(x, max_new_tokens=20, temperature=0.8, top_k=50)
        
        # 解码
        decoded = []
        for token in y[0].tolist():
            if token in itos:
                decoded.append(itos[token])
                if itos[token] == '\n':
                    break
        
        result = ' '.join(decoded)
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {result}")
        
        # 检查路径
        path_tokens = result.split()
        if '\n' in path_tokens:
            path_tokens = path_tokens[:path_tokens.index('\n')]
        
        path_numbers = []
        for t in path_tokens:
            if t.isdigit():
                path_numbers.append(int(t))
        
        if len(path_numbers) >= 2:
            if path_numbers[0] == int(source) and path_numbers[-1] == int(target):
                print("✓ Valid path!")
            else:
                print(f"✗ Invalid: source={path_numbers[0]}, target={path_numbers[-1]}")
        else:
            print("✗ Path too short")

if __name__ == "__main__":
    test_model_generation()