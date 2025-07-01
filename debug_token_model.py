# debug_token_model.py
import torch
import pickle
import os
from model import GPTConfig, GPT

def debug_generation():
    """调试Token级模型的生成问题"""
    
    # 加载数据
    data_dir = 'data/simple_graph/composition_90'
    with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    
    stoi, itos = meta['stoi'], meta['itos']
    
    print("="*60)
    print("DEBUG: Token-level Model Generation")
    print("="*60)
    
    # 检查词汇表
    print("\n1. Vocabulary Check:")
    print(f"   Vocab size: {len(stoi)}")
    print(f"   Sample mappings:")
    for i in range(5):
        print(f"     '{i}' -> {stoi.get(str(i), 'NOT FOUND')}")
    print(f"   Special tokens:")
    print(f"     '[PAD]' -> {stoi.get('[PAD]', 'NOT FOUND')}")
    print(f"     '\\n' -> {stoi.get('\\n', 'NOT FOUND' if '\\n' not in stoi else stoi['\\n'])}")
    
    # 检查数据格式
    print("\n2. Data Format Check:")
    test_file = os.path.join(data_dir, 'test.txt')
    with open(test_file, 'r') as f:
        lines = [line.strip() for line in f.readlines()[:5] if line.strip()]
    
    print("   Test data samples:")
    for line in lines:
        print(f"     {line}")
    
    # 加载模型（使用你训练的checkpoint）
    checkpoint_path = 'out/composition_standard_20250702_023335/ckpt_35000.pt'
    if os.path.exists(checkpoint_path):
        print(f"\n3. Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model_args = checkpoint['model_args']
        
        config = GPTConfig(**model_args)
        model = GPT(config).cuda()
        model.load_state_dict(checkpoint['model'])
        model.eval()
        
        # 测试生成
        print("\n4. Testing Generation:")
        source, target = 0, 60
        
        # 方法1：原始prompt格式
        prompt1 = f"{source} {target} {source}"
        tokens1 = prompt1.split()
        print(f"\n   Method 1 - Full prompt: '{prompt1}'")
        print(f"   Tokens: {tokens1}")
        
        ids1 = []
        for t in tokens1:
            if t in stoi:
                ids1.append(stoi[t])
                print(f"     '{t}' -> {stoi[t]}")
            else:
                print(f"     '{t}' -> NOT IN VOCAB!")
        
        if len(ids1) == len(tokens1):
            x = torch.tensor(ids1, dtype=torch.long).unsqueeze(0).cuda()
            
            with torch.no_grad():
                y = model.generate(x, max_new_tokens=20, temperature=0.01)
            
            # 正确的解码
            generated_tokens = []
            full_output = []
            for i, tid in enumerate(y[0].tolist()):
                if tid == 1:  # EOS
                    break
                if tid in itos:
                    token = itos[tid]
                    full_output.append(token)
                    # 只收集生成的部分（跳过prompt）
                    if i >= len(tokens1):
                        generated_tokens.append(token)
            
            print(f"\n   Full output: {full_output}")
            print(f"   Generated only: {generated_tokens}")
            
            # 解析为数字
            numbers = []
            for token in full_output:
                try:
                    numbers.append(int(token))
                except:
                    pass
            
            print(f"   Parsed numbers: {numbers}")
    
    # 建议的修复
    print("\n5. SUGGESTED FIX:")
    print("   The problem is in the decoding logic of train_composition_universal.py")
    print("   Need to properly track prompt length and decode tokens correctly.")

if __name__ == "__main__":
    debug_generation()