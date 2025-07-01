# correct_final_test.py
import torch
import pickle
import os
from model import GPTConfig, GPT

def test_model_correctly(checkpoint_path, data_dir, model_name):
    """正确测试模型的组合能力"""
    print(f"\n{'='*60}")
    print(f"Testing {model_name}")
    
    # 加载数据
    with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    
    stoi, itos = meta['stoi'], meta['itos']
    vocab_size = meta['vocab_size']
    
    # 加载模型
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_args = checkpoint['model_args']
    
    config = GPTConfig(**model_args)
    model = GPT(config).cuda()
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # 测试案例
    test_cases = [(0, 60), (5, 65), (10, 70), (15, 75), (20, 80)]
    success_count = 0
    
    print("\nS1->S3 Generation Results:")
    
    for source, target in test_cases:
        if vocab_size > 50:  # Standard model (token-level)
            # 构建prompt
            prompt = f"{source} {target} {source}"
            prompt_tokens = prompt.split()
            
            # 编码
            prompt_ids = []
            for token in prompt_tokens:
                if token in stoi:
                    prompt_ids.append(stoi[token])
            
            x = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0).cuda()
            
            # 生成
            with torch.no_grad():
                y = model.generate(x, max_new_tokens=20, temperature=0.1, top_k=5)
            
            # 解码完整序列
            generated_ids = y[0].tolist()
            full_sequence = []
            
            for tid in generated_ids:
                if tid == 1:  # newline
                    break
                if tid in itos:
                    try:
                        full_sequence.append(int(itos[tid]))
                    except:
                        pass
            
            # 完整路径就是整个序列！
            path = full_sequence
            
        else:  # Fixed model (character-level)
            # 构建prompt
            prompt_str = f"{source} {target}"
            
            # 字符级编码
            prompt_ids = []
            for char in prompt_str:
                if char in stoi:
                    prompt_ids.append(stoi[char])
            
            x = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0).cuda()
            
            # 生成
            with torch.no_grad():
                y = model.generate(x, max_new_tokens=30, temperature=0.1, top_k=5)
            
            # 解码并解析
            full_ids = y[0].tolist()
            chars = []
            for tid in full_ids:
                if tid == 1:  # newline
                    break
                if tid in itos:
                    chars.append(itos[tid])
            
            # 解析数字
            full_str = ''.join(chars)
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
            
            # 路径从第3个数字开始
            if len(numbers) >= 3:
                path = numbers[2:]
            else:
                path = []
        
        # 评估路径
        success = False
        s2_nodes = []
        
        if len(path) >= 2 and path[0] == source and path[-1] == target:
            # 找出路径中的S2节点
            s2_nodes = [n for n in path[1:-1] if 30 <= n < 60]
            if s2_nodes:
                success = True
                success_count += 1
        
        # 打印结果
        status = "✓" if success else "✗"
        print(f"  {status} {source}->{target}: {path}")
        if s2_nodes:
            print(f"     S2 nodes: {s2_nodes}")
    
    accuracy = success_count / len(test_cases) * 100
    print(f"\nAccuracy: {success_count}/{len(test_cases)} = {accuracy:.1f}%")
    
    return accuracy

# 主测试
if __name__ == "__main__":
    print("CORRECTED MODEL EVALUATION")
    
    # Standard模型
    acc1 = test_model_correctly(
        'out/composition_standard_20250702_023335/ckpt_35000.pt',
        'data/simple_graph/composition_90',
        'Standard Model (Token-level)'
    )
    
    # Fixed模型
    acc2 = test_model_correctly(
        'out/composition_fixed_20250702_044341/ckpt_5000.pt',
        'data/simple_graph/composition_90/composition_90_fixed',
        'Fixed Model (Character-level)'
    )
    
    print("\n" + "="*60)
    print("FINAL ANSWER:")
    print(f"• Standard Model: {acc1:.1f}% - {'✓ HAS' if acc1 > 50 else '✗ NO'} composition ability")
    print(f"• Fixed Model: {acc2:.1f}% - {'✓ HAS' if acc2 > 50 else '✗ NO'} composition ability")
    print("="*60)