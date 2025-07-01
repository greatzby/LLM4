# unified_test.py
import torch
import pickle
import os
from model import GPTConfig, GPT

def test_model(checkpoint_path, data_dir, model_name):
    """测试单个模型的组合能力"""
    print(f"\n{'='*60}")
    print(f"Testing {model_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Data: {data_dir}")
    
    try:
        # 加载数据
        with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
        
        stoi, itos = meta['stoi'], meta['itos']
        vocab_size = meta['vocab_size']
        
        print(f"Vocab size: {vocab_size}")
        print(f"Sample tokens: {list(stoi.items())[:5]}")
        
        # 加载模型
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model_args = checkpoint['model_args']
        
        config = GPTConfig(**model_args)
        model = GPT(config).cuda()
        model.load_state_dict(checkpoint['model'])
        model.eval()
        
        # 测试S1->S3（固定测试案例）
        test_cases = [
            (0, 60),
            (5, 65),
            (10, 70),
            (15, 75),
            (20, 80)
        ]
        
        correct = 0
        print("\nS1->S3 Test Results:")
        
        for source, target in test_cases:
            # 构建prompt - 关键是这里！
            if vocab_size < 100:  # 字符级tokenization
                prompt_str = f"{source} {target} {source}"
                prompt_ids = []
                for char in prompt_str:
                    if char in stoi:
                        prompt_ids.append(stoi[char])
            else:  # 数字级tokenization
                prompt_tokens = [str(source), str(target), str(source)]
                prompt_ids = []
                for token in prompt_tokens:
                    if token in stoi:
                        prompt_ids.append(stoi[token])
            
            x = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0).cuda()
            
            # 生成
            with torch.no_grad():
                y = model.generate(x, max_new_tokens=30, temperature=0.1, top_k=10)
            
            # 解码
            if vocab_size < 100:  # 字符级
                # 解码整个序列
                chars = []
                for tid in y[0].tolist():
                    if tid == 1: break
                    if tid in itos and tid > 1:
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
                
                if len(numbers) >= 3:
                    path = numbers[2:]  # 跳过prompt中的source target
                else:
                    path = []
                    
            else:  # 数字级
                generated = []
                for tid in y[0].tolist():
                    if tid == 1: break
                    if tid in itos and tid > 1:
                        token = itos[tid]
                        if token.isdigit():
                            generated.append(int(token))
                path = generated
            
            # 检查结果
            success = False
            if len(path) >= 2 and path[0] == source and path[-1] == target:
                # 检查是否经过S2 (30-59)
                has_s2 = any(30 <= n < 60 for n in path[1:-1])
                if has_s2:
                    success = True
                    correct += 1
            
            print(f"  {source}->{target}: {'✓' if success else '✗'} {path}")
        
        accuracy = correct / len(test_cases) * 100
        print(f"\nAccuracy: {correct}/{len(test_cases)} = {accuracy:.1f}%")
        
        return accuracy
        
    except Exception as e:
        print(f"Error testing model: {e}")
        return 0

# 测试两个模型
if __name__ == "__main__":
    print("UNIFIED MODEL COMPARISON TEST")
    print("="*80)
    
    # 模型1：失败的
    acc1 = test_model(
        'out/composition_fixed_20250702_044341/ckpt_5000.pt',
        'data/simple_graph/composition_90/composition_90_fixed',
        'Model 1 (Fixed)'
    )
    
    # 模型2：成功的
    acc2 = test_model(
        'out/composition_standard_20250702_023335/ckpt_35000.pt',
        'data/simple_graph/composition_90',
        'Model 2 (Standard)'
    )
    
    print(f"\n{'='*60}")
    print("FINAL CONCLUSION:")
    print(f"Model 1: {acc1:.1f}% - {'HAS' if acc1 > 50 else 'NO'} composition ability")
    print(f"Model 2: {acc2:.1f}% - {'HAS' if acc2 > 50 else 'NO'} composition ability")
    print("="*60)