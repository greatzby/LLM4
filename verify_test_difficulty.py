# verify_test_difficulty.py
import torch
import pickle
import os
from model import GPTConfig, GPT

def verify_test_difficulty():
    """检查test.txt中的案例难度"""
    
    # 加载模型和数据
    ckpt_path = 'out/composition_standard_20250702_023335/ckpt_35000.pt'
    checkpoint = torch.load(ckpt_path)
    
    data_dir = 'data/simple_graph/composition_90'
    with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    
    stoi, itos = meta['stoi'], meta['itos']
    
    # 创建模型
    model_args = checkpoint['model_args']
    config = GPTConfig(**model_args)
    model = GPT(config).cuda()
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # 读取test.txt
    test_file = os.path.join(data_dir, 'test.txt')
    with open(test_file, 'r') as f:
        test_lines = [line.strip() for line in f.readlines() if line.strip()]
    
    # 测试前10个S1->S3案例，使用不同温度
    print("Testing first 10 S1->S3 cases from test.txt:\n")
    
    s1s3_count = 0
    for line in test_lines:
        parts = line.split()
        if len(parts) >= 2:
            source, target = int(parts[0]), int(parts[1])
            if 0 <= source < 30 and 60 <= target < 90:
                s1s3_count += 1
                if s1s3_count > 10:
                    break
                
                print(f"Test case: {source} -> {target}")
                print(f"True path: {' '.join(parts)}")
                
                # 测试不同温度
                for temp in [0.1, 0.5, 0.8]:
                    prompt = f"{source} {target} {source}"
                    prompt_ids = []
                    for token in prompt.split():
                        if token in stoi:
                            prompt_ids.append(stoi[token])
                    
                    x = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0).cuda()
                    
                    with torch.no_grad():
                        y = model.generate(x, max_new_tokens=30, temperature=temp, top_k=50)
                    
                    generated = []
                    for token in y[0].tolist():
                        if token == 1:  # newline
                            break
                        if token in itos and token > 1:
                            generated.append(itos[token])
                    
                    pred_path = [int(t) for t in generated if t.isdigit()]
                    
                    # 检查结果
                    success = len(pred_path) >= 2 and pred_path[0] == source and pred_path[-1] == target
                    has_s2 = any(30 <= node < 60 for node in pred_path[1:-1])
                    
                    status = "✓" if success and has_s2 else "✗"
                    print(f"  temp={temp}: {status} {pred_path}")
                
                print()

if __name__ == "__main__":
    verify_test_difficulty()