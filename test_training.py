# test_training.py - 简单测试训练是否正常
import torch
import numpy as np
import pickle
import os
from model import GPTConfig, GPT

def test_basic_training():
    """测试基本的训练功能"""
    # 加载数据
    data_dir = 'data/simple_graph/composition_90'
    
    with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    
    stoi, itos = meta['stoi'], meta['itos']
    block_size = meta['block_size']
    vocab_size = meta['vocab_size']
    
    # 创建小模型
    config = GPTConfig(
        n_layer=1,
        n_head=1,
        n_embd=120,
        block_size=block_size,
        vocab_size=vocab_size,
        dropout=0.0,
        bias=False
    )
    
    model = GPT(config).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # 创建简单的训练数据
    # 格式：source target source ... target \n
    train_samples = [
        "0 30 0 22 30\n",
        "30 60 30 45 60\n",
    ]
    
    print("Training on simple examples...")
    model.train()
    
    for epoch in range(100):
        total_loss = 0
        
        for sample in train_samples:
            # 编码
            tokens = []
            for token in sample.strip().split():
                if token in stoi:
                    tokens.append(stoi[token])
                else:
                    tokens.append(stoi['\n'])
            
            # Pad到block_size
            while len(tokens) < block_size + 1:
                tokens.append(0)  # PAD
            
            tokens = tokens[:block_size + 1]
            
            # 准备输入
            x = torch.tensor(tokens[:-1], dtype=torch.long).unsqueeze(0).cuda()
            y = torch.tensor(tokens[1:], dtype=torch.long).unsqueeze(0).cuda()
            
            # 前向传播
            logits, loss = model(x, y)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {total_loss/len(train_samples):.4f}")
    
    # 测试生成
    print("\nTesting generation...")
    model.eval()
    
    test_prompts = [
        "0 30 0",
        "30 60 30",
    ]
    
    for prompt in test_prompts:
        tokens = []
        for token in prompt.split():
            if token in stoi:
                tokens.append(stoi[token])
        
        x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).cuda()
        
        with torch.no_grad():
            y = model.generate(x, max_new_tokens=10, temperature=1.0, top_k=10)
        
        # 解码
        decoded = []
        for token in y[0].tolist():
            if token in itos:
                decoded.append(itos[token])
        
        print(f"Prompt: {prompt}")
        print(f"Generated: {' '.join(decoded)}")
        print()

if __name__ == "__main__":
    test_basic_training()