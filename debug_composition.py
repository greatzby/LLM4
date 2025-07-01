# debug_composition.py - 用于调试的简化版本
import torch
import numpy as np
import pickle
import os

def debug_data():
    """调试数据格式"""
    # 加载数据
    data_dir = 'data/simple_graph/composition_90'
    
    # 1. 检查原始文本
    print("=== Checking raw text files ===")
    with open(os.path.join(data_dir, 'train_10.txt'), 'r') as f:
        lines = f.readlines()
        print(f"Train samples: {len(lines)}")
        for i in range(min(3, len(lines))):
            print(f"  Sample {i}: {lines[i].strip()}")
    
    with open(os.path.join(data_dir, 'test.txt'), 'r') as f:
        lines = f.readlines()
        print(f"\nTest samples: {len(lines)}")
        for i in range(min(3, len(lines))):
            print(f"  Sample {i}: {lines[i].strip()}")
    
    # 2. 检查元信息
    print("\n=== Checking meta info ===")
    with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    
    print(f"Block size: {meta['block_size']}")
    print(f"Vocab size: {meta['vocab_size']}")
    print(f"Sample mappings:")
    for i in [0, 1, 30, 60]:
        if str(i) in meta['stoi']:
            print(f"  '{i}' -> token {meta['stoi'][str(i)]}")
    
    # 3. 检查二进制数据
    print("\n=== Checking binary data ===")
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    print(f"Val data size: {len(val_data)}")
    
    # 显示前几个序列
    block_size = meta['block_size']
    data_size = block_size + 1
    
    for i in range(min(3, len(val_data) // data_size)):
        start = i * data_size
        seq = val_data[start:start + data_size]
        print(f"\nSequence {i}:")
        print(f"  Raw tokens: {seq[:10]}...")
        
        # 解码
        decoded = []
        for token in seq:
            if token in meta['itos']:
                decoded.append(meta['itos'][token])
            else:
                decoded.append('?')
        print(f"  Decoded: {' '.join(decoded[:10])}...")
        
        # 找到source和target
        non_pad = [t for t in seq if t > 1]
        if len(non_pad) >= 2:
            source_token = non_pad[0]
            target_token = non_pad[1]
            source = source_token - 2
            target = target_token - 2
            print(f"  Source: {source}, Target: {target}")

if __name__ == "__main__":
    debug_data()