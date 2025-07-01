# check_val_data.py
import numpy as np
import pickle
import os

def check_validation_data():
    data_dir = 'data/simple_graph/composition_90'
    
    # 加载元信息
    with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    
    # 加载验证数据
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    
    block_size = meta['block_size']
    data_size = block_size + 1
    num_sequences = len(val_data) // data_size
    
    print(f"Validation set: {num_sequences} sequences")
    
    # 统计路径类型
    s1_s2_count = 0
    s2_s3_count = 0
    s1_s3_count = 0
    
    for i in range(min(num_sequences, 100)):
        start_idx = i * data_size
        seq = val_data[start_idx:start_idx + data_size]
        
        # 找到source和target
        source_token = None
        target_token = None
        
        for token in seq:
            if token > 1:  # 不是PAD或newline
                if source_token is None:
                    source_token = token
                elif target_token is None:
                    target_token = token
                    break
        
        if source_token and target_token:
            source = source_token - 2
            target = target_token - 2
            
            if 0 <= source < 30 and 30 <= target < 60:
                s1_s2_count += 1
            elif 30 <= source < 60 and 60 <= target < 90:
                s2_s3_count += 1
            elif 0 <= source < 30 and 60 <= target < 90:
                s1_s3_count += 1
    
    print(f"\nPath type distribution (first 100):")
    print(f"  S1->S2: {s1_s2_count}")
    print(f"  S2->S3: {s2_s3_count}")
    print(f"  S1->S3: {s1_s3_count}")
    
    # 显示前几个序列
    print(f"\nFirst 5 sequences:")
    for i in range(min(5, num_sequences)):
        start_idx = i * data_size
        seq = val_data[start_idx:start_idx + 10]
        decoded = []
        for token in seq:
            if token in meta['itos']:
                decoded.append(meta['itos'][token])
        print(f"  {' '.join(decoded)}")

if __name__ == "__main__":
    check_validation_data()