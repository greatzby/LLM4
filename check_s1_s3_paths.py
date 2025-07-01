# check_s1_s3_paths.py
import pickle
import os

data_dir = 'data/simple_graph/composition_90/composition_90_fixed'

# 检查测试数据中S1->S3的路径
print("S1->S3 paths in test data:")
with open(os.path.join(data_dir, 'test.txt'), 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            source, target = int(parts[0]), int(parts[1])
            if source < 30 and target >= 60:  # S1->S3
                print(f"  {line.strip()}")
                if len(parts) > 4:  # 如果路径长度>2
                    print(f"    -> Multi-hop path found!")

# 检查训练数据中是否有多跳路径
print("\nChecking training data for multi-hop paths:")
multi_hop_count = 0
with open(os.path.join(data_dir, 'train_10.txt'), 'r') as f:
    for i, line in enumerate(f):
        parts = line.strip().split()
        if len(parts) > 4:  # 路径长度>2
            multi_hop_count += 1
            if multi_hop_count <= 5:
                print(f"  {line.strip()}")
    
print(f"\nTotal multi-hop paths in training: {multi_hop_count}")