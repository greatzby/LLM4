# check_data_format.py
import os

data_dir = 'data/simple_graph/composition_90/composition_90_fixed'

# 查看训练数据的前几行
print("Training data samples:")
with open(os.path.join(data_dir, 'train_10.txt'), 'r') as f:
    for i, line in enumerate(f):
        if i < 5:
            print(f"  {repr(line.strip())}")
        else:
            break

print("\nTest data samples:")
with open(os.path.join(data_dir, 'test.txt'), 'r') as f:
    for i, line in enumerate(f):
        if i < 5:
            print(f"  {repr(line.strip())}")
        else:
            break

# 检查数据中是否有正确的格式
print("\nChecking data format:")
with open(os.path.join(data_dir, 'train_10.txt'), 'r') as f:
    line = f.readline().strip()
    parts = line.split()
    print(f"First line parts: {parts}")
    print(f"Number of parts: {len(parts)}")