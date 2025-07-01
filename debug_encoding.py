# debug_encoding.py
import pickle
import os

# 加载元信息
data_dir = 'data/simple_graph/composition_90/composition_90_fixed'
with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)

stoi = meta['stoi']
itos = meta['itos']

print("Vocabulary (stoi):")
for k, v in sorted(stoi.items(), key=lambda x: x[1]):
    print(f"  '{k}' -> {v}")

print(f"\nTotal vocab size: {len(stoi)}")

# 测试编码
test_numbers = ["0", "30", "60", "3", "6"]
print("\nTesting encoding:")
for num in test_numbers:
    if num in stoi:
        print(f"  '{num}' -> {stoi[num]} ✓")
    else:
        print(f"  '{num}' -> NOT FOUND ✗")

# 看看如何正确编码多位数
print("\nEncoding '30' as characters:")
for char in "30":
    if char in stoi:
        print(f"  '{char}' -> {stoi[char]}")