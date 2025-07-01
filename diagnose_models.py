# diagnose_models.py
import pickle
import torch
import os

print("="*80)
print("MODEL COMPOSITION ABILITY DIAGNOSIS")
print("="*80)

# 1. 比较词汇表
print("\n1. VOCABULARY COMPARISON")
print("-"*40)

try:
    # composition_90_fixed词汇表
    with open('data/simple_graph/composition_90/composition_90_fixed/meta.pkl', 'rb') as f:
        meta_fixed = pickle.load(f)
    print("=== composition_90_fixed (Failed Model) ===")
    print(f"Vocab size: {meta_fixed['vocab_size']}")
    print(f"Tokens: {list(meta_fixed['stoi'].items())[:20]}")
except Exception as e:
    print(f"Error loading composition_90_fixed: {e}")

print()

try:
    # composition_90词汇表
    with open('data/simple_graph/composition_90/meta.pkl', 'rb') as f:
        meta_standard = pickle.load(f)
    print("=== composition_90 (Successful Model) ===")
    print(f"Vocab size: {meta_standard['vocab_size']}")
    print(f"Tokens: {list(meta_standard['stoi'].items())[:20]}")
except Exception as e:
    print(f"Error loading composition_90: {e}")

# 2. 训练数据格式
print("\n\n2. TRAINING DATA FORMAT")
print("-"*40)

try:
    print("=== composition_90_fixed训练数据 ===")
    with open('data/simple_graph/composition_90/composition_90_fixed/train_10.txt', 'r') as f:
        for i in range(5):
            print(f"  {f.readline().strip()}")
except Exception as e:
    print(f"Error reading fixed training data: {e}")

print()

try:
    print("=== composition_90训练数据 ===")
    with open('data/simple_graph/composition_90/train.txt', 'r') as f:
        for i in range(5):
            print(f"  {f.readline().strip()}")
except Exception as e:
    print(f"Error reading standard training data: {e}")

# 3. 模型配置
print("\n\n3. MODEL CONFIGURATIONS")
print("-"*40)

try:
    # 第一个模型（失败的）
    ckpt1 = torch.load('out/composition_fixed_20250702_044341/ckpt_5000.pt', map_location='cpu')
    print("=== Failed Model (composition_fixed) ===")
    print(f"  Iterations: 5000")
    print(f"  Config: {ckpt1.get('config', 'Not found')}")
    if 'model_args' in ckpt1:
        print(f"  Model args: {ckpt1['model_args']}")
except Exception as e:
    print(f"Error loading failed model: {e}")

print()

try:
    # 第二个模型（成功的）
    ckpt2 = torch.load('out/composition_standard_20250702_023335/ckpt_35000.pt', map_location='cpu')
    print("=== Successful Model (composition_standard) ===")
    print(f"  Iterations: 35000")
    print(f"  Config: {ckpt2.get('config', 'Not found')}")
    if 'model_args' in ckpt2:
        print(f"  Model args: {ckpt2['model_args']}")
except Exception as e:
    print(f"Error loading successful model: {e}")

# 4. 关键差异总结
print("\n\n4. KEY DIFFERENCES SUMMARY")
print("-"*40)

try:
    if 'meta_fixed' in locals() and 'meta_standard' in locals():
        print(f"Vocabulary Type:")
        print(f"  - Failed model: Character-level (vocab_size={meta_fixed['vocab_size']})")
        print(f"  - Success model: Token-level (vocab_size={meta_standard['vocab_size']})")
except:
    pass

print("\n" + "="*80)
print("END OF DIAGNOSIS")
print("="*80)