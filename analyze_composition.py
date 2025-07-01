# analyze_composition.py
import pickle
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

def analyze_checkpoint(checkpoint_path):
    """分析模型的注意力权重和内部表示"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_state = checkpoint['model']
    
    # 分析注意力模式
    # 这里可以添加更详细的分析
    
    return model_state

def compare_training_modes(standard_dir, mixed_dir):
    """比较不同训练模式的结果"""
    # 加载历史
    with open(os.path.join(standard_dir, 'history.pkl'), 'rb') as f:
        standard_history = pickle.load(f)
    
    with open(os.path.join(mixed_dir, 'history.pkl'), 'rb') as f:
        mixed_history = pickle.load(f)
    
    # 绘制对比图
    plt.figure(figsize=(10, 6))
    
    plt.plot(standard_history['iter'], standard_history['s1_s3_acc'], 
             'b-', label='Standard Training', linewidth=2)
    plt.plot(mixed_history['iter'], mixed_history['s1_s3_acc'], 
             'r-', label='Mixed Training', linewidth=2)
    
    plt.xlabel('Iteration')
    plt.ylabel('S1->S3 Accuracy')
    plt.title('Composition Ability Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('composition_comparison.png')
    
    print("Comparison saved to composition_comparison.png")

if __name__ == "__main__":
    # 使用示例
    # analyze_checkpoint('out/composition_standard_xxx/ckpt_50000.pt')
    # compare_training_modes('out/composition_standard_xxx', 'out/composition_mixed_xxx')
    pass