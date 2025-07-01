# verify_final_model.py
import torch
import pickle
import os
from model import GPTConfig, GPT

def verify_final_model(checkpoint_path):
    """验证最终模型的组合能力"""
    
    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path)
    
    # 显示训练历史
    history = checkpoint['history']
    print("Training History Summary:")
    print(f"  Final S1->S2: {history['ar_s1_s2'][-1]:.2%}")
    print(f"  Final S2->S3: {history['ar_s2_s3'][-1]:.2%}")
    print(f"  Final S1->S3: {history['ar_s1_s3'][-1]:.2%}")
    
    # 绘制学习曲线
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.plot(history['iter'], history['ar_s1_s2'], 'b-', label='S1->S2')
    plt.plot(history['iter'], history['ar_s2_s3'], 'g-', label='S2->S3')
    plt.plot(history['iter'], history['ar_s1_s3'], 'r-', label='S1->S3', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Compositional Generalization Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig('composition_learning_curve.png')
    print("\nLearning curve saved to composition_learning_curve.png")

if __name__ == "__main__":
    # 使用最新的checkpoint
    import glob
    ckpt_files = glob.glob('out/composition_fixed_*/ckpt_*.pt')
    if ckpt_files:
        latest_ckpt = max(ckpt_files, key=os.path.getctime)
        print(f"Using checkpoint: {latest_ckpt}")
        verify_final_model(latest_ckpt)
    else:
        print("No checkpoint found!")