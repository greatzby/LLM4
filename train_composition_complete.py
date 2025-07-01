# train_composition_fixed.py - 修复版的完整训练脚本
# 这是一个完整的文件，包含了所有修复

import os
import time
import math
import pickle
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx
from contextlib import nullcontext
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict

from model import GPTConfig, GPT
from logger import get_logger

def parse_args():
    parser = argparse.ArgumentParser(description='Composition Ability Training')
    parser.add_argument('--experiment_name', type=str, default='composition', help='Experiment name')
    parser.add_argument('--total_nodes', type=int, default=90, help='Total nodes (3 stages)')
    parser.add_argument('--n_layer', type=int, default=1)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--n_embd', type=int, default=120)
    parser.add_argument('--max_iters', type=int, default=50000)
    parser.add_argument('--test_interval', type=int, default=1000)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--train_paths_per_pair', type=int, default=10)
    parser.add_argument('--checkpoint_interval', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--training_mode', type=str, default='standard', 
                      choices=['standard', 'mixed', 'curriculum'])
    parser.add_argument('--mixed_ratio', type=float, default=0.1)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_k', type=int, default=50)
    return parser.parse_args()

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def encode(s, stoi):
    """编码字符串为token序列"""
    tokens = []
    for ch in s.split():
        if ch in stoi:
            tokens.append(stoi[ch])
    return tokens

def decode(l, itos):
    """解码token序列为字符串"""
    return " ".join([itos[i] for i in l if i in itos])

def classify_path_type(source, target, stages):
    """判断路径类型"""
    S1, S2, S3 = stages
    if source in S1 and target in S2:
        return 'S1->S2'
    elif source in S2 and target in S3:
        return 'S2->S3'
    elif source in S1 and target in S3:
        return 'S1->S3'
    else:
        return 'other'

def check_path_validity(G, path_nodes):
    """检查路径是否有效"""
    if len(path_nodes) < 2:
        return False
    for i in range(len(path_nodes) - 1):
        if not G.has_edge(str(path_nodes[i]), str(path_nodes[i+1])):
            return False
    return True

def analyze_composition_error(pred_path, target_path, stages):
    """分析组合错误类型"""
    S1, S2, S3 = stages
    
    if len(pred_path) < 2:
        return "too_short"
    
    pred_source, pred_target = pred_path[0], pred_path[-1]
    true_source, true_target = target_path[0], target_path[-1]
    
    if pred_source != true_source:
        return "wrong_source"
    if pred_target != true_target:
        return "wrong_target"
    
    # 检查是否经过了S2（对于S1->S3路径）
    if true_source in S1 and true_target in S3:
        has_s2 = any(node in S2 for node in pred_path[1:-1])
        if not has_s2:
            return "no_intermediate_stage"
    
    return "other"

@torch.no_grad()
def evaluate_tf_by_type(model, val_data, stages, stoi, itos, device, block_size, batch_size=64):
    """在验证集上分别评估不同路径类型的Teacher Forcing准确率"""
    model.eval()
    
    S1, S2, S3 = stages
    data_size = block_size + 1
    
    # 初始化统计
    results = {
        'S1->S2': {'correct': 0, 'total': 0},
        'S2->S3': {'correct': 0, 'total': 0},
        'S1->S3': {'correct': 0, 'total': 0},
        'overall': {'correct': 0, 'total': 0}
    }
    
    # 遍历验证数据
    num_sequences = min(len(val_data) // data_size, 200)  # 评估前200个序列
    
    for seq_idx in range(num_sequences):
        start_idx = seq_idx * data_size
        seq = val_data[start_idx:start_idx + data_size]
        
        # 找到source和target
        source_token = None
        target_token = None
        
        for i, token in enumerate(seq):
            if token > 1:  # 不是PAD(0)或newline(1)
                if source_token is None:
                    source_token = token
                elif target_token is None:
                    target_token = token
                    break
        
        if source_token is None or target_token is None:
            continue
            
        # 转换为节点ID
        source = source_token - 2
        target = target_token - 2
        
        # 判断路径类型
        path_type = classify_path_type(source, target, stages)
        if path_type == 'other':
            continue
        
        # 准备输入
        x = torch.from_numpy(seq[:block_size].astype(np.int64)).unsqueeze(0).to(device)
        y = torch.from_numpy(seq[1:1+block_size].astype(np.int64)).unsqueeze(0).to(device)
        
        # 获取预测
        with torch.no_grad():
            logits, _ = model(x, y)
        
        preds = torch.argmax(logits, dim=-1)
        
        # 计算准确率（只在非padding位置）
        mask = y[0] != 0
        if mask.sum() > 0:
            correct = (preds[0][mask] == y[0][mask]).sum().item()
            total = mask.sum().item()
            
            results[path_type]['correct'] += correct
            results[path_type]['total'] += total
            results['overall']['correct'] += correct
            results['overall']['total'] += total
    
    # 计算准确率
    for path_type in results:
        if results[path_type]['total'] > 0:
            results[path_type]['accuracy'] = results[path_type]['correct'] / results[path_type]['total']
        else:
            results[path_type]['accuracy'] = 0.0
    
    model.train()
    return results

@torch.no_grad()
def evaluate_ar_by_type(model, test_file, stages, stoi, itos, device, G, max_eval=50):
    """在测试集上使用Autoregressive生成评估不同路径类型"""
    model.eval()
    
    S1, S2, S3 = stages
    
    # 读取测试数据
    with open(test_file, 'r') as f:
        test_lines = [line.strip() for line in f.readlines() if line.strip()]
    
    # 分类测试数据
    test_by_type = {
        'S1->S2': [],
        'S2->S3': [],
        'S1->S3': []
    }
    
    for line in test_lines:
        parts = line.split()
        if len(parts) >= 2:
            source, target = int(parts[0]), int(parts[1])
            path_type = classify_path_type(source, target, stages)
            if path_type in test_by_type:
                test_by_type[path_type].append(line)
    
    # 结果统计
    results = {}
    
    for path_type, test_cases in test_by_type.items():
        results[path_type] = {
            'correct': 0,
            'total': min(len(test_cases), max_eval),
            'errors': defaultdict(int)
        }
        
        # 评估每个测试案例
        for idx, test_line in enumerate(test_cases[:max_eval]):
            parts = test_line.split()
            source, target = parts[0], parts[1]
            true_path = [int(p) for p in parts]
            
            # 准备输入prompt
            prompt = f"{source} {target} {source}"
            prompt_ids = encode(prompt, stoi)
            
            if not prompt_ids:
                continue
                
            x = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
            
            # 生成预测
            try:
                with torch.no_grad():
                    y = model.generate(x, max_new_tokens=30, 
                                      temperature=args.temperature, 
                                      top_k=args.top_k)
                
                # 解码并解析
                pred_str = decode(y[0].tolist(), itos)
                pred_parts = pred_str.split()
                
                # 提取预测路径
                pred_path = []
                for p in pred_parts:
                    if p == '\n':
                        break
                    if p.isdigit():
                        pred_path.append(int(p))
                
                # 检查正确性
                is_correct = False
                if len(pred_path) >= 2:
                    if pred_path[0] == int(source) and pred_path[-1] == int(target):
                        # 检查路径有效性
                        if check_path_validity(G, pred_path):
                            is_correct = True
                            results[path_type]['correct'] += 1
                
                # 如果错误，分析错误类型
                if not is_correct:
                    error_type = analyze_composition_error(pred_path, true_path, stages)
                    results[path_type]['errors'][error_type] += 1
            except Exception as e:
                results[path_type]['errors']['generation_error'] += 1
        
        # 计算准确率
        if results[path_type]['total'] > 0:
            results[path_type]['accuracy'] = results[path_type]['correct'] / results[path_type]['total']
        else:
            results[path_type]['accuracy'] = 0.0
    
    model.train()
    return results

def main():
    global args  # 让evaluate_ar_by_type能访问args
    args = parse_args()
    set_seed(args.seed)
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f'out/composition_{args.training_mode}_{timestamp}'
    os.makedirs(out_dir, exist_ok=True)
    
    # 设置logger
    logger = get_logger(os.path.join(out_dir, "train.log"))
    
    print("="*60)
    print(f"Composition Ability Training")
    print(f"Mode: {args.training_mode}")
    print(f"Model: {args.n_layer}L-{args.n_head}H-{args.n_embd}D")
    print("="*60)
    
    # 记录配置
    logger.info("="*60)
    logger.info(f"Configuration: {vars(args)}")
    logger.info("="*60)
    
    # 加载数据和元信息
    data_dir = os.path.join('data', 'simple_graph', f'{args.experiment_name}_{args.total_nodes}')
    
    # 加载阶段信息
    with open(os.path.join(data_dir, 'stage_info.pkl'), 'rb') as f:
        stage_info = pickle.load(f)
    stages = stage_info['stages']
    S1, S2, S3 = stages
    
    print(f"Stage info: S1={S1[:3]}..., S2={S2[:3]}..., S3={S3[:3]}...")
    
    # 加载元信息
    with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    
    stoi, itos = meta['stoi'], meta['itos']
    block_size = meta['block_size']
    vocab_size = meta['vocab_size']
    
    # 加载图
    G = nx.read_graphml(os.path.join(data_dir, 'composition_graph.graphml'))
    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # 加载数据
    train_data = np.memmap(os.path.join(data_dir, f'train_{args.train_paths_per_pair}.bin'), 
                          dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    test_file = os.path.join(data_dir, 'test.txt')
    
    print(f"Data loaded: train={len(train_data)//block_size} sequences, val={len(val_data)//block_size} sequences")
    
    # 初始化模型
    model_args = dict(
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        block_size=block_size,
        bias=False,
        vocab_size=vocab_size,
        dropout=0.0
    )
    
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf).to(args.device)
    
    print(f"Model initialized: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    
    # 优化器
    optimizer = model.configure_optimizers(
        weight_decay=1e-1,
        learning_rate=args.learning_rate,
        betas=(0.9, 0.95),
        device_type='cuda' if 'cuda' in args.device else 'cpu'
    )
    
    # 数据加载函数
    def get_batch(split):
        data = train_data if split == 'train' else val_data
        data_size = block_size + 1
        
        # 确保对齐到序列边界
        num_sequences = len(data) // data_size
        seq_indices = torch.randint(0, num_sequences, (args.batch_size,))
        ix = seq_indices * data_size
        
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        
        return x.to(args.device), y.to(args.device)
    
    # 训练历史
    history = {
        'iter': [],
        'train_loss': [],
        'val_loss': [],
        # Teacher Forcing 准确率（验证集）
        'tf_overall': [],
        'tf_s1_s2': [],
        'tf_s2_s3': [],
        'tf_s1_s3': [],
        # Autoregressive 准确率（测试集）
        'ar_s1_s2': [],
        'ar_s2_s3': [],
        'ar_s1_s3': [],
        # 错误分析
        's1_s3_errors': []
    }
    
    # 训练循环
    print("\nStarting training...")
    print(f"Evaluation: TF on validation set, AR on test set")
    running_loss = 0
    loss_count = 0
    
    for iter_num in range(args.max_iters + 1):
        # 学习率调度
        lr = args.learning_rate
        if iter_num < 2000:
            lr = args.learning_rate * iter_num / 2000
        elif iter_num > args.max_iters * 0.9:
            lr = args.learning_rate * 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # 定期评估
        if iter_num % args.test_interval == 0:
            # 计算平均训练损失
            avg_train_loss = running_loss / loss_count if loss_count > 0 else 0
            
            # 验证集损失
            model.eval()
            val_losses = []
            for _ in range(10):
                X_val, Y_val = get_batch('val')
                with torch.no_grad():
                    _, loss = model(X_val, Y_val)
                val_losses.append(loss.item())
            val_loss = np.mean(val_losses)
            
            # Teacher Forcing 评估（验证集）
            tf_results = evaluate_tf_by_type(model, val_data, stages, stoi, itos, args.device, block_size)
            
            # Autoregressive 评估（测试集）
            ar_results = evaluate_ar_by_type(model, test_file, stages, stoi, itos, args.device, G)
            
            # 记录历史
            history['iter'].append(iter_num)
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            
            # TF准确率
            history['tf_overall'].append(tf_results['overall']['accuracy'])
            history['tf_s1_s2'].append(tf_results['S1->S2']['accuracy'])
            history['tf_s2_s3'].append(tf_results['S2->S3']['accuracy'])
            history['tf_s1_s3'].append(tf_results['S1->S3']['accuracy'])
            
            # AR准确率
            history['ar_s1_s2'].append(ar_results['S1->S2']['accuracy'])
            history['ar_s2_s3'].append(ar_results['S2->S3']['accuracy'])
            history['ar_s1_s3'].append(ar_results['S1->S3']['accuracy'])
            history['s1_s3_errors'].append(ar_results['S1->S3']['errors'])
            
            # 打印结果
            print(f"\n{'='*60}")
            print(f"Iteration {iter_num}:")
            print(f"  Loss: train={avg_train_loss:.4f}, val={val_loss:.4f}")
            
            print(f"\n  Teacher Forcing (Validation Set):")
            print(f"    Overall: {tf_results['overall']['accuracy']:.2%}")
            print(f"    S1->S2: {tf_results['S1->S2']['accuracy']:.2%} ({tf_results['S1->S2']['total']} tokens)")
            print(f"    S2->S3: {tf_results['S2->S3']['accuracy']:.2%} ({tf_results['S2->S3']['total']} tokens)")
            print(f"    S1->S3: {tf_results['S1->S3']['accuracy']:.2%} ({tf_results['S1->S3']['total']} tokens)")
            
            print(f"\n  Autoregressive (Test Set):")
            print(f"    S1->S2: {ar_results['S1->S2']['accuracy']:.2%} ({ar_results['S1->S2']['correct']}/{ar_results['S1->S2']['total']})")
            print(f"    S2->S3: {ar_results['S2->S3']['accuracy']:.2%} ({ar_results['S2->S3']['correct']}/{ar_results['S2->S3']['total']})")
            print(f"    S1->S3: {ar_results['S1->S3']['accuracy']:.2%} ({ar_results['S1->S3']['correct']}/{ar_results['S1->S3']['total']})")
            
            # S1->S3错误分析
            if ar_results['S1->S3']['errors']:
                print(f"\n  S1->S3 Error Analysis:")
                for error_type, count in ar_results['S1->S3']['errors'].items():
                    print(f"    {error_type}: {count}")
            
            # 组合能力评估
            if ar_results['S1->S2']['accuracy'] > 0.8 and ar_results['S2->S3']['accuracy'] > 0.8:
                if ar_results['S1->S3']['accuracy'] < 0.2:
                    print("\n  ⚠️  Poor composition ability despite good basic performance!")
            
            # 记录日志
            logger.info(f"Iter {iter_num}: loss={avg_train_loss:.4f}, "
                       f"TF: S1->S3={tf_results['S1->S3']['accuracy']:.2%}, "
                       f"AR: S1->S3={ar_results['S1->S3']['accuracy']:.2%}")
            
            running_loss = 0
            loss_count = 0
            model.train()
        
        # 保存checkpoint
        if iter_num % args.checkpoint_interval == 0 and iter_num > 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'history': history,
                'config': vars(args)
            }
            checkpoint_path = os.path.join(out_dir, f'ckpt_{iter_num}.pt')
            torch.save(checkpoint, checkpoint_path)
            print(f"  Checkpoint saved to {checkpoint_path}")
        
        if iter_num == 0:
            continue
        
        # 训练步
        X, Y = get_batch('train')
        
        logits, loss = model(X, Y)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        running_loss += loss.item()
        loss_count += 1
    
    # 保存最终结果
    with open(os.path.join(out_dir, 'history.pkl'), 'wb') as f:
        pickle.dump(history, f)
    
    # 绘制图表（简化版）
    plt.figure(figsize=(15, 10))
    
    # 1. 训练损失
    plt.subplot(2, 3, 1)
    plt.plot(history['iter'], history['train_loss'], 'b-', label='Train')
    plt.plot(history['iter'], history['val_loss'], 'r-', label='Val')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    
    # 2. TF准确率
    plt.subplot(2, 3, 2)
    plt.plot(history['iter'], history['tf_s1_s2'], 'b-', label='S1->S2', marker='o')
    plt.plot(history['iter'], history['tf_s2_s3'], 'g-', label='S2->S3', marker='s')
    plt.plot(history['iter'], history['tf_s1_s3'], 'r-', label='S1->S3', marker='^')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Teacher Forcing by Path Type')
    plt.legend()
    plt.grid(True)
    
    # 3. AR准确率
    plt.subplot(2, 3, 3)
    plt.plot(history['iter'], history['ar_s1_s2'], 'b-', label='S1->S2', marker='o')
    plt.plot(history['iter'], history['ar_s2_s3'], 'g-', label='S2->S3', marker='s')
    plt.plot(history['iter'], history['ar_s1_s3'], 'r-', label='S1->S3', marker='^', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Autoregressive by Path Type')
    plt.legend()
    plt.grid(True)
    
    # 4. S1->S3对比
    plt.subplot(2, 3, 4)
    plt.plot(history['iter'], history['tf_s1_s3'], 'b-', label='TF', marker='o')
    plt.plot(history['iter'], history['ar_s1_s3'], 'r-', label='AR', marker='s')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('S1->S3: TF vs AR')
    plt.legend()
    plt.grid(True)
    
    # 5. 组合能力差距
    plt.subplot(2, 3, 5)
    if len(history['ar_s1_s2']) > 0:
        basic_avg = [(h1 + h2) / 2 for h1, h2 in zip(history['ar_s1_s2'], history['ar_s2_s3'])]
        composition_gap = [b - c for b, c in zip(basic_avg, history['ar_s1_s3'])]
        plt.plot(history['iter'], composition_gap, 'purple', linewidth=2)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        plt.xlabel('Iteration')
        plt.ylabel('Gap')
        plt.title('Composition Gap (Basic Avg - S1->S3)')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'composition_results.png'), dpi=150)
    plt.close()
    
    # 打印最终总结
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"\nFinal Results:")
    print(f"\nTeacher Forcing (Validation Set):")
    print(f"  S1->S2: {history['tf_s1_s2'][-1]:.2%}")
    print(f"  S2->S3: {history['tf_s2_s3'][-1]:.2%}")
    print(f"  S1->S3: {history['tf_s1_s3'][-1]:.2%} ← Composition (TF)")
    
    print(f"\nAutoregressive (Test Set):")
    print(f"  S1->S2: {history['ar_s1_s2'][-1]:.2%}")
    print(f"  S2->S3: {history['ar_s2_s3'][-1]:.2%}")
    print(f"  S1->S3: {history['ar_s1_s3'][-1]:.2%} ← Composition (AR)")
    
    # 组合能力评估
    if len(history['ar_s1_s2']) > 0:
        basic_performance = (history['ar_s1_s2'][-1] + history['ar_s2_s3'][-1]) / 2
        composition_performance = history['ar_s1_s3'][-1]
        
        print(f"\nComposition Analysis:")
        print(f"  Basic Path Average: {basic_performance:.2%}")
        print(f"  Composition Performance: {composition_performance:.2%}")
        print(f"  Composition Gap: {basic_performance - composition_performance:.2%}")
        
        if composition_performance < 0.1 and basic_performance > 0.8:
            print("\n⚠️  Model shows severe lack of compositional generalization!")
        elif composition_performance > 0.5:
            print("\n✅ Model demonstrates reasonable compositional ability!")
        else:
            print("\n🔶 Model shows limited compositional ability.")
    
    print(f"\nResults saved to: {out_dir}")

if __name__ == "__main__":
    main()