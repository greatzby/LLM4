# train_composition_fixed_eval.py
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
    parser.add_argument('--data_dir', type=str, default='data/simple_graph/composition_90_fixed')
    parser.add_argument('--n_layer', type=int, default=1)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--n_embd', type=int, default=120)
    parser.add_argument('--max_iters', type=int, default=50000)
    parser.add_argument('--test_interval', type=int, default=1000)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--train_paths_per_pair', type=int, default=10)
    parser.add_argument('--checkpoint_interval', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval_temperature', type=float, default=0.1)
    parser.add_argument('--eval_top_k', type=int, default=10)
    return parser.parse_args()

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# train_composition_fixed_eval_v3.py
# 复制原文件并修改evaluate_ar_correct函数

@torch.no_grad()
def evaluate_ar_correct(model, test_file, stages, stoi, itos, device, G, temperature=0.1, top_k=10, max_eval_per_type=50):
    """修复后的AR评估函数（正确解析生成格式）"""
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
            try:
                source, target = int(parts[0]), int(parts[1])
                # 提取真实路径（从第2个位置开始）
                true_path = [int(p) for p in parts[2:] if p.isdigit()]
                
                if source in S1 and target in S2:
                    test_by_type['S1->S2'].append((source, target, true_path))
                elif source in S2 and target in S3:
                    test_by_type['S2->S3'].append((source, target, true_path))
                elif source in S1 and target in S3:
                    test_by_type['S1->S3'].append((source, target, true_path))
            except:
                continue
    
    # 结果统计
    results = {}
    
    for path_type, test_cases in test_by_type.items():
        n_eval = min(len(test_cases), max_eval_per_type)
        results[path_type] = {
            'correct': 0,
            'total': n_eval,
            'errors': defaultdict(int),
            'examples': {'success': [], 'failure': []}
        }
        
        # 评估
        for idx in range(n_eval):
            source, target, true_path = test_cases[idx]
            
            # 构建prompt：source target（不包含路径开始）
            prompt_str = f"{source} {target}"
            
            # 按字符编码
            prompt_ids = []
            for char in prompt_str:
                if char in stoi:
                    prompt_ids.append(stoi[char])
            
            x = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
            
            # 生成
            with torch.no_grad():
                y = model.generate(x, max_new_tokens=50, temperature=temperature, top_k=top_k)
            
            # 获取生成的tokens（包括prompt）
            full_sequence = y[0].tolist()
            
            # 将整个序列解码为字符串
            full_chars = []
            for token_id in full_sequence:
                if token_id == 1:  # newline
                    break
                if token_id in itos and token_id > 1:
                    full_chars.append(itos[token_id])
            
            # 解析整个序列的数字
            full_str = ''.join(full_chars)
            all_numbers = []
            current_num = ""
            
            for char in full_str:
                if char == ' ':
                    if current_num and current_num.isdigit():
                        all_numbers.append(int(current_num))
                    current_num = ""
                elif char.isdigit():
                    current_num += char
            
            if current_num and current_num.isdigit():
                all_numbers.append(int(current_num))
            
            # 提取生成的路径（从第2个数字开始，因为前两个是source和target）
            if len(all_numbers) >= 3:
                generated_path = all_numbers[2:]  # 跳过source和target
            else:
                generated_path = []
            
            # 验证结果
            success = False
            error_type = None
            
            if len(generated_path) >= 2:
                # 检查路径的起点和终点
                if generated_path[0] == source and generated_path[-1] == target:
                    if path_type == 'S1->S3':
                        # 对于S1->S3，检查是否经过S2
                        has_s2 = any(node in S2 for node in generated_path[1:-1])
                        if has_s2:
                            # 验证路径有效性
                            path_valid = True
                            for i in range(len(generated_path) - 1):
                                if not G.has_edge(str(generated_path[i]), str(generated_path[i+1])):
                                    path_valid = False
                                    error_type = 'invalid_edge'
                                    break
                            if path_valid:
                                success = True
                        else:
                            error_type = 'no_s2_intermediate'
                    else:
                        # 对于S1->S2和S2->S3，验证路径有效性
                        path_valid = True
                        for i in range(len(generated_path) - 1):
                            if not G.has_edge(str(generated_path[i]), str(generated_path[i+1])):
                                path_valid = False
                                error_type = 'invalid_edge'
                                break
                        if path_valid:
                            success = True
                else:
                    error_type = 'wrong_endpoints'
            else:
                error_type = 'too_short'
            
            if success:
                results[path_type]['correct'] += 1
                if len(results[path_type]['examples']['success']) < 3:
                    results[path_type]['examples']['success'].append({
                        'source': source,
                        'target': target, 
                        'generated': generated_path,
                        'true': true_path,
                        'full_output': all_numbers
                    })
            else:
                results[path_type]['errors'][error_type if error_type else 'unknown'] += 1
                if len(results[path_type]['examples']['failure']) < 3:
                    results[path_type]['examples']['failure'].append({
                        'source': source,
                        'target': target,
                        'generated': generated_path,
                        'error': error_type,
                        'full_output': all_numbers,
                        'true': true_path
                    })
        
        # 计算准确率
        results[path_type]['accuracy'] = results[path_type]['correct'] / results[path_type]['total'] if results[path_type]['total'] > 0 else 0.0
    
    model.train()
    return results

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f'out/composition_fixed_{timestamp}'
    os.makedirs(out_dir, exist_ok=True)
    
    # 设置logger
    logger = get_logger(os.path.join(out_dir, "train.log"))
    
    print("="*60)
    print(f"Composition Ability Training (Fixed Evaluation)")
    print(f"Model: {args.n_layer}L-{args.n_head}H-{args.n_embd}D")
    print(f"Data: {args.data_dir}")
    print("="*60)
    
    # 记录配置
    logger.info("="*60)
    logger.info(f"Configuration: {vars(args)}")
    logger.info("="*60)
    
    # 加载数据
    data_dir = args.data_dir
    
    # 加载阶段信息
    with open(os.path.join(data_dir, 'stage_info.pkl'), 'rb') as f:
        stage_info = pickle.load(f)
    stages = stage_info['stages']
    S1, S2, S3 = stages
    
    print(f"Stage info: S1={len(S1)} nodes, S2={len(S2)} nodes, S3={len(S3)} nodes")
    
    # 加载元信息
    with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    
    stoi, itos = meta['stoi'], meta['itos']
    block_size = meta['block_size']
    vocab_size = meta['vocab_size']
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Block size: {block_size}")
    chars = [c for c in stoi.keys() if c not in ['[PAD]', '\n']]
    print(f"Character vocabulary: {chars}")
    
    # 加载图
    G = nx.read_graphml(os.path.join(data_dir, 'composition_graph.graphml'))
    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # 加载数据
    train_data = np.memmap(os.path.join(data_dir, f'train_{args.train_paths_per_pair}.bin'), 
                          dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    test_file = os.path.join(data_dir, 'test.txt')
    
    print(f"Data loaded: train={len(train_data)//(block_size+1)} sequences, val={len(val_data)//(block_size+1)} sequences")
    
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
        'ar_s1_s2': [],
        'ar_s2_s3': [],
        'ar_s1_s3': [],
        's1_s3_errors': []
    }
    
    # 训练循环
    print("\nStarting training...")
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
            
            # AR评估（使用修复的函数）
            ar_results = evaluate_ar_correct(model, test_file, stages, stoi, itos, args.device, G,
                                           temperature=args.eval_temperature, top_k=args.eval_top_k)
            
            # 记录历史
            history['iter'].append(iter_num)
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            history['ar_s1_s2'].append(ar_results['S1->S2']['accuracy'])
            history['ar_s2_s3'].append(ar_results['S2->S3']['accuracy'])
            history['ar_s1_s3'].append(ar_results['S1->S3']['accuracy'])
            history['s1_s3_errors'].append(ar_results['S1->S3']['errors'])
            
            # 打印结果
            print(f"\n{'='*60}")
            print(f"Iteration {iter_num}:")
            print(f"  Loss: train={avg_train_loss:.4f}, val={val_loss:.4f}")
            
            print(f"\n  Autoregressive Generation (Test Set):")
            for path_type in ['S1->S2', 'S2->S3', 'S1->S3']:
                res = ar_results[path_type]
                print(f"    {path_type}: {res['accuracy']:.2%} ({res['correct']}/{res['total']})")
                
                # 打印成功示例
                if res['examples']['success'] and iter_num <= 5000:  # 前期打印示例
                    ex = res['examples']['success'][0]
                    print(f"      ✓ Example: {ex['source']}→{ex['target']}, generated: {ex['generated']}")
                
                # 打印错误分布
                if res['errors']:
                    error_str = ', '.join([f"{k}: {v}" for k, v in res['errors'].items()])
                    print(f"      Errors: {error_str}")
            
            # 组合能力评估
            if iter_num >= 10000:
                s1_s2_acc = ar_results['S1->S2']['accuracy']
                s2_s3_acc = ar_results['S2->S3']['accuracy']
                s1_s3_acc = ar_results['S1->S3']['accuracy']
                
                if s1_s2_acc > 0.8 and s2_s3_acc > 0.8:
                    if s1_s3_acc > 0.8:
                        print("\n  ✅ Model demonstrates strong compositional ability!")
                    elif s1_s3_acc < 0.2:
                        print("\n  ⚠️  Poor composition ability despite good basic performance!")
            
            # 记录日志
            logger.info(f"Iter {iter_num}: loss={avg_train_loss:.4f}, "
                       f"S1->S2={ar_results['S1->S2']['accuracy']:.2%}, "
                       f"S2->S3={ar_results['S2->S3']['accuracy']:.2%}, "
                       f"S1->S3={ar_results['S1->S3']['accuracy']:.2%}")
            
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
    
    # 绘制结果图
    plot_results(history, out_dir)
    
    # 打印最终总结
    print_final_summary(history, logger)
    
    print(f"\nResults saved to: {out_dir}")

def plot_results(history, out_dir):
    """绘制训练结果"""
    plt.figure(figsize=(15, 10))
    
    # 1. 损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(history['iter'], history['train_loss'], 'b-', label='Train')
    plt.plot(history['iter'], history['val_loss'], 'r-', label='Val')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    
    # 2. AR准确率
    plt.subplot(2, 2, 2)
    plt.plot(history['iter'], history['ar_s1_s2'], 'b-', label='S1->S2', marker='o', markersize=3)
    plt.plot(history['iter'], history['ar_s2_s3'], 'g-', label='S2->S3', marker='s', markersize=3)
    plt.plot(history['iter'], history['ar_s1_s3'], 'r-', label='S1->S3', marker='^', markersize=3, linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Autoregressive Generation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1.05)
    
    # 3. 组合能力差距
    plt.subplot(2, 2, 3)
    if len(history['ar_s1_s2']) > 0:
        basic_avg = [(h1 + h2) / 2 for h1, h2 in zip(history['ar_s1_s2'], history['ar_s2_s3'])]
        composition_gap = [b - c for b, c in zip(basic_avg, history['ar_s1_s3'])]
        plt.plot(history['iter'], composition_gap, 'purple', linewidth=2)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        plt.xlabel('Iteration')
        plt.ylabel('Gap')
        plt.title('Composition Gap (Basic Avg - S1->S3)')
        plt.grid(True)
    
    # 4. S1->S3准确率单独展示
    plt.subplot(2, 2, 4)
    plt.plot(history['iter'], history['ar_s1_s3'], 'r-', linewidth=2, marker='o', markersize=4)
    plt.axhline(y=1.0, color='g', linestyle='--', alpha=0.3, label='Perfect')
    plt.axhline(y=0.9, color='orange', linestyle='--', alpha=0.3, label='90%')
    plt.axhline(y=0.8, color='red', linestyle='--', alpha=0.3, label='80%')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('S1->S3 Composition Performance')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'training_results.png'), dpi=150)
    plt.close()

def print_final_summary(history, logger):
    """打印最终总结"""
    print(f"\n{'='*60}")
    print("Training Complete!")
    
    if not history['ar_s1_s2']:
        print("No evaluation data available.")
        return
    
    print(f"\nFinal Results:")
    print(f"  S1->S2: {history['ar_s1_s2'][-1]:.2%}")
    print(f"  S2->S3: {history['ar_s2_s3'][-1]:.2%}")
    print(f"  S1->S3: {history['ar_s1_s3'][-1]:.2%} ← Composition Performance")
    
    # 分析
    basic_performance = (history['ar_s1_s2'][-1] + history['ar_s2_s3'][-1]) / 2
    composition_performance = history['ar_s1_s3'][-1]
    
    print(f"\nComposition Analysis:")
    print(f"  Basic Path Average: {basic_performance:.2%}")
    print(f"  Composition Performance: {composition_performance:.2%}")
    print(f"  Composition Gap: {abs(basic_performance - composition_performance):.2%}")
    
    if composition_performance > 0.8:
        print("\n✅ Model demonstrates STRONG compositional generalization!")
        logger.info("RESULT: Strong compositional generalization observed")
    elif composition_performance > 0.5:
        print("\n🔶 Model shows moderate compositional ability.")
        logger.info("RESULT: Moderate compositional ability")
    else:
        print("\n⚠️  Model shows limited compositional generalization.")
        logger.info("RESULT: Limited compositional generalization")

if __name__ == "__main__":
    main()