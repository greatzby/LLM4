# train_composition_universal.py
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
    parser = argparse.ArgumentParser(description='Universal Composition Training')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--n_layer', type=int, default=1)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--n_embd', type=int, default=120)
    parser.add_argument('--max_iters', type=int, default=50000)
    parser.add_argument('--test_interval', type=int, default=1000)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--checkpoint_interval', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval_temperature', type=float, default=0.1)
    parser.add_argument('--eval_top_k', type=int, default=10)
    return parser.parse_args()

@torch.no_grad()
def evaluate_ar_universal(model, test_file, stages, stoi, itos, device, G, 
                         temperature=0.1, top_k=10, max_eval_per_type=50):
    """通用的AR评估函数（自动检测编码类型）"""
    model.eval()
    
    S1, S2, S3 = stages
    
    # 检测编码类型
    vocab_size = len(stoi)
    is_token_level = vocab_size > 50  # Token级通常有90+词汇
    
    print(f"  Detected encoding: {'Token-level' if is_token_level else 'Character-level'} (vocab_size={vocab_size})")
    
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
            
            # 构建prompt
            if is_token_level:
                # Token级编码
                prompt = f"{source} {target} {source}"
                prompt_tokens = prompt.split()
                prompt_ids = []
                for token in prompt_tokens:
                    if token in stoi:
                        prompt_ids.append(stoi[token])
            else:
                # 字符级编码
                prompt_str = f"{source} {target}"
                prompt_ids = []
                for char in prompt_str:
                    if char in stoi:
                        prompt_ids.append(stoi[char])
            
            x = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
            
            # 生成
            with torch.no_grad():
                y = model.generate(x, max_new_tokens=50, temperature=temperature, top_k=top_k)
            
            # 解码
            full_sequence = y[0].tolist()
            
            if is_token_level:
                # Token级解码
                generated_path = []
                for i, token_id in enumerate(full_sequence):
                    if token_id == 1:  # EOS
                        break
                    if token_id in itos and i >= len(prompt_tokens):  # 跳过prompt
                        try:
                            num = int(itos[token_id])
                            generated_path.append(num)
                        except:
                            pass
            else:
                # 字符级解码
                full_chars = []
                for token_id in full_sequence:
                    if token_id == 1:  # newline
                        break
                    if token_id in itos and token_id > 1:
                        full_chars.append(itos[token_id])
                
                # 解析数字
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
                
                # 提取路径
                if len(all_numbers) >= 3:
                    generated_path = all_numbers[2:]
                else:
                    generated_path = []
            
            # 验证结果
            success = False
            error_type = None
            
            if len(generated_path) >= 2:
                if generated_path[0] == source and generated_path[-1] == target:
                    if path_type == 'S1->S3':
                        has_s2 = any(node in S2 for node in generated_path[1:-1])
                        if has_s2:
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
                        'true': true_path
                    })
            else:
                results[path_type]['errors'][error_type if error_type else 'unknown'] += 1
                if len(results[path_type]['examples']['failure']) < 3:
                    results[path_type]['examples']['failure'].append({
                        'source': source,
                        'target': target,
                        'generated': generated_path,
                        'error': error_type,
                        'true': true_path
                    })
        
        results[path_type]['accuracy'] = results[path_type]['correct'] / results[path_type]['total'] if results[path_type]['total'] > 0 else 0.0
    
    model.train()
    return results

def main():
    args = parse_args()
    
    # 设置种子
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    encoding_type = "token" if "improved" in args.data_dir or "composition_90" in args.data_dir else "char"
    out_dir = f'out/composition_{encoding_type}_{timestamp}'
    os.makedirs(out_dir, exist_ok=True)
    
    # 设置logger
    logger = get_logger(os.path.join(out_dir, "train.log"))
    
    print("="*60)
    print(f"Universal Composition Training")
    print(f"Model: {args.n_layer}L-{args.n_head}H-{args.n_embd}D")
    print(f"Data: {args.data_dir}")
    print("="*60)
    
    # 加载数据和元信息
    data_dir = args.data_dir
    
    # 加载阶段信息
    with open(os.path.join(data_dir, 'stage_info.pkl'), 'rb') as f:
        stage_info = pickle.load(f)
    stages = stage_info['stages']
    
    # 加载元信息
    with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    
    stoi, itos = meta['stoi'], meta['itos']
    block_size = meta['block_size']
    vocab_size = meta['vocab_size']
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Encoding type: {'Token-level' if vocab_size > 50 else 'Character-level'}")
    
    # 加载图
    graph_path = os.path.join(data_dir, 'composition_graph.graphml')
    if not os.path.exists(graph_path):
        # 尝试父目录
        graph_path = os.path.join(os.path.dirname(data_dir), 'composition_graph.graphml')
    G = nx.read_graphml(graph_path)
    
    # 找到正确的训练数据文件
    train_file_candidates = [
        f'train_10.bin',
        f'train.bin',
        'train_10.bin'
    ]
    
    train_data = None
    for candidate in train_file_candidates:
        train_path = os.path.join(data_dir, candidate)
        if os.path.exists(train_path):
            train_data = np.memmap(train_path, dtype=np.uint16, mode='r')
            print(f"Using training data: {candidate}")
            break
    
    if train_data is None:
        raise FileNotFoundError(f"No training data found in {data_dir}")
    
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    test_file = os.path.join(data_dir, 'test.txt')
    
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
        'ar_s1_s3': []
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
            
            # AR评估（使用通用函数）
            ar_results = evaluate_ar_universal(
                model, test_file, stages, stoi, itos, args.device, G,
                temperature=args.eval_temperature, top_k=args.eval_top_k
            )
            
            # 记录和打印结果
            history['iter'].append(iter_num)
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            history['ar_s1_s2'].append(ar_results['S1->S2']['accuracy'])
            history['ar_s2_s3'].append(ar_results['S2->S3']['accuracy'])
            history['ar_s1_s3'].append(ar_results['S1->S3']['accuracy'])
            
            print(f"\n{'='*60}")
            print(f"Iteration {iter_num}:")
            print(f"  Loss: train={avg_train_loss:.4f}, val={val_loss:.4f}")
            
            print(f"\n  Autoregressive Generation:")
            for path_type in ['S1->S2', 'S2->S3', 'S1->S3']:
                res = ar_results[path_type]
                print(f"    {path_type}: {res['accuracy']:.2%} ({res['correct']}/{res['total']})")
                
                if res['examples']['success'] and iter_num <= 5000:
                    ex = res['examples']['success'][0]
                    print(f"      ✓ {ex['source']}→{ex['target']}: {ex['generated']}")
            
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
            torch.save(checkpoint, os.path.join(out_dir, f'ckpt_{iter_num}.pt'))
        
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
    
    print(f"\nTraining complete! Results saved to: {out_dir}")

if __name__ == "__main__":
    main()