import os
import argparse
import yaml
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.dataset import SRDataset
from utils.metrics import psnr_y
from PIL import Image
from models.srcnn import SRCNN
from models.fsrcnn import FSRCNN
from models.espcn import ESPCN
from models.edsr import EDSR
from models.imdn import IMDN
import matplotlib.pyplot as plt

# 设置随机种子
# seed = 3407
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic = True

# 模型字典，方便通过名称选择模型
MODELS = {'srcnn': SRCNN, 'fsrcnn': FSRCNN, 'espcn': ESPCN, 'edsr': EDSR, 'imdn': IMDN}

def setup_from_yaml():
    """
    从 YAML 配置文件中加载参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    args = parser.parse_args()
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    setattr(args, kk, vv)
            else:
                setattr(args, k, v)
    return args

def save_ckpt(model, path):
    """
    保存模型检查点
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({'model': model.state_dict()}, path)

def evaluate(model, loader, device, scale, crit):
    """
    在验证集上评估模型性能
    """
    model.eval()
    losses = []
    scores = []
    with torch.no_grad():
        for batch in loader:
            lr, hr = batch['lr'].to(device), batch['hr'].to(device)
            sr = model(lr)
            loss = crit(sr, hr)
            losses.append(loss.item())
            to_uint8 = lambda t: (t.clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255.0).round().astype('uint8')
            scores.append(psnr_y(Image.fromarray(to_uint8(sr[0])), Image.fromarray(to_uint8(hr[0])), shave=scale))
    model.train()
    loss_avg = sum(losses) / len(losses) if losses else 0.0
    score_avg = sum(scores) / len(scores) if scores else 0.0
    return loss_avg, score_avg

def log_line(fp, text):
    """
    打印并记录日志
    """
    print(text)
    fp.write(text + '\n')
    fp.flush()

def plot_metrics(args, train_loss, train_psnr, val_loss, val_psnr, save_dir):
    """
    绘制训练和验证的 Loss 和 PSNR 曲线
    """
    model_name = f"{args.model.upper()}_x{args.scale}"
    plt.figure(figsize=(12, 4))
    # 绘制 Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Train Loss', color='blue')
    plt.plot(val_loss, label='Validation Loss', color='orange')
    plt.title(f'{model_name}_Loss Curve', fontsize=16)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.6, alpha=0.7)
    # 绘制 PSNR 曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_psnr, label='Train PSNR', color='green')
    plt.plot(val_psnr, label='Validation PSNR', color='red')
    plt.title(f'{model_name}_PSNR Curve', fontsize=16)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('PSNR (dB)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.6, alpha=0.7)
    # 调整布局并保存图片
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'metrics.png')
    plt.savefig(save_path)
    plt.close()

def main():
    """
    主函数，负责训练和验证流程
    """
    args = setup_from_yaml()
    # 设置设备为 GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device(f'cuda:3')
    torch.cuda.set_device(device)

    # 加载训练集和验证集
    train_set = SRDataset(args.model, args.train_dir, scale=args.scale, patch_size=args.patch_size, augment=True, is_train=True)
    val_set = SRDataset(args.model, args.val_dir, scale=args.scale, is_train=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1)

    # 初始化模型
    model = {
        'srcnn': SRCNN(in_channels=3),
        'fsrcnn': FSRCNN(in_channels=3, scale=args.scale),
        'espcn': ESPCN(in_channels=3, scale=args.scale),
        'edsr': EDSR(in_channels=3, scale=args.scale, n_feats=64, n_resblocks=16),
        'imdn': IMDN(in_channels=3, scale=args.scale, num_features=50, num_blocks=6, distillation_rate=0.25, act_slope=0.05)
    }[args.model].to(device)

    # 初始化优化器
    if args.model == 'srcnn':
        opt = {
            'SGD': torch.optim.SGD([
                {"params": model.features.parameters()},
                {"params": model.map.parameters()},
                {"params": model.reconstruction.parameters(), "lr": float(args.lr) * 0.1}],
                lr=float(args.lr), momentum=0.9, weight_decay=1e-4),
            'Adam': torch.optim.Adam([
                {"params": model.features.parameters()},
                {"params": model.map.parameters()},
                {"params": model.reconstruction.parameters(), "lr": float(args.lr) * 0.1}], 
                lr=float(args.lr), betas=(0.9, 0.999))
        }[args.opt]
    else:
        opt = {
            'SGD': torch.optim.SGD(model.parameters(), lr=float(args.lr), momentum=0.9, weight_decay=1e-4),
            'Adam': torch.optim.Adam(model.parameters(), lr=float(args.lr), betas=(0.9, 0.999))
        }[args.opt]

    # 损失函数
    crit = {
        'L1': nn.L1Loss(),
        'MSE': nn.MSELoss()
    }[args.crit]

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=10, min_lr=1e-6)

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    log_path = os.path.join(args.save_dir, f"{args.model}_x{args.scale}.log")
    
    # 记录训练和验证的历史数据
    train_loss_history = []
    train_psnr_history = []
    val_loss_history = []
    val_psnr_history = []

    with open(log_path, 'w', encoding='utf-8') as fp:
        # 记录超参数信息
        log_line(fp, "====================")
        log_line(fp, f"Model: {args.model.upper()}")
        log_line(fp, f"Scale: x{args.scale}")
        log_line(fp, f"Epochs: {args.epochs}")
        log_line(fp, f"Batch Size: {args.batch_size}")
        log_line(fp, f"Patch Size: {args.patch_size}")
        log_line(fp, f"LR: {args.lr}")
        log_line(fp, f"Optimizer: {args.opt}")
        log_line(fp, f"Criterion: {args.crit}")
        log_line(fp, "====================")
        best = 0.0

        # 开始训练
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            total_loss = 0.0
            total_psnr = 0.0
            n = 0
            
            for b in train_loader:
                lr, hr = b['lr'].to(device), b['hr'].to(device)
                batch_size = lr.size(0)
                opt.zero_grad(set_to_none=True)
                sr = model(lr)
                n += batch_size

                loss = crit(sr, hr)
                loss.backward()
                opt.step()
                total_loss += loss.item() * batch_size
                
                to_uint8 = lambda t: (t.clamp(0, 1).cpu().detach().permute(1, 2, 0).numpy() * 255.0).round().astype('uint8')
                batch_psnr = 0.0
                for i in range(batch_size): 
                    psnry = psnr_y(
                        Image.fromarray(to_uint8(sr[i])),
                        Image.fromarray(to_uint8(hr[i])),
                        shave=args.scale
                    )
                    batch_psnr += psnry
                total_psnr += batch_psnr
                
            # 计算平均训练损失和 PSNR
            train_loss = total_loss / max(1, n)
            train_psnr = total_psnr / max(1, n)
            val_loss, val_psnr = evaluate(model, val_loader, device, args.scale, crit)

            # 更新学习率
            scheduler.step(val_psnr)

            # 记录历史数据
            train_loss_history.append(train_loss)
            train_psnr_history.append(train_psnr)
            val_loss_history.append(val_loss)
            val_psnr_history.append(val_psnr)

            dt = time.time() - t0
            log_line(fp, f"[Epoch {epoch:03d}] train_loss={train_loss:.4f} | train_psnr={train_psnr:.2f} dB | val_loss={val_loss:.4f} | val_psnr={val_psnr:.2f} dB | time={dt:.1f}s")
            
            # 保存最佳模型
            if val_psnr > best:
                best = val_psnr
                save_ckpt(model, os.path.join(args.save_dir, 'best.pt'))
        
        # 记录最佳 PSNR
        log_line(fp, f"\nBest PSNR: {best:.2f} dB")
        plot_metrics(args, train_loss_history, train_psnr_history, val_loss_history, val_psnr_history, args.save_dir)
        log_line(fp, f"Training complete. Metrics plot saved to {args.save_dir}/metrics.png")


if __name__ == '__main__':
    main()
