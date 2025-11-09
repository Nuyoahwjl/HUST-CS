import os
import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Argument parser setup
    p = argparse.ArgumentParser()
    p.add_argument('json_files', nargs='+')
    p.add_argument('--out_dir', default='report')
    args = p.parse_args()

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Load JSON files and create DataFrame
    rows = [json.load(open(j, 'r')) for j in args.json_files]
    df = pd.DataFrame(rows).sort_values(['scale', 'model']).reset_index(drop=True)
    df.to_csv(os.path.join(args.out_dir, 'summary.csv'), index=False)

    # Generate bar plot for PSNR and SSIM
    plt.figure()
    df.plot(x='model', y=['psnr_y', 'ssim_y'], kind='bar')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'bars_psnr_ssim.png'))
    plt.close()

    # Generate scatter plot for PSNR vs FPS
    plt.figure()
    plt.scatter(df['fps'], df['psnr_y'])
    for i, txt in enumerate(df['model']):
        plt.annotate(txt, (df['fps'][i], df['psnr_y'][i]))
    plt.xlabel('FPS')
    plt.ylabel('PSNR (Y)')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'scatter_psnr_vs_fps.png'))
    plt.close()

    # Write summary markdown file
    with open(os.path.join(args.out_dir, 'summary.md'), 'w', encoding='utf-8') as f:
        f.write('# 超分辨率对比报告\n\n')
        f.write('## 指标汇总表\n\n')
        f.write(df.to_markdown(index=False))
        f.write('\n\n')
        f.write('## 可视化\n')
        f.write('- PSNR/SSIM 柱状图：bars_psnr_ssim.png\n')
        f.write('- PSNR vs FPS 散点图：scatter_psnr_vs_fps.png\n')

if __name__ == '__main__':
    main()
