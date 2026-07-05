"""
预处理脚本：将 MIMIC-IV labevents.csv.gz 按 subject_id 分区存储
运行一次后，LabEventsLoader 可以快速按患者查询化验数据。

用法:
    python scripts/build_lab_index.py [--data_root D:/drug/data] [--chunk_size 200000]

输出:
    data/mimic/lab_index/
        ├── manifest.json        # 索引元信息
        ├── subject_0000.pkl     # subject_id 0~999 的化验数据
        ├── subject_1000.pkl     # subject_id 1000~1999 的化验数据
        └── ...
"""
import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# 需要提取的关键化验项目 itemid
KEY_ITEMIDS = {
    50912, 51006, 50920,  # 肾功能: 肌酐, BUN, eGFR
    50861, 50878, 50862, 50885,  # 肝功能: ALT, AST, 白蛋白, 胆红素
    50976, 51237, 50977,  # 凝血: PT, INR, PTT
    50971, 50983, 50902,  # 电解质: 钾, 钠, 氯
    51222, 51265, 51301,  # 血常规: 血红蛋白, 血小板, 白细胞
    50931,  # 血糖
    51003,  # 肌钙蛋白T
}


def build_index(data_root: str, chunk_size: int = 200000):
    """构建按 subject_id 分区的化验数据索引。"""
    data_root = Path(data_root)
    lab_path = data_root / "mimic" / "labevents.csv.gz"
    output_dir = data_root / "mimic" / "lab_index"

    if not lab_path.exists():
        print(f"错误: labevents 文件不存在: {lab_path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"输入: {lab_path}")
    print(f"输出目录: {output_dir}")
    print(f"分块大小: {chunk_size:,}")
    print(f"关键 itemid: {len(KEY_ITEMIDS)} 个")
    print()

    # 按 subject_id 的千位分桶
    buckets = defaultdict(list)
    total_rows = 0
    matched_rows = 0

    print("正在分块读取和分区...")
    chunk_iter = pd.read_csv(
        lab_path,
        compression='gzip',
        chunksize=chunk_size,
        usecols=['subject_id', 'itemid', 'charttime', 'valuenum',
                 'valueuom', 'ref_range_lower', 'ref_range_upper', 'flag'],
        dtype={'subject_id': int, 'itemid': int, 'valuenum': float},
        low_memory=False,
    )

    for chunk_num, chunk in enumerate(chunk_iter, 1):
        total_rows += len(chunk)

        # 过滤关键 itemid
        filtered = chunk[chunk['itemid'].isin(KEY_ITEMIDS)]
        matched_rows += len(filtered)

        # 按 subject_id 分桶（每 1000 个 subject_id 一个文件）
        for sid, group in filtered.groupby('subject_id'):
            bucket_key = (sid // 1000) * 1000
            buckets[bucket_key].append(group)

        if chunk_num % 10 == 0:
            print(f"  已处理 {chunk_num} 块, 总行数: {total_rows:,}, 匹配: {matched_rows:,}")

    print(f"\n读取完成: 总行数 {total_rows:,}, 匹配关键项目 {matched_rows:,}")
    print(f"分桶数: {len(buckets)}")

    # 写入分桶文件
    print("\n正在写入分区文件...")
    manifest = {
        'total_rows': total_rows,
        'matched_rows': matched_rows,
        'key_itemids': sorted(KEY_ITEMIDS),
        'buckets': {},
    }

    for bucket_key, dfs in sorted(buckets.items()):
        bucket_df = pd.concat(dfs, ignore_index=True)
        file_name = f"subject_{bucket_key:06d}.pkl"
        file_path = output_dir / file_name
        bucket_df.to_pickle(str(file_path))

        sids = bucket_df['subject_id'].unique().tolist()
        manifest['buckets'][str(bucket_key)] = {
            'file': file_name,
            'rows': len(bucket_df),
            'subjects': len(sids),
            'subject_range': [int(min(sids)), int(max(sids))],
        }

    # 写入 manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n完成！索引文件已保存到: {output_dir}")
    print(f"  manifest.json: {manifest_path}")
    print(f"  分区文件数: {len(buckets)}")
    print(f"  总行数: {matched_rows:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="构建 MIMIC-IV 化验数据索引")
    parser.add_argument('--data_root', default='D:/drug/data', help='数据根目录')
    parser.add_argument('--chunk_size', type=int, default=200000, help='分块大小')
    args = parser.parse_args()

    build_index(args.data_root, args.chunk_size)
