'''
python data_pipeline/retrieval_second.py --data_path ./data/processed/Agriculture/agriculture_generated_all.csv --output_dir ./data/processed/Agriculture/retrieval --normalize_ts --k 10
'''
import pandas as pd
import numpy as np
from dtaidistance import dtw
import pickle
import time
import argparse
from typing import Dict, List, Set



class UnionWeightedRetriever:
    """
    策略2: 三视角并集 + 加权打分 + 序列相似度
    """

    def __init__(self, historical_df, config=None):
        """
        Args:
            historical_df: 历史样本库（训练集）
            config: 配置字典
        """
        self.historical_pool = historical_df.copy()

        # 默认配置
        self.config = {
            "k": 10,
            "enable_time_filter": True,
            "sequence_metric": "dtw",
            "normalize_ts": False,

            # 视角权重（文本匹配）
            "w_temporal": 0.3,
            "w_background": 0.4,
            "w_statistical": 0.3,

            # 综合得分权重
            "alpha": 0.4,  # 文本得分权重
            "beta": 0.6,  # 序列相似度权重

            # 背景视角配置
            "background_top_n": 50,  # 背景视角取Top-N
        }

        if config:
            self.config.update(config)

        print(f"=" * 60)
        print(f"初始化并集加权检索器 (策略2)")
        print(f"=" * 60)
        print(f"  历史库大小: {len(self.historical_pool)}")
        print(f"  Top-K: {self.config['k']}")
        print(f"  时间过滤: {self.config['enable_time_filter']}")
        print(f"    规则: 候选 window_end < 查询 window_end")
        print(f"  序列度量: {self.config['sequence_metric']}")
        print(f"  标准化: {self.config['normalize_ts']}")
        print(f"\n  视角权重:")
        print(f"    时间:  {self.config['w_temporal']:.2f}")
        print(f"    背景: {self.config['w_background']:.2f}")
        print(f"    统计: {self.config['w_statistical']:.2f}")
        print(f"\n  综合得分权重:")
        print(f"    文本: {self.config['alpha']:.2f}")
        print(f"    序列: {self.config['beta']:.2f}")
        print(f"=" * 60)

    def normalize_sequence(self, seq):
        """z-score标准化"""
        seq = np.array(seq, dtype=float)
        mean = np.mean(seq)
        std = np.std(seq)

        if std < 1e-8:
            return seq - mean

        return (seq - mean) / std

    def compute_sequence_similarity(self, x1, x2):
        """计算序列相似度 (DTW)"""
        try:
            seq1 = np.array(eval(x1) if isinstance(x1, str) else x1, dtype=float)
            seq2 = np.array(eval(x2) if isinstance(x2, str) else x2, dtype=float)

            if len(seq1) != len(seq2):
                return 0.0

            if self.config['normalize_ts']:
                seq1 = self.normalize_sequence(seq1)
                seq2 = self.normalize_sequence(seq2)

            distance = dtw.distance(seq1, seq2)
            similarity = 1.0 / (1.0 + distance)

            return similarity

        except Exception as e:
            return 0.0

    def filter_by_time(self, pool, query_sample):
        """时间过滤"""
        if not self.config['enable_time_filter']:
            return pool

        query_end = pd.to_datetime(query_sample['window_end'])
        filtered = pool[pd.to_datetime(pool['window_end']) < query_end]

        return filtered

    def compute_jaccard_similarity(self, set1_str, set2_str):
        """
        计算Jaccard相似度

        Args:
            set1_str: "2,4" 或 ""
            set2_str: "1,2,4" 或 ""

        Returns:
            float:  Jaccard相似度 (0-1)
        """
        if pd.isna(set1_str) or set1_str == '':
            return 0.0

        if pd.isna(set2_str) or set2_str == '':
            return 0.0

        set1 = set(set1_str.split(','))
        set2 = set(set2_str.split(','))

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        if union == 0:
            return 0.0

        return intersection / union

    def retrieve_temporal(self, query_sample, pool):
        """
        时间视角检索

        Returns:
            dict: {sample_id: match_score}
        """
        query_pattern = query_sample['temporal_pattern']

        matched = pool[pool['temporal_pattern'] == query_pattern]

        # 匹配的样本得分为1.0
        result = {sid: 1.0 for sid in matched['sample_id'].values}

        return result

    def retrieve_background(self, query_sample, pool):
        """
        背景视角检索（返回Top-N个Jaccard相似度最高的）

        Returns:
            dict: {sample_id: jaccard_similarity}
        """
        query_patterns = query_sample['background_patterns']

        if pd.isna(query_patterns) or query_patterns == '':
            return {}

        similarities = []

        for idx, row in pool.iterrows():
            sim = self.compute_jaccard_similarity(
                query_patterns,
                row['background_patterns']
            )

            if sim > 0:  # 只保留有交集的
                similarities.append({
                    'sample_id': row['sample_id'],
                    'similarity': sim
                })

        # 按相似度排序，取Top-N
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        top_n = similarities[:self.config['background_top_n']]

        result = {item['sample_id']: item['similarity'] for item in top_n}

        return result

    def retrieve_statistical(self, query_sample, pool):
        """
        统计视角检索

        Returns:
            dict: {sample_id: match_score}
        """
        query_pattern = query_sample['statistical_pattern']

        matched = pool[pool['statistical_pattern'] == query_pattern]

        result = {sid: 1.0 for sid in matched['sample_id'].values}

        return result

    def get_matched_perspectives(self, sample_id, temp_dict, back_dict, stat_dict):
        """判断样本匹配了哪些视角"""
        perspectives = []

        if sample_id in temp_dict:
            perspectives.append(1)

        if sample_id in back_dict:
            perspectives.append(2)

        if sample_id in stat_dict:
            perspectives.append(3)

        return perspectives

    def retrieve(self, query_sample):
        """
        单样本检索

        Returns:
            dict
        """
        start_time = time.time()

        query_id = query_sample['sample_id']

        # ===== Step 0: 时间过滤 =====
        pool = self.filter_by_time(self.historical_pool, query_sample)

        # 排除自己
        pool = pool[pool['sample_id'] != query_id]

        if len(pool) == 0:
            return self._empty_result(query_sample, 0)

        # ===== Step 1: 三视角检索 =====

        temp_dict = self.retrieve_temporal(query_sample, pool)
        back_dict = self.retrieve_background(query_sample, pool)
        stat_dict = self.retrieve_statistical(query_sample, pool)

        # ===== Step 2: 取并集 =====

        all_candidate_ids = set(temp_dict.keys()) | set(back_dict.keys()) | set(stat_dict.keys())

        if len(all_candidate_ids) == 0:
            return self._empty_result(query_sample, len(pool), len(temp_dict), len(back_dict), len(stat_dict))

        # ===== Step 3: 计算综合得分 =====

        candidates = []

        # 提取候选样本数据
        candidate_df = pool[pool['sample_id'].isin(all_candidate_ids)]

        w_temp = self.config['w_temporal']
        w_back = self.config['w_background']
        w_stat = self.config['w_statistical']
        alpha = self.config['alpha']
        beta = self.config['beta']

        for idx, row in candidate_df.iterrows():
            sample_id = row['sample_id']

            # 3.1 文本匹配得分
            temp_match = temp_dict.get(sample_id, 0.0)
            back_match = back_dict.get(sample_id, 0.0)
            stat_match = stat_dict.get(sample_id, 0.0)

            text_score = (
                    w_temp * temp_match +
                    w_back * back_match +
                    w_stat * stat_match
            )

            # 3.2 序列相似度
            seq_score = self.compute_sequence_similarity(
                query_sample['x_ot'],
                row['x_ot']
            )

            # 3.3 综合得分
            final_score = alpha * text_score + beta * seq_score

            # 3.4 匹配的视角
            matched_perspectives = self.get_matched_perspectives(
                sample_id,
                temp_dict,
                back_dict,
                stat_dict
            )

            candidates.append({
                'sample_id': sample_id,
                'text_score': text_score,
                'sequence_similarity': seq_score,
                'final_score': final_score,
                'matched_perspectives': matched_perspectives,

                # 分项得分（用于分析）
                'temporal_match': temp_match,
                'background_match': back_match,
                'statistical_match': stat_match
            })

        # ===== Step 4: 排序返回Top-K =====

        candidates.sort(key=lambda x: x['final_score'], reverse=True)

        k = self.config['k']
        top_k = candidates[:k]

        # 添加rank
        for rank, item in enumerate(top_k, start=1):
            item['rank'] = rank

        # ===== 返回结果 =====

        result = {
            "query_info": {
                "window_start": query_sample['window_start'],
                "window_end": query_sample['window_end'],
                "temporal_pattern": query_sample['temporal_pattern'],
                "statistical_pattern": query_sample['statistical_pattern'],
                "background_patterns": query_sample['background_patterns']
            },

            "retrieved_samples": top_k,

            "metadata": {
                "filtered_pool_size": len(pool),
                "temporal_candidates": len(temp_dict),
                "statistical_candidates": len(stat_dict),
                "background_candidates": len(back_dict),
                "union_size": len(all_candidate_ids),
                "n_retrieved": len(top_k),
                "retrieval_time": time.time() - start_time
            }
        }

        return result

    def _empty_result(self, query_sample, pool_size=0, temp_size=0, back_size=0, stat_size=0):
        """构造空结果"""
        return {
            "query_info": {
                "window_start": query_sample['window_start'],
                "window_end": query_sample['window_end'],
                "temporal_pattern": query_sample['temporal_pattern'],
                "statistical_pattern": query_sample['statistical_pattern'],
                "background_patterns": query_sample['background_patterns']
            },

            "retrieved_samples": [],

            "metadata": {
                "filtered_pool_size": pool_size,
                "temporal_candidates": temp_size,
                "statistical_candidates": stat_size,
                "background_candidates": back_size,
                "union_size": 0,
                "n_retrieved": 0,
                "retrieval_time": 0
            }
        }

    def retrieve_batch(self, query_df, save_path=None):
        """批量检索"""
        results = {}

        print(f"\n{'=' * 60}")
        print(f"开始批量检索:  {len(query_df)} 个样本")
        print(f"{'=' * 60}\n")

        for idx, query_sample in query_df.iterrows():
            query_id = query_sample['sample_id']

            result = self.retrieve(query_sample)
            results[query_id] = result

            if (idx + 1) % 50 == 0:
                print(f"  进度: {idx + 1}/{len(query_df)}")

        print(f"\n{'=' * 60}")
        print(f"✓ 检索完成")
        print(f"{'=' * 60}")

        self._print_statistics(results)

        if save_path:
            with open(save_path, 'wb') as f:
                pickle.dump(results, f)
            print(f"\n✓ 已保存:  {save_path}")

        return results

    def _print_statistics(self, results):
        """打印统计信息"""
        n_total = len(results)
        n_empty = sum(1 for r in results.values() if r['metadata']['n_retrieved'] == 0)
        n_retrieved_list = [r['metadata']['n_retrieved'] for r in results.values()]
        union_sizes = [r['metadata']['union_size'] for r in results.values()]

        print(f"\n检索统计:")
        print(f"  总样本数:  {n_total}")
        print(f"  检索为空: {n_empty} ({n_empty / n_total * 100:.1f}%)")
        print(f"  平均检索数: {np.mean(n_retrieved_list):.2f}")
        print(f"  最少检索数: {np.min(n_retrieved_list)}")
        print(f"  最多检索数: {np.max(n_retrieved_list)}")

        print(f"\n并集统计:")
        print(f"  平均并集大小: {np.mean(union_sizes):.2f}")
        print(f"  最小并集:  {np.min(union_sizes)}")
        print(f"  最大并集: {np.max(union_sizes)}")

        # 综合得分统计
        all_final_scores = []
        all_text_scores = []
        all_seq_scores = []

        for r in results.values():
            for sample in r['retrieved_samples']:
                all_final_scores.append(sample['final_score'])
                all_text_scores.append(sample['text_score'])
                all_seq_scores.append(sample['sequence_similarity'])

        if all_final_scores:
            print(f"\n得分统计:")
            print(
                f"  综合得分: 均值={np.mean(all_final_scores):.4f}, 范围=[{np.min(all_final_scores):.4f}, {np.max(all_final_scores):.4f}]")
            print(
                f"  文本得分: 均值={np.mean(all_text_scores):.4f}, 范围=[{np.min(all_text_scores):.4f}, {np.max(all_text_scores):.4f}]")
            print(
                f"  序列相似度: 均值={np.mean(all_seq_scores):.4f}, 范围=[{np.min(all_seq_scores):.4f}, {np.max(all_seq_scores):.4f}]")


def save_statistics(results, output_path):
    """保存统计报告"""
    stats = []

    for query_id, result in results.items():
        avg_final = np.mean([s['final_score'] for s in result['retrieved_samples']]) if result[
            'retrieved_samples'] else 0
        avg_text = np.mean([s['text_score'] for s in result['retrieved_samples']]) if result['retrieved_samples'] else 0
        avg_seq = np.mean([s['sequence_similarity'] for s in result['retrieved_samples']]) if result[
            'retrieved_samples'] else 0

        stats.append({
            'query_sample_id': query_id,
            'n_retrieved': result['metadata']['n_retrieved'],
            'union_size': result['metadata']['union_size'],
            'temporal_candidates': result['metadata']['temporal_candidates'],
            'background_candidates': result['metadata']['background_candidates'],
            'statistical_candidates': result['metadata']['statistical_candidates'],
            'avg_final_score': avg_final,
            'avg_text_score': avg_text,
            'avg_sequence_similarity': avg_seq,
            'retrieval_time': result['metadata']['retrieval_time']
        })

    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(output_path, index=False)
    print(f"  ✓ 统计报告已保存: {output_path}")


def main(args):
    """主函数"""

    print("加载数据集...")
    df = pd.read_csv(args.data_path)

    print(f"\n数据集概览:")
    print(f"  训练集: {len(df[df['split'] == 'train'])} 样本")
    print(f"  验证集: {len(df[df['split'] == 'val'])} 样本")
    print(f"  测试集: {len(df[df['split'] == 'test'])} 样本")

    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']
    test_df = df[df['split'] == 'test']

    config = {
        "k": args.k,
        "enable_time_filter": True,
        "sequence_metric": "dtw",
        "normalize_ts": args.normalize_ts,

        # 视角权重
        "w_temporal": args.w_temporal,
        "w_background": args.w_background,
        "w_statistical": args.w_statistical,

        # 综合得分权重
        "alpha": args.alpha,
        "beta": args.beta,

        "background_top_n": args.background_top_n,
    }

    retriever = UnionWeightedRetriever(train_df, config)

    if args.test_single:
        print("\n" + "=" * 60)
        print("单样本检索测试")
        print("=" * 60)

        test_sample = test_df.iloc[0]

        print(f"\n查询样本:  {test_sample['sample_id']}")
        print(f"  时间窗口: {test_sample['window_start']} ~ {test_sample['window_end']}")
        print(f"  时间模式: {test_sample['temporal_pattern']}")
        print(f"  统计模式:  {test_sample['statistical_pattern']}")
        print(f"  背景模式: {test_sample['background_patterns']}")

        result = retriever.retrieve(test_sample)

        print(f"\n检索结果:")
        print(f"  时间过滤后历史库大小: {result['metadata']['filtered_pool_size']}")
        print(f"  时间候选:  {result['metadata']['temporal_candidates']}")
        print(f"  背景候选: {result['metadata']['background_candidates']}")
        print(f"  统计候选: {result['metadata']['statistical_candidates']}")
        print(f"  并集大小: {result['metadata']['union_size']}")
        print(f"  检索数量: {result['metadata']['n_retrieved']}")
        print(f"  检索耗时: {result['metadata']['retrieval_time']:.4f} 秒")

        if result['retrieved_samples']:
            print(f"\nTop-5 样本:")
            for item in result['retrieved_samples'][: 5]:
                perspectives_str = ','.join(map(str, item['matched_perspectives']))
                print(f"\n  Rank {item['rank']}: {item['sample_id']}")
                print(f"    综合得分: {item['final_score']:.4f}")
                print(
                    f"    文本得分: {item['text_score']:.4f} (时间:{item['temporal_match']:.2f}, 背景:{item['background_match']:.2f}, 统计:{item['statistical_match']:.2f})")
                print(f"    序列相似度: {item['sequence_similarity']:.4f}")
                print(f"    匹配视角: [{perspectives_str}]")

    if args.retrieve_test:
        print("\n" + "=" * 60)
        print("批量检索 - 测试集")
        print("=" * 60)

        test_results = retriever.retrieve_batch(
            test_df,
            save_path=args.output_dir + '/retrieval_results_test_union.pkl'
        )

        save_statistics(
            test_results,
            args.output_dir + '/retrieval_statistics_test_union.csv'
        )

    if args.retrieve_train:
        print("\n" + "=" * 60)
        print("批量检索 - 训练集")
        print("=" * 60)

        train_results = retriever.retrieve_batch(
            train_df,
            save_path=args.output_dir + '/retrieval_results_train_union.pkl'
        )

        save_statistics(
            train_results,
            args.output_dir + '/retrieval_statistics_train_union.csv'
        )

    if args.retrieve_val and len(val_df) > 0:
        print("\n" + "=" * 60)
        print("批量检索 - 验证集")
        print("=" * 60)

        val_results = retriever.retrieve_batch(
            val_df,
            save_path=args.output_dir + '/retrieval_results_val_union.pkl'
        )

        save_statistics(
            val_results,
            args.output_dir + '/retrieval_statistics_val_union.csv'
        )

    print("\n" + "=" * 60)
    print("全部完成！")
    print("=" * 60)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='时间序列检索 - 策略2: 并集+加权打分+DTW')

    parser.add_argument("--data_path", type=str, required=True,
                        help="输入CSV文件路径")
    parser.add_argument("--output_dir", type=str, default='.',
                        help="输出目录")

    parser.add_argument("--k", type=int, default=10,
                        help="返回Top-K个样本")
    parser.add_argument("--normalize_ts", action="store_true",
                        help="在计算距离前对时间序列做z-score标准化")

    # 视角权重
    parser.add_argument("--w_temporal", type=float, default=0.5,
                        help="时间视角权重")
    parser.add_argument("--w_background", type=float, default=0.2,
                        help="背景视角权重")
    parser.add_argument("--w_statistical", type=float, default=0.3,
                        help="统计视角权重")

    # 综合得分权重
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="文本得分权重")
    parser.add_argument("--beta", type=float, default=0.9,
                        help="序列相似度权重")

    # 背景视角配置
    parser.add_argument("--background_top_n", type=int, default=50,
                        help="背景视角取Top-N个")

    # 控制选项
    parser.add_argument("--test_single", action="store_true",
                        help="运行单样本测试")
    parser.add_argument("--retrieve_test", action="store_true",
                        help="检索测试集")
    parser.add_argument("--retrieve_train", action="store_true",
                        help="检索训练集")
    parser.add_argument("--retrieve_val", action="store_true",
                        help="检索验证集")

    args = parser.parse_args()

    if not (args.retrieve_test or args.retrieve_train or args.retrieve_val):
        args.retrieve_test = True
        args.retrieve_train = True
        args.retrieve_val = True

    main(args)
