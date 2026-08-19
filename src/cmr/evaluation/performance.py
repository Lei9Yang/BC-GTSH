import time

import numpy as np


def hammingDist(B1, B2):
    """汉明距离计算辅助函数（补全缺失依赖）"""
    q = B1.shape[1]
    dist = 0.5 * (q - np.dot(B1, B2.T))
    return dist

def _ndcg_at_k(gnd_sorted, k):
    """NDCG辅助函数（补全缺失依赖）"""
    rel = gnd_sorted[:k].astype(float)
    dcg = np.sum(rel / np.log2(np.arange(2, k + 2)))
    idcg = np.sum(np.sort(rel)[::-1] / np.log2(np.arange(2, k + 2)))
    return dcg / (idcg + 1e-12)


# topK = [10, 20, 50, 100]  # 计算Top10/20/50/100的MAP/Precision/Recall
# N = [0.1, 0.5]  # 比例：取检索库前10%、50%样本，计算对应MAP
# ndcgKs = [1000]
# precisionCurveKs = [10, 20, 50, 100, 200, 500, 1000]

def calPerformance(
    queryLabel,
    retrievalLabel,
    queryB,
    retrievalB,
    topK,
    N,
    ndcgKs,
    precisionCurveKs,
    hashcode_generate_time,
    retrieval_start_time,
    train_elapsed_time,
    efficiency=None,
):
    """
    计算多标签哈希检索指标:
    - MAP
    - TopK MAP / Precision / Recall
    - NDCG@K (K 由 ndcgKs 指定)
    - TopN(比例) MAP 【修改点：N为检索集占比，不再是固定个数】
    - PR 曲线(按汉明距离阈值)
    - Top-K Precision 曲线(K 由 precisionCurveKs 指定)
    - P@H<=2
    - Fisher 判别比
    """
    numQuery = queryLabel.shape[0]
    numRetrieval = retrievalLabel.shape[0]  # 新增：检索库总样本数量
    nt = len(topK)
    nn = len(N)
    nd = len(ndcgKs)
    nk = len(precisionCurveKs)

    map_score = 0.0
    topkMap = np.zeros(nt)
    topkPre = np.zeros(nt)
    topkRec = np.zeros(nt)
    ndcg_scores = np.zeros(nd)
    topnMap = np.zeros(nn)  # 原topnPre -> topnMap，存储比例截断下的AP累加
    topk_precision_curve = np.zeros(nk)
    ap_per_query = np.full(numQuery, np.nan, dtype=np.float64)

    p_at_h_le_2 = 0.0
    fisher_ratio_sum = 0.0
    fisher_valid_queries = 0
    valid_query_count = 0

    search_start = time.perf_counter()
    hamm = hammingDist(queryB, retrievalB)
    sorted_indices = np.argsort(hamm, axis=1)
    search_time = time.perf_counter() - search_start
    metric_start = time.perf_counter()
    pr_recall_levels = np.linspace(0.0, 1.0, 101, dtype=np.float64)
    pr_precision_sum = np.zeros_like(pr_recall_levels, dtype=np.float64)

    for i in range(numQuery):
        gnd = (queryLabel[i] @ retrievalLabel.T) > 0
        tsum = np.sum(gnd)
        if tsum == 0:
            continue
        valid_query_count += 1

        sorted_idx = sorted_indices[i]
        gnd_sorted = gnd[sorted_idx]

        tindex = np.where(gnd_sorted == 1)[0] + 1
        count = np.arange(1, len(tindex) + 1)
        ap_i = float(np.mean(count / tindex))
        ap_per_query[i] = ap_i
        map_score += ap_i

        for j in range(nt):
            k = topK[j]
            tgnd = gnd_sorted[:k]
            pos = np.sum(tgnd)
            if pos == 0:
                continue

            tindex_k = np.where(tgnd == 1)[0] + 1
            count_k = np.arange(1, len(tindex_k) + 1)
            topkMap[j] += np.mean(count_k / tindex_k)
            topkPre[j] += pos / k
            topkRec[j] += pos / tsum

        # NDCG@K with independent K settings
        for j in range(nd):
            k = int(ndcgKs[j])
            k = max(1, min(k, len(gnd_sorted)))
            ndcg_scores[j] += _ndcg_at_k(gnd_sorted, k)

        # ========== 核心修改：N为比例，计算前比例检索样本的MAP ==========
        for j in range(nn):
            ratio = N[j]
            # 根据比例计算截断数量，向上取整保证至少取1个
            cut_k = max(1, int(np.ceil(numRetrieval * ratio)))
            cut_k = min(cut_k, numRetrieval)  # 不超过检索库总数
            tgnd_ratio = gnd_sorted[:cut_k]
            pos_ratio = np.sum(tgnd_ratio)
            if pos_ratio == 0:
                continue
            # 计算该截断下的AP并累加
            tindex_ratio = np.where(tgnd_ratio == 1)[0] + 1
            count_ratio = np.arange(1, len(tindex_ratio) + 1)
            ap_ratio = np.mean(count_ratio / tindex_ratio)
            topnMap[j] += ap_ratio

        # 101-point PR curve on fixed recall levels [0.00, 1.00].
        # For each recall level r, use the best precision where recall >= r.
        hit_count = np.cumsum(gnd_sorted.astype(np.float64))
        rank_count = np.arange(1, len(gnd_sorted) + 1, dtype=np.float64)
        precision_by_rank = hit_count / rank_count
        recall_by_rank = hit_count / float(tsum)
        precision_envelope = np.maximum.accumulate(precision_by_rank[::-1])[::-1]
        recall_indices = np.searchsorted(recall_by_rank, pr_recall_levels, side="left")
        valid_levels = recall_indices < len(precision_envelope)
        pr_precision_sum[valid_levels] += precision_envelope[recall_indices[valid_levels]]

        # Top-K precision curve with independent K settings
        for j in range(nk):
            k = int(precisionCurveKs[j])
            k = max(1, min(k, len(gnd_sorted)))
            topk_precision_curve[j] += np.sum(gnd_sorted[:k]) / float(k)

        within = hamm[i, :] <= 2
        if np.any(within):
            p_at_h_le_2 += float(np.mean(gnd[within]))

        pos_dist = hamm[i, gnd]
        neg_dist = hamm[i, ~gnd]
        if pos_dist.size > 0 and neg_dist.size > 0:
            # Fisher ratio in paper: E[||a-n||_2^2] / E[||a-p||_2^2]
            # Here distances are in Hamming space; keep the same expectation form.
            e_neg = np.mean(np.square(neg_dist.astype(np.float64)))
            e_pos = np.mean(np.square(pos_dist.astype(np.float64)))
            fisher_ratio_sum += float(e_neg / (e_pos + 1e-12))
            fisher_valid_queries += 1

    metric_time = time.perf_counter() - metric_start
    calperformance_time = search_time + metric_time
    retrieval_all_time = calperformance_time

    denom = max(valid_query_count, 1)

    efficiency_record = dict(efficiency or {})
    efficiency_record.update({
        'search_time_s': float(search_time),
        'metric_time_s': float(metric_time),
        'evaluation_total_time_s': float(search_time + metric_time),
        'query_count': int(numQuery),
        'database_size': int(numRetrieval),
    })

    result = {
        'map': map_score / denom,
        'map_all_queries': map_score / max(numQuery, 1),
        'query_coverage': valid_query_count / max(numQuery, 1),
        'ap_per_query': ap_per_query,
        'topkMap': topkMap / denom,
        'topkPre': topkPre / denom,
        'topkRec': topkRec / denom,
        'ndcgKs': np.asarray(ndcgKs, dtype=np.int32),
        'ndcgAtKs': ndcg_scores / denom,
        # 修改key：topnMap替代原topnPre，同时保存比例参数
        'ratioNs': np.array(N, dtype=np.float64),
        'topnMap': topnMap / denom,
        'prCurve': {
            'recall_levels': pr_recall_levels,
            'precision': pr_precision_sum / denom,
            'recall': pr_recall_levels,
        },
        'topKPrecisionCurve': {
            'Ks': np.asarray(precisionCurveKs, dtype=np.int32),
            'precision': topk_precision_curve / denom,
        },
        'p_at_h_le_2': p_at_h_le_2 / denom,
        'fisher_ratio': fisher_ratio_sum / fisher_valid_queries if fisher_valid_queries > 0 else 0.0,
        'valid_query_count': int(valid_query_count),
        'total_query_count': int(numQuery),
        'train_elapsed_time': train_elapsed_time,
        'hashcode_generate_time': hashcode_generate_time,
        'calperformance_time': calperformance_time,
        'retrieval_all_time': retrieval_all_time,
        'efficiency': efficiency_record,
    }

    return result
