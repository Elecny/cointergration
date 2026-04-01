import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen

def johansen_test_detailed(data, det_order=0, k_ar_diff=1, signif_level=0.05):
    """
    详细的 Johansen 协整检验
    
    参数:
    data : DataFrame, 待检验的变量序列
    det_order : 确定性项阶数 (-1:无, 0:常数, 1:常数+趋势)
    k_ar_diff : 滞后阶数
    signif_level : 显著性水平 (0.01, 0.05, 0.10)
    
    返回:
    包含详细结果的字典
    """
    # 执行检验
    result = coint_johansen(data, det_order=det_order, k_ar_diff=k_ar_diff)
    
    # 确定临界值列索引
    if signif_level == 0.01:
        cv_col = 2
    elif signif_level == 0.05:
        cv_col = 1
    elif signif_level == 0.10:
        cv_col = 0
    else:
        raise ValueError("signif_level must be 0.01, 0.05, or 0.10")
    
    # 确定协整秩
    trace_rank = 0
    maxeig_rank = 0
    
    for i in range(len(result.lr1)):
        if result.lr1[i] > result.cvt[i, cv_col]:
            trace_rank += 1
        else:
            break
    
    for i in range(len(result.lr2)):
        if result.lr2[i] > result.cvm[i, cv_col]:
            maxeig_rank += 1
        else:
            break
    
    # 输出结果
    print("=" * 80)
    print(f"Johansen 协整检验 (显著性水平: {signif_level*100:.0f}%)")
    print("=" * 80)
    print(f"样本量: {len(data)}")
    print(f"变量数: {len(data.columns)}")
    print(f"变量名: {list(data.columns)}")
    print(f"滞后阶数 (差分): {k_ar_diff}")
    print(f"确定性项类型: {det_order}")
    print("-" * 80)
    
    # 迹检验详细结果
    print("\n【迹检验】")
    print(f"{'原假设':<25} {'统计量':<12} {'临界值':<12} {'p值近似':<12} {'结论':<10}")
    print("-" * 75)
    
    for i in range(len(result.lr1)):
        if i == 0:
            hypothesis = f"H0: r = 0"
            conclusion = "拒绝" if trace_rank > i else "不拒绝"
        else:
            hypothesis = f"H0: r ≤ {i-1}"
            conclusion = "拒绝" if trace_rank > i else "不拒绝"
        
        trace_stat = result.lr1[i]
        cv = result.cvt[i, cv_col]
        # 近似 p 值 (基于临界值比较)
        if trace_stat > result.cvt[i, 2]:
            p_val = "< 0.01"
        elif trace_stat > result.cvt[i, 1]:
            p_val = "< 0.05"
        elif trace_stat > result.cvt[i, 0]:
            p_val = "< 0.10"
        else:
            p_val = "> 0.10"
        
        print(f"{hypothesis:<25} {trace_stat:<12.4f} {cv:<12.4f} {p_val:<12} {conclusion:<10}")
    
    # 最大特征值检验详细结果
    print("\n【最大特征值检验】")
    print(f"{'原假设':<25} {'统计量':<12} {'临界值':<12} {'p值近似':<12} {'结论':<10}")
    print("-" * 75)
    
    for i in range(len(result.lr2)):
        if i == 0:
            hypothesis = f"H0: r = 0"
            conclusion = "拒绝" if maxeig_rank > i else "不拒绝"
        else:
            hypothesis = f"H0: r = {i}"
            conclusion = "拒绝" if maxeig_rank > i else "不拒绝"
        
        maxeig_stat = result.lr2[i]
        cv = result.cvm[i, cv_col]
        
        if maxeig_stat > result.cvm[i, 2]:
            p_val = "< 0.01"
        elif maxeig_stat > result.cvm[i, 1]:
            p_val = "< 0.05"
        elif maxeig_stat > result.cvm[i, 0]:
            p_val = "< 0.10"
        else:
            p_val = "> 0.10"
        
        print(f"{hypothesis:<25} {maxeig_stat:<12.4f} {cv:<12.4f} {p_val:<12} {conclusion:<10}")
    
    # 输出协整秩
    print("\n" + "=" * 80)
    print(f"结论:")
    print(f"  迹检验建议的协整秩: {trace_rank}")
    print(f"  最大特征值检验建议的协整秩: {maxeig_rank}")
    
    # 输出协整向量 (标准化)
    print("\n【标准化协整向量】")
    # 按最大特征值对应的特征向量排序
    evec = result.evec
    eig = result.eig
    
    # 按特征值降序排序
    idx = np.argsort(eig)[::-1]
    evec_sorted = evec[:, idx]
    
    # 标准化 (第一个变量系数为 1)
    for i in range(len(eig)):
        if evec_sorted[0, i] != 0:
            beta = evec_sorted[:, i] / evec_sorted[0, i]
            print(f"  协整向量 {i+1} (特征值: {eig[idx[i]]:.4f}):")
            for j, col in enumerate(data.columns):
                print(f"    {col}: {beta[j]:.6f}")
            print(f"    协整方程: {data.columns[0]} = ", end="")
            terms = []
            for j in range(1, len(data.columns)):
                if beta[j] > 0:
                    terms.append(f"- {beta[j]:.4f}*{data.columns[j]}")
                else:
                    terms.append(f"+ {-beta[j]:.4f}*{data.columns[j]}")
            print(" ".join(terms))
            print()
    
    # 返回结果字典
    return {
        'trace_stat': result.lr1,
        'trace_crit': result.cvt,
        'maxeig_stat': result.lr2,
        'maxeig_crit': result.cvm,
        'eigenvalues': result.eig,
        'eigenvectors': result.evec,
        'trace_rank': trace_rank,
        'maxeig_rank': maxeig_rank
    }

# 示例使用
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    n = 200
    
    # 创建三个序列，其中两个协整
    e1 = np.random.normal(0, 1, n)
    e2 = np.random.normal(0, 1, n)
    x1 = np.cumsum(np.random.normal(0, 1, n))
    x2 = 2 * x1 + e1
    x3 = np.cumsum(np.random.normal(0, 1, n))  # 随机游走
    
    data = pd.DataFrame({
        'GDP': x1,
        'Consumption': x2,
        'Investment': x3
    })
    
    # 执行检验
    result = johansen_test_detailed(data, det_order=0, k_ar_diff=1, signif_level=0.05)