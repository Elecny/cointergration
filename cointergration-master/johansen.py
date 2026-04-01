import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen

def johansen_test(data, det_order=0, k_ar_diff=1):
    """
    Johansen 协整检验
    
    参数:
    data : DataFrame, 待检验的变量序列
    det_order : 确定性项阶数
                -1 : 无确定性项
                0 : 常数项
                1 : 常数项和趋势项
    k_ar_diff : 滞后阶数（差分阶数）
    
    返回:
    检验结果
    """
    # 执行 Johansen 检验
    result = coint_johansen(data, det_order=det_order, k_ar_diff=k_ar_diff)
    
    # 输出结果
    print("=" * 60)
    print("Johansen 协整检验结果")
    print("=" * 60)
    
    # 迹检验
    print("\n【迹检验】")
    print(f"{'原假设':<30} {'统计量':<12} {'10%临界值':<12} {'5%临界值':<12} {'1%临界值':<12}")
    print("-" * 78)
    
    for i in range(len(result.lr1)):
        r0 = f"r = {i} (不存在协整关系)" if i == 0 else f"r <= {i-1} (至多{i-1}个协整关系)"
        trace_stat = result.lr1[i]
        cv_90 = result.cvt[i, 0]  # 90% 临界值
        cv_95 = result.cvt[i, 1]  # 95% 临界值
        cv_99 = result.cvt[i, 2]  # 99% 临界值
        
        print(f"{r0:<30} {trace_stat:<12.4f} {cv_90:<12.4f} {cv_95:<12.4f} {cv_99:<12.4f}")
    
    # 最大特征值检验
    print("\n【最大特征值检验】")
    print(f"{'原假设':<30} {'统计量':<12} {'10%临界值':<12} {'5%临界值':<12} {'1%临界值':<12}")
    print("-" * 78)
    
    for i in range(len(result.lr2)):
        r0 = f"r = {i} (不存在协整关系)" if i == 0 else f"r <= {i-1} (至多{i-1}个协整关系)"
        maxeig_stat = result.lr2[i]
        cv_90 = result.cvm[i, 0]  # 90% 临界值
        cv_95 = result.cvm[i, 1]  # 95% 临界值
        cv_99 = result.cvm[i, 2]  # 99% 临界值
        
        print(f"{r0:<30} {maxeig_stat:<12.4f} {cv_90:<12.4f} {cv_95:<12.4f} {cv_99:<12.4f}")
    
    # 协整向量
    print("\n【协整向量 (beta)】")
    print(result.evec)
    
    # 特征值
    print("\n【特征值】")
    print(result.eig)
    
    return result

# 示例使用
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    n = 200
    
    # 创建两个协整的序列
    e = np.random.normal(0, 1, n)
    x1 = np.cumsum(np.random.normal(0, 1, n))
    x2 = x1 + e  # 协整关系
    
    data = pd.DataFrame({
        'x1': x1,
        'x2': x2
    })
    
    # 执行检验
    result = johansen_test(data, det_order=0, k_ar_diff=1)