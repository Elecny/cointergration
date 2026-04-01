import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

def adf_test(series, signif=0.05):
    """
    ADF平稳性检验函数
    :param series: 一维时间序列数据 (list/np.array/pd.Series)
    :param signif: 显著性水平，默认0.05（5%）
    :return: 打印检验结果，返回检验统计量、p值等
    """
    # 去除空值（必须处理，否则报错）
    series = series.dropna() if isinstance(series, pd.Series) else pd.Series(series).dropna()
    
    # ADF检验核心代码
    result = adfuller(series, autolag='AIC')
    
    # 提取结果
    adf_stat = result[0]  # ADF统计量
    p_value = result[1]   # p值
    critical_values = result[4]  # 临界值
    
    # 输出结果
    print("="*50)
    print("          ADF 单位根检验结果")
    print("="*50)
    print(f'ADF 统计量   : {adf_stat:.4f}')
    print(f'p 值         : {p_value:.4f}')
    print(f'临界值       :')
    for key, val in critical_values.items():
        print(f'    {key}: {val:.4f}')
    
    # 判断平稳性
    print("-"*50)
    if p_value <= signif and adf_stat < critical_values['5%']:
        print(f"✅ 结论：在 {signif*100}% 显著性水平下，**序列平稳**（拒绝原假设）")
    else:
        print(f"❌ 结论：在 {signif*100}% 显著性水平下，**序列非平稳**（不能拒绝原假设）")
    print("="*50)
    
    return result