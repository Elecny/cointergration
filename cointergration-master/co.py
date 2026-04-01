import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint

# ===================== 1. ADF 平稳性检验（复用版） =====================
def adf_test(series, name="序列"):
    series = series.dropna()
    result = adfuller(series, autolag='AIC')
    p_val = result[1]
    is_stationary = p_val < 0.05
    print(f"📊 {name} ADF检验 p值 = {p_val:.4f} → {'平稳' if is_stationary else '非平稳'}")
    return is_stationary

# ===================== 2. EG两步法协整检验（核心） =====================
def cointegration_test_excel(file_path1,file_path2, col1, col2):
    """
    对Excel中的两列时间序列做协整检验
    :param file_path: Excel文件路径（如"data.xlsx"）
    :param col1: 第一列变量名
    :param col2: 第二列变量名
    """
    # 1. 读取Excel数据
    df1 = pd.read_excel(file_path1)
    df1 = df1[col1].dropna()  # 去除空值
    
    df2 = pd.read_excel(file_path2)
    df2 = df2[col2].dropna()  # 去除空值
    
    # ==============================================
    # ✅✅✅ 这里修复：去掉数字里的千分位逗号
    # ==============================================
    df1 = df1.astype(str).str.replace(',', '').astype(float)
    df2 = df2.astype(str).str.replace(',', '').astype(float)

    print(f"✅ 成功读取数据，共 {len(df1)} 行")
    print(f"检验变量：{col1} vs {col2}\n")

    # 2. 检验原始序列平稳性（协整前提：都非平稳）
    print("="*50)
    print("第一步：原始序列平稳性检验")
    print("="*50)
    s1_stationary = adf_test(df1, col1)
    s2_stationary = adf_test(df2, col2)

    if s1_stationary or s2_stationary:
        print("\n❌ 警告：协整要求两个序列都必须是非平稳！")
        return

    # 3. 执行协整检验（EG两步法）
    print("\n" + "="*50)
    print("第二步：协整性检验（EG两步法）")
    print("="*50)
    
    # 核心代码：直接检验协整
    res = coint(df1, df2)
    stat, p_val, crit_values = res
    
    print(f"协整检验统计量: {stat:.4f}")
    print(f"p值           : {p_val:.4f}")
    print(f"临界值 (1% 5% 10%): {crit_values}")

    # 4. 输出结论
    print("\n" + "-"*50)
    if p_val < 0.05:
        print(f"✅ 结论：**{col1} 和 {col2} 存在协整关系（长期均衡）**")
    else:
        print(f"❌ 结论：**{col1} 和 {col2} 不存在协整关系**")
    print("-"*50)

# ===================== 【你只需要改这里！】 =====================
if __name__ == "__main__":
    # 1. Excel文件路径（相对路径/绝对路径都可以）
    EXCEL_FILE1 = "/home/mony/Myfolder/PythonProj/row_data/row_date/SoyaBean_oil2025.xlsx"
    EXCEL_FILE2 = "/home/mony/Myfolder/PythonProj/row_data/row_date/Colza_oil2025.xlsx"
    
    # 2. 你要检验的两列列名（必须和Excel里一致）
    VAR1 = "最高价"
    VAR2 = "最高价"
    
    # 3. 执行检验
    cointegration_test_excel(EXCEL_FILE1,EXCEL_FILE2, VAR1, VAR2)