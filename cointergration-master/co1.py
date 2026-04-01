import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint

# ===================== 1. ADF 平稳性检验 =====================
def adf_test(series, name):
    series = series.dropna()
    result = adfuller(series, autolag='AIC')
    p_val = result[1]
    is_stationary = p_val < 0.05
    print(f"📊 {name} ADF检验 p值 = {p_val:.4f} → {'平稳' if is_stationary else '非平稳'}")
    return is_stationary

# ===================== 2. EG两步法协整检验 =====================
def cointegration_test_excel(file_path1, file_path2, col1, col2, date_col="交易日期"):
    """
    读取两个Excel文件，按日期对齐后做协整检验
    """
    # 1. 读取两个文件
    df1 = pd.read_excel(file_path1)
    df2 = pd.read_excel(file_path2)

    # 2. 清洗数据：去掉逗号
    df1[col1] = df1[col1].astype(str).str.replace(',', '').astype(float)
    df2[col2] = df2[col2].astype(str).str.replace(',', '').astype(float)

    # ===================== 去掉日期里的 “-” =====================
    df1[date_col] = df1[date_col].astype(str).str.replace('-', '')  # 2025-01-01 → 20250101
    df2[date_col] = df2[date_col].astype(str).str.replace('-', '')  # 2025-01-01 → 20250101

    # 3. 按日期合并
    df1 = df1[[date_col, col1]].dropna()
    df2 = df2[[date_col, col2]].dropna()
    
    df = pd.merge(df1, df2, on=date_col, how="inner")  # 取共同日期
    df = df.dropna()

    print(f"✅ 对齐后共 {len(df)} 行有效数据")
    print(f"检验变量：{col1}(文件1) vs {col2}(文件2)\n")

    # 4. 平稳性检验
    print("="*50)
    print("第一步：原始序列平稳性检验")
    print("="*50)
    s1_stationary = adf_test(df[col1], col1)
    s2_stationary = adf_test(df[col2], col2)

    if s1_stationary or s2_stationary:
        print("\n❌ 警告：协整要求两个序列都必须是非平稳！")
        return

    # 5. 协整检验
    print("\n" + "="*50)
    print("第二步：协整性检验（EG两步法）")
    print("="*50)
    
    res = coint(df[col1], df[col2])
    stat, p_val, crit_values = res
    
    print(f"协整检验统计量: {stat:.4f}")
    print(f"p值           : {p_val:.4f}")
    print(f"临界值 (1% 5% 10%): {crit_values}")

    print("\n" + "-"*50)
    if p_val < 0.05:
        print(f"✅ 结论：**{col1} 和 {col2} 存在协整关系（长期均衡）**")
    else:
        print(f"❌ 结论：**{col1} 和 {col2} 不存在协整关系**")
    print("-"*50)

# ===================== 【你只需要改这里！】 =====================
if __name__ == "__main__":
    EXCEL_FILE1 = "row_data/row_data/Colza_oil2025.xlsx"
    EXCEL_FILE2 = "row_data/row_data/Palm_oil2025.xlsx"
    
    VAR1 = "closing_price"
    VAR2 = "closing_price"
    
    # 执行
    cointegration_test_excel(EXCEL_FILE1, EXCEL_FILE2, VAR1, VAR2)


