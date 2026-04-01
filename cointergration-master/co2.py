import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
import matplotlib.pyplot as plt
from pathlib import Path

# ===================== 1. ADF 平稳性检验 =====================
def adf_test(series, name):
    series = series.dropna()
    result = adfuller(series, autolag='AIC')
    p_val = result[1]
    is_stationary = p_val < 0.05
    print(f"📊 {name} ADF检验 p值 = {p_val:.4f} → {'平稳' if is_stationary else '非平稳'}")
    return is_stationary, p_val

# ===================== 2. EG两步法协整检验（无日期版本） =====================
def cointegration_test_direct(data1, data2, name1, name2):
    """
    直接对两个序列进行协整检验（无日期对齐）
    """
    # 确保长度一致
    min_len = min(len(data1), len(data2))
    data1 = data1.iloc[:min_len].reset_index(drop=True)
    data2 = data2.iloc[:min_len].reset_index(drop=True)
    
    print(f"✅ 共 {min_len} 行有效数据")
    print(f"检验变量：{name1} vs {name2}\n")

    # 4. 平稳性检验
    print("="*50)
    print("第一步：原始序列平稳性检验")
    print("="*50)
    s1_stationary, p1 = adf_test(data1, name1)
    s2_stationary, p2 = adf_test(data2, name2)

    if not s1_stationary and not s2_stationary:
        print("\n✅ 两个序列均为非平稳，满足协整前提条件")
        
        # 检验一阶差分是否平稳
        print("\n【一阶差分平稳性检验】")
        diff1 = data1.diff().dropna()
        diff2 = data2.diff().dropna()
        s1_diff, p1_diff = adf_test(diff1, f"{name1}_diff")
        s2_diff, p2_diff = adf_test(diff2, f"{name2}_diff")
        
        if s1_diff and s2_diff:
            print("✅ 两个序列均为一阶单整 I(1)")
        else:
            print("⚠️  警告：序列可能不是严格的 I(1)")
    else:
        print("\n⚠️  警告：协整要求两个序列都必须是非平稳 I(1) 序列！")
        if s1_stationary:
            print(f"   {name1} 是平稳序列")
        if s2_stationary:
            print(f"   {name2} 是平稳序列")

    # 5. 协整检验
    print("\n" + "="*50)
    print("第二步：协整性检验（EG两步法）")
    print("="*50)
    
    res = coint(data1, data2)
    stat, p_val, crit_values = res
    
    print(f"协整检验统计量: {stat:.4f}")
    print(f"p值           : {p_val:.4f}")
    print(f"临界值 (1% 5% 10%):")
    print(f"   1%  : {crit_values[0]:.4f}")
    print(f"   5%  : {crit_values[1]:.4f}")
    print(f"   10% : {crit_values[2]:.4f}")

    print("\n" + "-"*50)
    if p_val < 0.05:
        print(f"✅ 结论：**{name1} 和 {name2} 存在协整关系（长期均衡）**")
    else:
        print(f"❌ 结论：**{name1} 和 {name2} 不存在协整关系**")
    print("-"*50)
    
    # 6. 估计协整方程
    print("\n" + "="*50)
    print("第三步：协整方程估计")
    print("="*50)
    
    X = sm.add_constant(data2)
    model = sm.OLS(data1, X).fit()
    residuals = model.resid
    
    print(f"{name1} = {model.params[0]:.4f} + {model.params[1]:.4f} * {name2}")
    print(f"R² = {model.rsquared:.4f}")
    print(f"调整 R² = {model.rsquared_adj:.4f}")
    
    # 残差平稳性检验
    resid_adf = adfuller(residuals, autolag='AIC')
    resid_p = resid_adf[1]
    print(f"\n残差 ADF 检验 p值: {resid_p:.4f}")
    if resid_p < 0.05:
        print("✅ 残差平稳，进一步确认存在协整关系")
    else:
        print("❌ 残差非平稳，协整关系不成立")
    
    return {
        'coint_stat': stat,
        'coint_pval': p_val,
        'coint_crit': crit_values,
        'model': model,
        'residuals': residuals
    }


# ===================== 4. 从Excel文件读取（修正版） =====================
def cointegration_test(file_path1, file_path2, col1, col2):
    try:
        # 1. 读取两个文件
        df1 = pd.read_excel(file_path1)
        df2 = pd.read_excel(file_path2)
        
        print(f"文件1的列名: {df1.columns.tolist()}")
        print(f"文件2的列名: {df2.columns.tolist()}")
        
        # 2. 检查列是否存在
        if col1 not in df1.columns:
            print(f"❌ 错误：文件1中没有找到列 '{col1}'")
            print(f"   可用的列名: {df1.columns.tolist()}")
            return None
        
        if col2 not in df2.columns:
            print(f"❌ 错误：文件2中没有找到列 '{col2}'")
            print(f"   可用的列名: {df2.columns.tolist()}")
            return None
        
        # 3. 清洗数据：去掉逗号并转换为数值
        df1[col1] = df1[col1].astype(str).str.replace(',', '').str.strip()
        df2[col2] = df2[col2].astype(str).str.replace(',', '').str.strip()
        
        # 处理空值和0值
        df1[col1] = pd.to_numeric(df1[col1], errors='coerce')
        df2[col2] = pd.to_numeric(df2[col2], errors='coerce')
        
        # 移除NaN和0值
        df1 = df1[df1[col1] > 0].reset_index(drop=True)
        df2 = df2[df2[col2] > 0].reset_index(drop=True)
        
        # 4. 直接进行协整检验（无日期对齐）
        result = cointegration_test_direct(
            df1[col1], 
            df2[col2], 
            col1, 
            col2
        )
        
        return result
        
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    
# ===================== 2. EG两步法协整检验（修正版） =====================
def cointegration_test_direct(data1, data2, name1, name2):
    """
    直接对两个序列进行协整检验（无日期对齐）- 修正版
    """
    # 确保长度一致
    min_len = min(len(data1), len(data2))
    data1 = data1.iloc[:min_len].reset_index(drop=True)
    data2 = data2.iloc[:min_len].reset_index(drop=True)
    
    print(f"✅ 共 {min_len} 行有效数据")
    print(f"检验变量：{name1} vs {name2}\n")

    # 4. 平稳性检验
    print("="*50)
    print("第一步：原始序列平稳性检验")
    print("="*50)
    s1_stationary, p1 = adf_test(data1, name1)
    s2_stationary, p2 = adf_test(data2, name2)

    if not s1_stationary and not s2_stationary:
        print("\n✅ 两个序列均为非平稳，满足协整前提条件")
        
        # 检验一阶差分是否平稳
        print("\n【一阶差分平稳性检验】")
        diff1 = data1.diff().dropna()
        diff2 = data2.diff().dropna()
        s1_diff, p1_diff = adf_test(diff1, f"{name1}_diff")
        s2_diff, p2_diff = adf_test(diff2, f"{name2}_diff")
        
        if s1_diff and s2_diff:
            print("✅ 两个序列均为一阶单整 I(1)")
        else:
            print("⚠️  警告：序列可能不是严格的 I(1)")
    else:
        print("\n⚠️  警告：协整要求两个序列都必须是非平稳 I(1) 序列！")
        if s1_stationary:
            print(f"   {name1} 是平稳序列")
        if s2_stationary:
            print(f"   {name2} 是平稳序列")

    # 5. 协整检验
    print("\n" + "="*50)
    print("第二步：协整性检验（EG两步法）")
    print("="*50)
    
    res = coint(data1, data2)
    stat, p_val, crit_values = res
    
    print(f"协整检验统计量: {stat:.4f}")
    print(f"p值           : {p_val:.4f}")
    print(f"临界值 (1% 5% 10%):")
    print(f"   1%  : {crit_values[0]:.4f}")
    print(f"   5%  : {crit_values[1]:.4f}")
    print(f"   10% : {crit_values[2]:.4f}")

    print("\n" + "-"*50)
    if p_val < 0.05:
        print(f"✅ 结论：**{name1} 和 {name2} 存在协整关系（长期均衡）**")
    else:
        print(f"❌ 结论：**{name1} 和 {name2} 不存在协整关系**")
    print("-"*50)
    
    # 6. 估计协整方程
    print("\n" + "="*50)
    print("第三步：协整方程估计")
    print("="*50)
    
    X = sm.add_constant(data2)
    model = sm.OLS(data1, X).fit()
    residuals = model.resid
    
    # 修正：使用 .iloc 或直接访问参数值
    # 方法1：使用 iloc（推荐）
    const = model.params.iloc[0] if hasattr(model.params, 'iloc') else model.params[0]
    coef = model.params.iloc[1] if hasattr(model.params, 'iloc') else model.params[1]
    
    print(f"{name1} = {const:.4f} + {coef:.4f} * {name2}")
    print(f"R² = {model.rsquared:.4f}")
    print(f"调整 R² = {model.rsquared_adj:.4f}")
    
    # 残差平稳性检验
    resid_adf = adfuller(residuals, autolag='AIC')
    resid_p = resid_adf[1]
    print(f"\n残差 ADF 检验 p值: {resid_p:.4f}")
    if resid_p < 0.05:
        print("✅ 残差平稳，进一步确认存在协整关系")
    else:
        print("❌ 残差非平稳，协整关系不成立")
    
    return {
        'coint_stat': stat,
        'coint_pval': p_val,
        'coint_crit': crit_values,
        'model': model,
        'residuals': residuals,
        'const': const,
        'coef': coef
    }

# ===================== 5. 可视化 =====================
def plot_cointegration(data1, data2, residuals, name1, name2):
    """
    绘制协整检验结果图
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 原始序列
    axes[0, 0].plot(data1, label=name1, alpha=0.7, linewidth=2)
    axes[0, 0].plot(data2, label=name2, alpha=0.7, linewidth=2)
    axes[0, 0].set_title('原始序列', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 散点图
    axes[0, 1].scatter(data2, data1, alpha=0.6, s=30)
    # 添加回归线
    z = np.polyfit(data2, data1, 1)
    p = np.poly1d(z)
    axes[0, 1].plot(data2, p(data2), "r--", alpha=0.8, label=f'拟合线: {z[0]:.2f}x + {z[1]:.2f}')
    axes[0, 1].set_xlabel(name2, fontsize=10)
    axes[0, 1].set_ylabel(name1, fontsize=10)
    axes[0, 1].set_title('散点图及协整关系', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 残差序列
    axes[1, 0].plot(residuals, color='red', alpha=0.7, linewidth=1.5)
    axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    axes[1, 0].fill_between(range(len(residuals)), 0, residuals, alpha=0.3, color='red')
    axes[1, 0].set_title('残差序列', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('观测值序号')
    axes[1, 0].set_ylabel('残差')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 残差直方图和Q-Q图
    axes[1, 1].hist(residuals, bins=20, alpha=0.7, color='red', edgecolor='black')
    axes[1, 1].axvline(x=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].set_title('残差分布', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('残差')
    axes[1, 1].set_ylabel('频数')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'协整检验结果: {name1} vs {name2}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

# ===================== 【主程序】 =====================
if __name__ == "__main__":
    
    # 注意：列名应该是实际Excel中的列名
    EXCEL_FILE1 = "row_data/row_data/Colza_oil2025.xlsx"
    EXCEL_FILE2 = "row_data/row_data/Palm_oil2025.xlsx"
    
    VAR1 = "closing_price"
    VAR2 = "closing_price" 
    
    # 如果文件存在，执行检验
    import os
    if os.path.exists(EXCEL_FILE1) and os.path.exists(EXCEL_FILE2):
        result_excel = cointegration_test_direct(EXCEL_FILE1, EXCEL_FILE2, VAR1, VAR2)