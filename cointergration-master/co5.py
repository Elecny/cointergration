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
    
    # 正确访问模型参数
    const_param = model.params.iloc[0]
    coef_param = model.params.iloc[1]
    
    print(f"{name1} = {const_param:.4f} + {coef_param:.4f} * {name2}")
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
        'hedge_ratio': coef_param,
        'intercept': const_param
    }

# ===================== 3. 从Excel文件读取（修复日期问题） =====================
def cointegration_test_excel(file_path1, file_path2, col1, col2):
    """
    读取两个Excel文件并做协整检验
    """
    try:
        # 1. 读取两个文件
        print("正在读取Excel文件...")
        df1 = pd.read_excel(file_path1)
        df2 = pd.read_excel(file_path2)
        
        print(f"\n文件1的列名: {df1.columns.tolist()}")
        print(f"文件2的列名: {df2.columns.tolist()}")
        print(f"\n文件1的形状: {df1.shape}")
        print(f"文件2的形状: {df2.shape}")
        
        # 2. 检查列是否存在
        if col1 not in df1.columns:
            print(f"❌ 错误：文件1中没有找到列 '{col1}'")
            print(f"   可用的列名: {df1.columns.tolist()}")
            return None
        
        if col2 not in df2.columns:
            print(f"❌ 错误：文件2中没有找到列 '{col2}'")
            print(f"   可用的列名: {df2.columns.tolist()}")
            return None
        
        # 3. 数据清洗和转换
        print("\n正在清洗数据...")
        
        # 处理可能的逗号分隔符
        df1[col1] = df1[col1].astype(str).str.replace(',', '').str.strip()
        df2[col2] = df2[col2].astype(str).str.replace(',', '').str.strip()
        
        # 转换为数值，无法转换的变为NaN
        df1[col1] = pd.to_numeric(df1[col1], errors='coerce')
        df2[col2] = pd.to_numeric(df2[col2], errors='coerce')
        
        # 移除NaN和0值
        df1 = df1[df1[col1] > 0].reset_index(drop=True)
        df2 = df2[df2[col2] > 0].reset_index(drop=True)
        
        print(f"清洗后 {col1} 数据量: {len(df1)}")
        print(f"清洗后 {col2} 数据量: {len(df2)}")
        
        # 4. 尝试修复日期列
        date_col1 = None
        date_col2 = None
        
        # 检查是否有日期列
        if '交易日期' in df1.columns:
            date_col1 = '交易日期'
        elif '交易日期' in df2.columns:
            date_col2 = '交易日期'
        
        # 尝试修复日期格式
        if date_col1:
            print(f"\n文件1发现日期列: {date_col1}")
            print(f"文件1日期列前5行: {df1[date_col1].head().tolist()}")
            
            # 尝试多种方式解析日期
            try:
                # 如果是Excel序列号格式
                if pd.api.types.is_numeric_dtype(df1[date_col1]):
                    df1[date_col1] = pd.to_datetime(df1[date_col1], origin='1899-12-30', unit='D')
                else:
                    df1[date_col1] = pd.to_datetime(df1[date_col1], errors='coerce')
            except:
                df1[date_col1] = pd.to_datetime(df1[date_col1], errors='coerce')
            
            print(f"解析后日期范围: {df1[date_col1].min()} 到 {df1[date_col1].max()}")
        
        if date_col2:
            print(f"\n文件2发现日期列: {date_col2}")
            print(f"文件2日期列前5行: {df2[date_col2].head().tolist()}")
            
            # 尝试多种方式解析日期
            try:
                # 如果是Excel序列号格式
                if pd.api.types.is_numeric_dtype(df2[date_col2]):
                    df2[date_col2] = pd.to_datetime(df2[date_col2], origin='1899-12-30', unit='D')
                else:
                    df2[date_col2] = pd.to_datetime(df2[date_col2], errors='coerce')
            except:
                df2[date_col2] = pd.to_datetime(df2[date_col2], errors='coerce')
            
            print(f"解析后日期范围: {df2[date_col2].min()} 到 {df2[date_col2].max()}")
        
        # 5. 决定使用哪种对齐方式
        use_date_alignment = False
        if date_col1 and date_col2:
            # 检查是否有有效的日期数据
            valid_dates1 = df1[date_col1].notna().sum()
            valid_dates2 = df2[date_col2].notna().sum()
            
            if valid_dates1 > 0 and valid_dates2 > 0:
                use_date_alignment = True
                print(f"\n✅ 两个文件都有有效日期，将使用日期对齐")
                print(f"   文件1有效日期数: {valid_dates1}")
                print(f"   文件2有效日期数: {valid_dates2}")
            else:
                print(f"\n⚠️  日期数据无效，将使用索引对齐")
        else:
            print(f"\n⚠️  缺少日期列，将使用索引对齐")
        
        if use_date_alignment:
            # 按日期对齐（保留所有数据，不主动去重）
            print("\n正在按日期对齐数据...")
            
            # 移除日期为NaN的行
            df1_clean = df1[df1[date_col1].notna()].copy()
            df2_clean = df2[df2[date_col2].notna()].copy()
            
            # 合并数据，使用inner join只保留共同日期的数据
            merged = pd.merge(
                df1_clean[[date_col1, col1]], 
                df2_clean[[date_col2, col2]], 
                left_on=date_col1, 
                right_on=date_col2, 
                how='inner'
            )
            
            if len(merged) > 0:
                print(f"日期对齐后共 {len(merged)} 行数据")
                print(f"共同日期范围: {merged[date_col1].min()} 到 {merged[date_col1].max()}")
                
                data1 = merged[col1]
                data2 = merged[col2]
                
                # 检查数据是否充足
                if len(data1) < 30:
                    print(f"⚠️  警告：对齐后数据量不足30，结果可能不可靠")
                
                # 进行协整检验
                result = cointegration_test_direct(data1, data2, "菜籽油收盘价", "棕榈油收盘价")
                return result
            else:
                print("⚠️  日期对齐失败，没有共同日期，将使用索引对齐")
        
        # 如果日期对齐失败或不需要日期对齐，使用索引对齐
        print("\n使用索引对齐方式（按数据顺序对齐）...")
        result = cointegration_test_direct(
            df1[col1], 
            df2[col2], 
            "菜籽油收盘价", 
            "棕榈油收盘价"
        )
        
        return result
        
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        import traceback
        traceback.print_exc()
        return None

# ===================== 【主程序】 =====================
if __name__ == "__main__":
    print("="*60)
    print("协整检验分析程序")
    print("="*60)
    
    # Excel文件路径
    EXCEL_FILE1 = "row_data/row_data/Colza_oil2025.xlsx"
    EXCEL_FILE2 = "row_data/row_data/Palm_oil2025.xlsx"
    
    # 列名
    VAR1 = "closing_price"
    VAR2 = "closing_price"
    
    # 检查文件是否存在
    import os
    if os.path.exists(EXCEL_FILE1) and os.path.exists(EXCEL_FILE2):
        print(f"\n正在分析文件:")
        print(f"  文件1: {EXCEL_FILE1}")
        print(f"  文件2: {EXCEL_FILE2}")
        print(f"  变量1: {VAR1}")
        print(f"  变量2: {VAR2}")
        print()
        
        # 执行协整检验
        result = cointegration_test_excel(EXCEL_FILE1, EXCEL_FILE2, VAR1, VAR2)
        
        if result:
            print("\n" + "="*60)
            print("✅ 协整检验完成！")
            print("="*60)
            
            # 输出主要结果摘要
            print(f"\n【检验结果摘要】")
            print(f"协整检验统计量: {result['coint_stat']:.4f}")
            print(f"p值: {result['coint_pval']:.4f}")
            print(f"对冲比率: {result['hedge_ratio']:.4f}")
            print(f"截距项: {result['intercept']:.4f}")
            
            if result['coint_pval'] < 0.05:
                print(f"\n✅ 结论：菜籽油和棕榈油价格存在长期协整关系")
                print(f"   可以进行配对交易策略")
            else:
                print(f"\n❌ 结论：菜籽油和棕榈油价格不存在长期协整关系")
                print(f"   不适合进行配对交易")
            
    else:
        print(f"\n❌ 文件不存在，请检查路径:")
        print(f"   {EXCEL_FILE1}: {'存在' if os.path.exists(EXCEL_FILE1) else '不存在'}")
        print(f"   {EXCEL_FILE2}: {'存在' if os.path.exists(EXCEL_FILE2) else '不存在'}")