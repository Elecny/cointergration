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
    return is_stationary, p_val

# ===================== 2. EG两步法协整检验 =====================
def cointegration_test_direct(data1, data2, name1, name2):
    """
    直接对两个序列进行协整检验
    """
    # 确保长度一致
    min_len = min(len(data1), len(data2))
    data1 = data1.iloc[:min_len].reset_index(drop=True)
    data2 = data2.iloc[:min_len].reset_index(drop=True)
    
    print(f"✅ 共 {min_len} 行有效数据")
    print(f"检验变量：{name1} vs {name2}\n")

    # 平稳性检验
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

    # 协整检验
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
    
    # 估计协整方程
    print("\n" + "="*50)
    print("第三步：协整方程估计")
    print("="*50)
    
    X = sm.add_constant(data2)
    model = sm.OLS(data1, X).fit()
    residuals = model.resid
    
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

# ===================== 3. 从Excel文件读取（修正版） =====================
def cointegration_test_excel(file_path1, file_path2, col1, col2):
    """
    读取两个Excel文件并做协整检验
    """
    try:
        # 1. 读取两个文件
        df1 = pd.read_excel(file_path1)
        df2 = pd.read_excel(file_path2)
        
        print(f"文件1的列名: {df1.columns.tolist()}")
        print(f"文件2的列名: {df2.columns.tolist()}")
        print(f"文件1的形状: {df1.shape}")
        print(f"文件2的形状: {df2.shape}")
        
        # 2. 检查列是否存在
        if col1 not in df1.columns:
            print(f"❌ 错误：文件1中没有找到列 '{col1}'")
            return None
        
        if col2 not in df2.columns:
            print(f"❌ 错误：文件2中没有找到列 '{col2}'")
            return None
        
        # 3. 数据清洗和转换
        # 处理可能的逗号分隔符
        df1[col1] = df1[col1].astype(str).str.replace(',', '').str.strip()
        df2[col2] = df2[col2].astype(str).str.replace(',', '').str.strip()
        
        # 转换为数值
        df1[col1] = pd.to_numeric(df1[col1], errors='coerce')
        df2[col2] = pd.to_numeric(df2[col2], errors='coerce')
        
        # 移除NaN和0值
        df1 = df1[df1[col1] > 0].reset_index(drop=True)
        df2 = df2[df2[col2] > 0].reset_index(drop=True)
        
        print(f"\n清洗后 {col1} 数据量: {len(df1)}")
        print(f"清洗后 {col2} 数据量: {len(df2)}")
        
        # 4. 处理日期列 - 统一转换为数值格式（去除横杠）
        if '交易日期' in df1.columns:
            # 将日期转换为字符串，去除横杠
            df1['交易日期_clean'] = df1['交易日期'].astype(str).str.replace('-', '').str.replace('/', '')
            # 只保留8位数字（年月日）
            df1['交易日期_clean'] = df1['交易日期_clean'].str.extract(r'(\d{8})')
            # 转换为数值类型
            df1['交易日期_clean'] = pd.to_numeric(df1['交易日期_clean'], errors='coerce')
            print(f"\n文件1日期示例（清洗后）: {df1['交易日期_clean'].dropna().head(10).tolist()}")
        
        if '交易日期' in df2.columns:
            # 将日期转换为字符串，去除横杠
            df2['交易日期_clean'] = df2['交易日期'].astype(str).str.replace('-', '').str.replace('/', '')
            # 只保留8位数字（年月日）
            df2['交易日期_clean'] = df2['交易日期_clean'].str.extract(r'(\d{8})')
            # 转换为数值类型
            df2['交易日期_clean'] = pd.to_numeric(df2['交易日期_clean'], errors='coerce')
            print(f"文件2日期示例（清洗后）: {df2['交易日期_clean'].dropna().head(10).tolist()}")
        
        # 5. 按日期对齐（使用数值格式的日期）
        if '交易日期_clean' in df1.columns and '交易日期_clean' in df2.columns:
            print("\n检测到日期列，尝试按日期对齐数据（使用数值格式）...")
            
            # 移除日期为NaN的行
            df1_valid = df1.dropna(subset=['交易日期_clean']).reset_index(drop=True)
            df2_valid = df2.dropna(subset=['交易日期_clean']).reset_index(drop=True)
            
            print(f"有效日期数据量 - 文件1: {len(df1_valid)}, 文件2: {len(df2_valid)}")
            
            if len(df1_valid) > 0 and len(df2_valid) > 0:
                # 先按日期分组，计算每个日期的平均价格（避免一对多匹配）
                df1_grouped = df1_valid.groupby('交易日期_clean')[col1].mean().reset_index()
                df2_grouped = df2_valid.groupby('交易日期_clean')[col2].mean().reset_index()
                
                print(f"分组后文件1唯一日期数: {len(df1_grouped)}")
                print(f"分组后文件2唯一日期数: {len(df2_grouped)}")
                
                # 按日期进行合并
                merged = pd.merge(
                    df1_grouped, 
                    df2_grouped, 
                    on='交易日期_clean', 
                    how='inner'
                )
                
                print(f"日期对齐后共 {len(merged)} 行数据")
                
                if len(merged) > 0:
                    # 使用正确的列名（合并后列名保持不变）
                    data1 = merged[col1]
                    data2 = merged[col2]
                    
                    print(f"对齐后数据统计:")
                    print(f"  日期范围: {merged['交易日期_clean'].min()} 到 {merged['交易日期_clean'].max()}")
                    print(f"  菜籽油价格范围: {data1.min():.2f} - {data1.max():.2f}")
                    print(f"  棕榈油价格范围: {data2.min():.2f} - {data2.max():.2f}")
                    
                    # 检查数据是否充足
                    if len(data1) < 30:
                        print(f"⚠️  警告：对齐后数据量不足30，结果可能不可靠")
                    
                    # 进行协整检验
                    result = cointegration_test_direct(
                        data1, data2, 
                        "菜籽油收盘价", 
                        "棕榈油收盘价"
                    )
                    return result
                else:
                    print("⚠️  日期无法对齐（没有共同日期）")
            else:
                print("⚠️  日期数据无效")
        
        # 如果没有日期列或日期无法对齐，使用索引对齐
        print("\n使用索引对齐方式（按数据顺序对齐）...")
        # 取较短的序列长度
        min_len = min(len(df1), len(df2))
        data1 = df1[col1].iloc[:min_len]
        data2 = df2[col2].iloc[:min_len]
        
        result = cointegration_test_direct(
            data1, data2, 
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
    VAR1 = "closing_price"  # 菜籽油收盘价
    VAR2 = "closing_price"  # 棕榈油收盘价
    
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
            print("分析完成！")
            print("="*60)
            
            # 显示对冲策略参数
            print(f"\n【对冲策略参数】")
            print(f"对冲比率: 1份菜籽油 = {result['hedge_ratio']:.4f} 份棕榈油")
            print(f"截距项: {result['intercept']:.4f}")
            
            # 保存残差
            result_df = pd.DataFrame({
                'residuals': result['residuals'].values
            })
            result_df.to_excel("cointegration_residuals.xlsx", index=False)
            print(f"\n✅ 残差已保存到 cointegration_residuals.xlsx")
            
    else:
        print(f"\n❌ 文件不存在，请检查路径:")
        print(f"   {EXCEL_FILE1}: {'存在' if os.path.exists(EXCEL_FILE1) else '不存在'}")
        print(f"   {EXCEL_FILE2}: {'存在' if os.path.exists(EXCEL_FILE2) else '不存在'}")