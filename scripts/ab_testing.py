# scripts/ab_testing.py
"""
AB测试工具模块
功能：
1. 根据历史转化率计算所需样本量
2. 模拟AB实验数据（用于演示/测试）
3. 分析真实或模拟的AB实验结果（卡方检验）
4. 可视化两组转化率对比
"""
# scripts/ab_testing.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, norm
from statsmodels.stats.power import NormalIndPower

# 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei']   # 使用黑体
plt.rcParams['axes.unicode_minus'] = False

# 获取项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(BASE_DIR, 'output', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)


# ===================== 1. 样本量计算（方案一：NormalIndPower）=====================
def calc_sample_size(p0, p1, alpha=0.05, power=0.8, alternative='larger'):
    """
    计算AB检验所需每组样本量（比例型指标）
    使用 statsmodels 的 NormalIndPower（基于 Cohen's h 效应量）
    p0: 基准转化率
    p1: 预期转化率
    alpha: 显著性水平 (默认0.05)
    power: 统计功效 (默认0.8)
    alternative: 'larger' 单尾检验（仅关注是否提升），'two-sided' 双尾检验
    """
    # 计算 Cohen's h 效应量
    h = 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p0)))

    # 功效分析
    power_analysis = NormalIndPower()
    n = power_analysis.solve_power(effect_size=h,
                                   alpha=alpha,
                                   power=power,
                                   ratio=1.0,
                                   alternative=alternative)
    return int(np.ceil(n))


# ===================== 2. 模拟AB实验数据 =====================
def simulate_ab_test(total_users, p_control, p_treatment, random_seed=42):
    """
    模拟AB实验分组及转化结果
    total_users: 总用户数（会平均分成两组）
    p_control: 对照组真实转化率
    p_treatment: 实验组真实转化率
    返回 DataFrame:
        group: 'control' 或 'treatment'
        converted: 0或1
    """
    np.random.seed(random_seed)
    n_per_group = total_users // 2
    control = np.random.binomial(1, p_control, n_per_group)
    treatment = np.random.binomial(1, p_treatment, n_per_group)
    df = pd.DataFrame({
        'group': ['control'] * n_per_group + ['treatment'] * n_per_group,
        'converted': np.concatenate([control, treatment])
    })
    return df


# ===================== 3. AB实验结果分析（卡方检验）=====================
def analyze_ab_test(df, alpha=0.05):
    """
    分析AB实验结果
    df: 包含 'group' 和 'converted' 列的 DataFrame
    alpha: 显著性水平
    返回: (控制组转化率, 实验组转化率, p值, 是否显著, 相对提升)
    """
    contingency = pd.crosstab(df['group'], df['converted'])
    print("列联表:")
    print(contingency)

    chi2, p, dof, expected = chi2_contingency(contingency)

    control_rate = contingency.loc['control', 1] / contingency.loc['control'].sum()
    treatment_rate = contingency.loc['treatment', 1] / contingency.loc['treatment'].sum()
    lift = (treatment_rate - control_rate) / control_rate if control_rate > 0 else 0

    significant = p < alpha
    return control_rate, treatment_rate, p, significant, lift


# ===================== 4. 结果可视化 =====================
def plot_ab_results(control_rate, treatment_rate, p, significant, lift, save_path=None):
    labels = ['对照组', '实验组']
    rates = [control_rate, treatment_rate]
    colors = ['#3498db', '#e74c3c']

    plt.figure(figsize=(6, 5))
    bars = plt.bar(labels, rates, color=colors, alpha=0.7)
    plt.ylim(0, max(rates) * 1.2)
    plt.ylabel('转化率')
    plt.title(f'AB测试结果 (p={p:.4f}, {"显著" if significant else "不显著"})\n相对提升 {lift:.2%}')

    for bar, rate in zip(bars, rates):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{rate:.2%}', ha='center', va='bottom', fontsize=10)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存至 {save_path}")
    plt.show()


# ===================== 5. 主流程 =====================
if __name__ == "__main__":
    baseline_rate = 0.3427  # 当日复购率 34.27%
    expected_rate = 0.38  # 预期提升到 38%
    alpha = 0.05
    power = 0.8

    # 计算所需样本量
    n_required = calc_sample_size(baseline_rate, expected_rate, alpha, power, alternative='larger')
    print(f"每组所需最小样本量 (单尾检验): {n_required}")
    print(f"两组总样本量: {n_required * 2}")

    # 模拟实验（用计算出的样本量）
    total_users = n_required * 2
    df_sim = simulate_ab_test(total_users, baseline_rate, expected_rate)
    print("\n模拟数据前5行:")
    print(df_sim.head())

    # 分析结果
    control_rate, treatment_rate, p_val, sig, lift = analyze_ab_test(df_sim)
    print(f"\n对照组转化率: {control_rate:.2%}")
    print(f"实验组转化率: {treatment_rate:.2%}")
    print(f"相对提升: {lift:.2%}")
    print(f"p值: {p_val:.6f}")
    if sig:
        print("结论: 实验组显著优于对照组，策略有效 ✅")
    else:
        print("结论: 未检测到显著差异，策略无效 ❌")

    # 保存图表
    save_path = os.path.join(FIGURES_DIR, 'ab_test_result.png')
    plot_ab_results(control_rate, treatment_rate, p_val, sig, lift, save_path)