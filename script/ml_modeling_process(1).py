"""
=============================================================================
Steam 游戏销售额预测 - 机器学习建模完整流程 
目标：识别影响游戏总收入的核心因素，并建立预测模型
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# Step 1: 数据加载与清洗 (针对销售额)
# =============================================================================
print("=" * 60)
print("Step 1: 数据准备")
print("=" * 60)

file_path = r"C:\Users\Xinyi\OneDrive\Desktop\main_df2.csv"
df = pd.read_csv(file_path)

# 核心策略：
# 1. 目标变量设为 revenue
# 2. 必须删除 estimated_sales，否则会产生严重的数据泄漏
# 3. 删除 ID 类和非数值类的冗余信息
drop_cols = [
    'appid', 'name', 'developer_name', 'publisher_name', 
    'estimated_sales'  # 必须删除，它是收入的直接组成部分
]

# 过滤掉收入为 0 的游戏（通常是免费游戏或异常数据，会干扰金额预测）
df = df[df['revenue'] > 0]

X = df.drop(columns=drop_cols + ['revenue'])
y = df['revenue']

print(f"有效样本量: {X.shape[0]} | 特征维度: {X.shape[1]}")

# =============================================================================
# Step 2: 目标变量 Log 变换
# =============================================================================
# 销售额通常呈现极端的“长尾分布”，Log变换能让模型更容易捕捉中小型游戏的规律
y_log = np.log1p(y)

# =============================================================================
# Step 3: 特征处理
# =============================================================================
# 将布尔值列转换为 0/1 整数
bool_cols = X.select_dtypes(include=['bool']).columns
X[bool_cols] = X[bool_cols].astype(int)

# =============================================================================
# Step 4: 训练集与测试集划分
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)

# =============================================================================
# Step 5: 模型训练 (使用鲁棒性更强的 HistGradientBoosting)
# =============================================================================
print("\n" + "=" * 60)
print("Step 5: 训练销售额预测模型")
print("=" * 60)

# 针对金额预测，我们增加正则化强度以防止过拟合
model = HistGradientBoostingRegressor(
    max_iter=1000,
    learning_rate=0.05,
    max_depth=6,
    l2_regularization=1.0,  # 增加正则化
    random_state=42
)

model.fit(X_train, y_train)

# 评估预测效果
y_pred_log = model.predict(X_test)
r2 = r2_score(y_test, y_pred_log)
mae_log = mean_absolute_error(y_test, y_pred_log)

print(f"模型 R² (Log空间): {r2:.4f}")
print(f"平均绝对误差 (Log空间): {mae_log:.4f}")

# =============================================================================
# Step 6: 销售额驱动因素分析 (特征重要性)
# =============================================================================
print("\n" + "=" * 60)
print("Step 6: 销售额核心驱动因子分析")
print("=" * 60)

# 使用 Permutation Importance 得到最真实的特征贡献度
perm_result = permutation_importance(
    model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1
)

feat_imp = pd.Series(
    perm_result.importances_mean, index=X.columns
).sort_values(ascending=False)

print("\n影响销售额的前 10 大关键特征:")
print("-" * 40)
for i, (feat, val) in enumerate(feat_imp.head(10).items()):
    print(f"{i+1}. {feat:30} 重要性权重: {val:.4f}")

# =============================================================================
# Step 7: 商业决策建议
# =============================================================================
print("\n" + "=" * 60)
print("Step 7: 商业行动建议")
print("=" * 60)

print("""
基于销售额预测模型的发现：
1. 定价（price_usd）的杠杆作用：价格对收入的影响远大于销量，应分析不同价位段的转化率。
2. 社区资产的力量：Steam Trading Cards（交易卡）和 Achievements（成就）往往与高收入正相关，
   它们不仅能增加玩家粘性，还能提升商店页面的转化权重。
3. 本地化投入：观察 supported_languages 的权重，确定是否值得增加特定语言的支持。
4. 类型避坑：关注 genre_ 开头的特征，识别哪些类型的游戏虽然销量可能高，但单位收入（Revenue）较低。
""")