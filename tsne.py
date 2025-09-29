import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# 读取数据
E = pd.read_csv(r'result\machine_learning\merged_emotion_features.csv')
I = pd.read_csv(r'result\machine_learning\merged_au_intensities.csv')
C = pd.read_csv(r'result\machine_learning\merged_au_correlations.csv')

# 重新排列行
name_order = E['姓名'].tolist()
I = I.set_index('姓名').reindex(name_order).reset_index()
C = C.set_index('姓名').reindex(name_order).reset_index()

group, ABC, CABS = E['group'], E['ABC'], E['克氏']
E = E[[col for col in E.columns if col not in ['姓名', 'group', 'ABC', '克氏']]]
I = I[[col for col in I.columns if col not in ['姓名', 'group', 'ABC', '克氏']]]
C = C[[col for col in C.columns if col not in ['姓名', 'group', 'ABC', '克氏']]]

E = E.fillna(0)
I = I.fillna(0)
C = C.fillna(0)

# 合并特征
EI = pd.concat([E, I], axis=1)
EC = pd.concat([E, C], axis=1)
IC = pd.concat([C, I], axis=1)
EIC = pd.concat([E, I, C], axis=1)

feature_sets = {
    'E': E,
    'EI': EI,
    'EC': EC,
    'IC': C,
    'EIC': EIC
}

# 创建Nx1的子图布局
fig, axes = plt.subplots(1, len(feature_sets), figsize=(len(feature_sets)*7, 7))
# 创建Excel写入器
writer = pd.ExcelWriter('result/classify/scatter_data_scaled2.xlsx', engine='openpyxl')

for i, (feature_type, X) in enumerate(feature_sets.items()):
    ax = axes[i]


    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # t-SNE降维
    tsne = TSNE(n_components=2, 
                random_state=42, 
                perplexity=30, 

                learning_rate=200)
    X_tsne = tsne.fit_transform(X_scaled)

    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    # 将坐标缩放到[-10, 10]范围
    scaler_minmax = MinMaxScaler(feature_range=(-10, 10))
    X_final = np.column_stack([
        scaler_minmax.fit_transform(X_tsne[:, 0].reshape(-1, 1)).flatten(),
        scaler_minmax.fit_transform(X_tsne[:, 1].reshape(-1, 1)).flatten()
    ])
    
    # 绘制散点图
    colors = ['#2ecc71', '#e74c3c']
    labels = ['TD', 'ASD']
    
    for i, label in enumerate([0, 1]):
        mask = group == label
        ax.scatter(X_final[mask, 0], X_final[mask, 1], 
                  c=colors[i], label=labels[i], 
                  alpha=0.7, s=100)
    
    # 设置子图标题和标签
    ax.set_title(f'{feature_type}', fontsize=14, pad=15)
    ax.set_xlabel('t-SNE dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE dimension 2', fontsize=12)
    ax.legend(title='Group', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.3)

    # 保存数据到Excel
    sheet_name = feature_type.replace(' ', '_')
    scatter_df = pd.DataFrame({
        'Group': group,
        'X': X_final[:, 0],
        'Y': X_final[:, 1]
    })
    scatter_df.to_excel(writer, sheet_name=sheet_name, index=False)
    
# 保存Excel
writer.close()

# 调整布局
plt.tight_layout()
plt.show()