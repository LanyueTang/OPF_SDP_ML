import numpy as np
import pandas as pd

# 加载npy文件
preds = np.load('/home/goatoine/Documents/Lanyue/models/GNN/result/Apr11_evaluator/preds.npy')
targets = np.load('/home/goatoine/Documents/Lanyue/models/GNN/result/Apr11_evaluator/targets.npy')

# 转为DataFrame
df_preds = pd.DataFrame(preds)
df_targets = pd.DataFrame(targets)

# 保存为CSV
df_preds.to_csv('/home/goatoine/Documents/Lanyue/models/GNN/result/Apr11_evaluator/preds.csv', index=False)
df_targets.to_csv('/home/goatoine/Documents/Lanyue/models/GNN/result/Apr11_evaluator/targets.csv', index=False)

print("转换完成，已生成preds.csv和targets.csv")