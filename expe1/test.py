# 导入所需的模块
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 已有数据
boston = load_boston()

X = np.mat(boston.data).reshape((506, 13))
y = np.mat(boston.target).reshape((506, 1))

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
# 拟合
reg = LinearRegression()
reg.fit(X_train, y_train)
# print('权重：', reg.coef_)

y_pre = reg.predict(X_test)

sse = np.array([i**2 for i in (y_pre - np.mean(y_test))]).sum()
sst = np.array([i**2 for i in (y_test - np.mean(y_test))]).sum()
ssr = np.array([i**2 for i in (np.array(y_test) - y_pre)]).sum()
R_2 = sse / sst
print("SSE = ", sse)
print("SST = ", sst)
print("SSR = ", ssr)
print("R_2 = ", R_2)

plt.plot(y_pre)
plt.plot(np.array(y_test)[:, 0])
plt.show()
# # 可视化
# prediction = reg.predict(height)                # 根据高度，按照拟合的曲线预测温度值
# plt.figure('海拔高度~温度关系曲线拟合结果', figsize=(12,8))
# plt.rcParams['font.family'] = ['sans-serif']    # 设置matplotlib 显示中文
# plt.rcParams['font.sans-serif'] = ['SimHei']    # 设置matplotlib 显示中文
# plt.xlabel('温度')
# plt.ylabel('高度')
# plt.scatter(temp, height, c='black')
# plt.plot(prediction, height, c='r')
# plt.show()