# -*- coding: utf-8 -*-

import pandas as pd
# 参数初始化
discfile = '../data/arima_data.xls'
forecastnum = 5

# 读取数据，指定日期列为指标，pandas自动将“日期”列识别为Datetime格式
data = pd.read_excel(discfile, index_col = '日期')

# 时序图
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
data.plot()
plt.show()

# 自相关图
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(data).show()

# 平稳性检测
from statsmodels.tsa.stattools import adfuller as ADF
print('原始序列的ADF检验结果为：', ADF(data['销量']))
# 返回值依次为adf、pvalue、usedlag、nobs、critical values、icbest、regresults、resstore

# 差分后的结果
D_data = data.diff().dropna()
D_data.columns = ['销量差分']
D_data.plot()  # 时序图
plt.show()
plot_acf(D_data).show()  # 自相关图
from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(D_data).show()  # 偏自相关图
print('差分序列的ADF检验结果为：', ADF(D_data['销量差分']))  # 平稳性检测

# 白噪声检验
from statsmodels.stats.diagnostic import acorr_ljungbox
print('差分序列的白噪声检验结果为：', acorr_ljungbox(D_data, lags=1))  # 返回统计量和p值

from statsmodels.tsa.arima_model import ARIMA

# 定阶
data['销量'] = data['销量'].astype(float) 
pmax = int(len(D_data)/10)  # 一般阶数不超过length/10
qmax = int(len(D_data)/10)  # 一般阶数不超过length/10
bic_matrix = []  # BIC矩阵
for p in range(pmax+1):
  tmp = []
  for q in range(qmax+1):
    try:  # 存在部分报错，所以用try来跳过报错。
      tmp.append(ARIMA(data, (p,1,q)).fit().bic)
    except:
      tmp.append(None)
  bic_matrix.append(tmp)

bic_matrix = pd.DataFrame(bic_matrix)  # 从中可以找出最小值

p,q = bic_matrix.stack().idxmin()  # 先用stack展平，然后用idxmin找出最小值位置。
print('BIC最小的p值和q值为：%s、%s' %(p,q)) 
model = ARIMA(data, (p,1,q)).fit()  # 建立ARIMA(0, 1, 1)模型
print('模型报告为：\n', model.summary2())
print('预测未来5天，其预测结果、标准误差、置信区间如下：\n', model.forecast(5))

