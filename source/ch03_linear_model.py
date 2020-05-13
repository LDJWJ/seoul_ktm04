#!/usr/bin/env python
# coding: utf-8

# ## ch03 선형모델 - linear model

# * Machine Learning with sklearn @ DJ,Lim
# * date : 20/04

# * 선형회귀(linear regression)는 100여 년 전에 개발되었다.
# * 선형 모델은 입력 특성에 대한 선형 함수를 만들어 예측을 수행
# * 특성이 하나일 때는 직선, 두개일 때는 평면, 더 높은 차원 초평면(hyperplane)
# * KnnRegressor과 비교해 보면 직선이 사용한 예측이 더 제약이 있음.
# * 특성이 많은 데이터 셋이라면 선형 모델은 휼륭한 성능을 갖는다.

# In[1]:


from IPython.display import display, Image


# In[2]:


display(Image(filename='img/linear_model01.png'))


# * x1~xp는 데이터 포인트에 대한 특성
# * w와 b는 모델이 학습할 파라미터
# * ^y = w1 * x1 + b 는 특성이 하나인 데이터 셋

# * 선형회귀 또는 최소제곱법(OLS)은 가장 간단하고 오래된 회귀용 선형 알고리즘. 
# * 선형 회귀는 예측과 훈련 세트에 있는 타깃 y사이의 평균제곱오차(mean squared error)를 최소화하는 파라미터 w와 b를 찾는다.
# * 평균 제곱 오차는 예측값과 타깃값의 차이를 제곱하여 더한 후에 샘플의 개수로 나눈 것.

# In[3]:


display(Image(filename='img/linear_model02_mse.png'))


# In[4]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib
print(np.__version__)
print(matplotlib.__version__)


# * mglearn은 numpy 1.16를 필요함.

# In[7]:


import mglearn
import sklearn
print( sklearn.__version__)
print( mglearn.__version__)

# 설치가 안되어 있을 경우, 설치 필요.
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# ### 01 회귀 선형 모델 그래프로 살펴보기

# In[8]:


mglearn.plots.plot_linear_regression_wave()


# ### 02 Boston 데이터 셋을 활용한 회귀 모델 만들어보기
# <pre>
# (1) 모델 만들기 [  모델명 = 모델객체() ]
# (2) 모델 학습 시키기 [ 모델명.fit() ]
# (3) 모델을 활용한 예측하기 [ 모델명.predict() ]
# (4) 모델 평가
# </pre>

# In[9]:


from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston


# In[10]:


boston = load_boston()
X = boston.data       # 입력 데이터  - 문제
y = boston.target     # 출력 데이터  - 답


# ### 데이터 준비하기

# In[11]:


X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                           random_state=42)


# In[17]:


model = LinearRegression().fit(X_train, y_train)   # 학습
pred = model.predict(X_test)
pred


# In[24]:


import pandas as pd


# In[28]:


dat = pd.DataFrame( {"y_test":y_test, "prediction":pred , "diff":y_test - pred} )
dat


# In[31]:


dat['ABS']= abs(dat['diff'])
dat['diff*diff']= dat['diff'] ** (2)
dat


# ### 평가 지표
#  * MAE(mean absolute error)
#  * MSE(mean squared error)
#  * RMSE(root mean squared error)

# ### MAE (mean absolute error)
#  * 각각의 값에 절대값을 취한다. 이를 전부 더한 후, 갯수로 나누어주기

# In[34]:


### MSE, MAE, RMSE, RMLSE
sum(abs(dat['diff']))/len(dat['diff'])


# ### MSE (mean squared error)
# * (실제값-예측값) ^ 2 의 합를 데이터의 샘플의 개수로 나누어준것

# In[35]:


mse_value = sum(dat['diff'] ** 2) / len(dat['diff'])
mse_value


# In[40]:


from sklearn.metrics import mean_squared_error


# In[41]:


mean_squared_error(y_test, pred)


# ### RMSE (root mean squared error)
# * (실제값-예측값) ^ 2 의 합을 데이터의 샘플의 개수로 나누어 준 이후에 제곱근 씌우기

# In[42]:


# (1) 제곱에 루트를 씌워구하기  (2) 제곱한 값을 길이로 나누기
result = np.sqrt(mse_value)
print(result)


# ### 결정계수
#  * 통계학에서 선형모형이 주어진 자료에 적합도를 재는 척도

# In[43]:


# R^2의 값을 구하기- 결정계수 구하기
print("훈련 데이터 세트 점수 : {:.2f}".format(model.score(X_train, y_train)))
print("테스트 데이터 세트 점수 : {:.2f}".format(model.score(X_test, y_test)))


# ### 실습 과제 1
# * 아래 대회에서 데이터 셋을 다운로드 후, 다중선형 회귀 모델을 만들어보자.
#    * URL : https://www.kaggle.com/c/2019-2nd-ml-month-with-kakr/data
# * MAE, MSE, RMSE를 구해보자

# ### 도전
# * 다중 선형 회귀 모델을 만들고 이를 예측을 수행한 후, 제출해 보자.
