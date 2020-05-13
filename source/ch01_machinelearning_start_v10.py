#!/usr/bin/env python
# coding: utf-8

# ## ch01 머신러닝 시작하기

# * Machine Learning with sklearn @ DJ,Lim
# * date : 20/04/05

# ### 01 기본 개념 이해하기

# * <b>샘플(sample) or 데이터 포인터(data point)</b> : 하나의 개체 혹은 행을 말한다.
# * 특성 or 속성(feature) : 샘플의 속성, 즉 열을 가르킨다.

# ### 02 기본 라이브러리 이해하기

# ### scikit-learn(사이킷 런)
# * 오픈 소스입니다.
# * 매우 인기 높고 독보적인 파이썬 머신러닝 라이브러리입니다.
# * url : http://scikit-learn.org/stable/documentation
# * 사용자 가이드 : https://scikit-learn.org/stable/user_guide.html

# ### Numpy
# * 파이썬으로 과학 계산을 하기 위한 꼭 필요한 패키지
# * 다차원 배열을 위한 기능
# * 선형 대수 연산 기능
# * 푸리에 변환 같은 고수준 수학 함수와 유사 난수 생성기 기능
# * url : https://www.numpy.org/

# ### SciPy
# * SciPy(https://www.scipy.org/scipylib) 과학 계산용 함수를 모아놓은 파이썬 패키지.
# * 고성능 선형대수 기능, 함수 최적화, 신호 처리, 특수한 수학 함수와 통계 분포 등
# * 희소 행렬 기능

# ### Matplotlib
# * 파이썬 대표적인 과학 계산용 그래프 라이브러리

# ### Pandas
# * 데이터 처리와 분석을 위한 파이썬 라이브러리
# * url : https://pandas.pydata.org/
# * R의 data.frame을 본떠 설계한 **Dataframe**이라는 데이터 구조를 기반으로 만들어짐.
# * SQL 처럼 테이블 쿼리나 조인을 수행 가능함.
# * xlsx, csv등의 다양한 파일과 데이터베이스에서 데이터를 읽어들 일 수 있음.
# * 참고 도서 : 파이썬 라이브러리를 활용한 데이터 분석
# - https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html

# ### mglearn
# * 깃허브에 있는 코드와 함께 작성.

# In[1]:


from IPython.display import display, Image


# ### 03 라이브러리 소프트웨어 버전 확인
# * 라이브러리이름.\__version__
# * 라이브러리이름.version

# In[2]:


import sys
print("파이썬 버전 :", sys.version)


# In[3]:


import pandas as pd
print("판다스 버전 :", pd.__version__)


# In[4]:


import matplotlib
import numpy as np
import scipy as sp


# ### 직접 해보기1 
# * matplotlib, numpy, scipy의 각각의 버전을 확인해 보자.

# ### 04  머신러닝 모델을 위한 iris 데이터를 준비

# ### 데이터 : 붓꽃
# * 종류 : setosa, versicolor, virginica
# * 데이터 내용 : 붓꽃의 꽃잎과 꽃받침
# * 우리가 해결하려고 하는 문제 : 데이터를 주고 붓꽃의 종류 예측하기

# In[5]:


display(Image(filename='img/iris_setosa01.png'))


# ### 용어 이해하기
# * <b>클래스(class)</b> : 출력될 수 있는 값들. 붓꽃의 종류들, 붓꽃의 종류는 세 클래스 중 하나에 속한다.
# * <b>레이블(label)</b> : 데이터 포인트 하나(붓꽃 하나)에 대한 기대 출력. 특정 데이터 포인트에 대한 출력

# ### 데이터 준비

# In[6]:


from sklearn.datasets import load_iris
iris = load_iris()
iris


# In[7]:


# iris 데이터 셋의 key값들
print(iris.keys())
print(iris['target_names']) # 붓꽃의 label의 종류명
print(iris['target'])   # 붓꽃의 종류의 label의 값 
print(iris['feature_names'])  # 붓꽃의 꽃잎과 꽃받침의 feature 이름 
print(iris['data'])  # 붓꽃의 꽃잎과 꽃받침의 값


# In[8]:


# iris 데이터 셋의 설명 확인
iris['DESCR']


# In[9]:


# iris 데이터 셋의 행열 확인
print( iris['data'].shape )
print( iris['feature_names'])
print( iris['data'][:5])      # 5개의 데이터 확인
print( iris['target_names'][:5]) 
print( iris['target'][:5]) 


# ### 데이터의 크기 확인 
# * 데이터의 사이즈 확인 : 데이터.shape
# * 데이터의 자료형 확인 : type(데이터자료형) 

# In[10]:


print(iris['target'].shape)  # 타켓
print(iris['data'].shape)
print(type(iris['target']) , type(iris['data']) )


# ### 05 데이터를 훈련 데이터와 테스트 데이터로 나누기

# * 훈련 데이터 : 실제 공부를 위한 데이터 셋(실제 문제지의 문제)
# * 테스트 데이터 : 공부 후, 실제 잘 동작하는지 확인하기 위한 데이터 (모의고사시험)
# * 내용 : 모델을 새 데이터에 적용하기 전에 우리가 만든 모델이 잘 동작하는지 확인하기 위해 테스트 데이터를 활용하여 평가한다.

# #### 훈련 데이터 셋(training set) : 머신러닝 모델을 만들 때 쓰는 데이터 셋
# #### 테스트 데이터 셋(test set) : 모델이 얼마나 잘 작동하는지 쓰는 데이터 셋
# * 테스트 데이터 셋을 또는 홀드아웃(hold-out set)이라 한다.

# * scikit-learn은 데이터 셋을 나눠주기 위해 train_test_split 함수를 이용.
# * train_test_split 함수는 기본적으로 75% 훈련 세트, 25%의 테스트 세트

# In[11]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris['data'], 
                                                    iris['target'],
                                                   random_state=0)


# In[12]:


# 데이터 사이즈
print(X_train.shape)   # 훈련 데이터 셋 사이즈
print(X_test.shape)    # 테스트 데이터 셋 사이즈
print(y_train.shape)   # 훈련 데이터 레이블 사이즈 
print(y_test.shape)    # 테스트 데이터 레이블 사이즈


# ### 06. 데이터 살펴보기 - 시각화
# * 머신러닝 모델을 만들기 전 머신러닝 없이도 풀 수 있는 문제는 아닌지, 혹은 필요한 정보가 누락이 없는지 확인
# * 산점도(SCATTER PLOT)를 이용하여 확인.
# * 2개의 변수만 사용 가능하여 산점도 행렬(scatter matrix)를 사용

# In[13]:


import seaborn as sns


# In[14]:


iris_df = pd.DataFrame(X_train, columns=iris.feature_names)
iris_df['y'] = y_train
iris_df['y'] = iris_df['y'].astype('category')


# In[15]:


print(iris_df.shape)
print(iris_df.info())


# In[16]:


sns.pairplot(iris_df.iloc[ : ,0:4])   # 1~4열 선택


# In[17]:


pd.plotting.scatter_matrix(iris_df, c=y_train,     # 색 
                          figsize=(15,15),         # 크기 
                          marker='o',
                          hist_kwds={'bins':20},   # 막대의 개수
                          s=60,     # size
                          alpha=0.8 )  # 투명도


# ### 07 첫번째 머신러닝 모델 만들기
# * k-최근접 이웃(k-nearest neighbors, k-NN) 알고리즘 : 
#   * 훈련 데이터에서 새로운 데이터 포인트에 가장 가까운 'k개'의 이웃을 찾는다.
#   * 이웃들의 클래스 중 빈도가 가장 높은 클래스를 예측값으로 사용
# 

# ### 모델 만들기 

# In[18]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)


# ### 모델 학습시키기

# In[19]:


knn.fit(X_train, y_train)


# ### 새로운 데이터로 예측해 보기

# In[20]:


X_new = np.array([[5, 2.9, 1, 0.2]])


# In[21]:


### 예측시키기
pred = knn.predict(X_new)
pred_targetname = iris['target_names'][pred]
print("예측 : ", pred)
print("예측한 타깃의 이름: ", pred_targetname)


# ### 08 내가 만든 모델 평가하기

# In[22]:


y_pred = knn.predict(X_test)
print("예측값 :\n", y_pred)


# In[23]:


print("테스트 세트의 정확도 : {:.2f}".format(np.mean(y_pred == y_test)))


# ### 실습해 보기
# * titanic 데이터 셋을 활용하여 knn 모델을 구현해 보자

# ### REF
# sklearn score 매개변수 : https://scikit-learn.org/stable/modules/model_evaluation.html
