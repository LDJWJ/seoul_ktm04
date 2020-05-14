#!/usr/bin/env python
# coding: utf-8

# ## ch2 지도학습 - knn

# * Machine Learning with sklearn @ DJ,Lim
# * date : 20/04

# * 지도학습은 대표적인 머신러닝 방법론중의 하나이다.
# * 지도학습은 입력과 출력 샘플 데이터가 있고, 주어진 입력으로부터 출력을 예측하고자 할 때 사용.
# * 지도학습에는 두가지 종류 <b>분류(classification), 회귀(regression)</b>이 있다. 
# * knn은 사용자가 쉽게 이해할 수 있는 대표적인 지도학습 방법중에 하나로, 분류와 회귀에 다 사용된다.

# ### 학습 내용
# * 01 지도학습의 종류
# * 02 knn 알고리즘 시각화
# * 03 knn을 이용한 유방암 데이터 실습

# ### 01 지도학습의 종류

# ### 분류(Classification)

# * 분류는 가능성 있는 여러 클래스 레이블(class label)중 **하나를 예측하는 것**이다.
# * 분류는 두개의 클래스로 분류하는 <b>이진 분류(binary classification)</b>과 셋 이상의 클래스로 분류하는 <b>다중 분류(multiclass classification)</b>로 나누어진다.
# * 이진 분류는 질문의 답이 예/아니오 등의 예. 이진분류의 양성(positive) 클래스, 음성(negativ) 클래스라고 한다.

# ### 회귀(Regression)

# * 회귀는 연속적인 숫자, 또는 프로그래밍 용어로 말하면 <b>부동소수점수(수학 용어로는 실수)를 예측</b>하는 것. 수치형 데이터를 예측.
# * 예로는 어떤 사람의 나이, 키, 몸무게 정해진 수의 값이 아닌 해당 예측 값은 수치형 데이터.

# ### 02 knn 알고리즘 시각화
# * 설치 : !pip install mglearn

# In[3]:


import mglearn
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# ### knn의 k가 1인 경우의 알고리즘(회귀-수치형 값의 예측)

# In[4]:


mglearn.plots.plot_knn_regression(n_neighbors=1)


# In[5]:


mglearn.plots.plot_knn_regression(n_neighbors=3)


# ### knn의 k가 3인 경우의 알고리즘(분류- 범주형 값의 예측)

# In[6]:


mglearn.plots.plot_knn_classification(n_neighbors=3)


# In[7]:


mglearn.plots.plot_knn_classification(n_neighbors=5)


# ### 일반화, 과대적합, 과소적합

# * 모델이 처음보는 데이터에 대해 예측이 가능하다면 이를 훈련세트에서 테스트 세트로 **일반화(generalization)**되었다고 한다.
# * 아주 복잡한 모델을 만든다면 훈련세트에만 정확한 모델이 된다.(과대적합)  
#    * 과대적합(overfitting)는 모델이 훈련 세트의 각 샘플에 너무 가깝게 맞춰져서 새로운 데이터가 일반화되기 어려울 때 발생.
# * 반대로 모델이 너무 간단해서 잘 예측을 못함.(과소적합-underfitting)

# ### 03 유방암 데이터 셋 실습

# * 데이터 셋 : 위스콘신 유방암(Wisconsin Breast Cancer)데이터 셋
# * 각 종양은 양성(benign-해롭지 않은 종양)과 악성(malignant-암 종양)

# In[8]:


from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt


# In[9]:


cancer = load_breast_cancer()
print("cancer.keys() : \n{}".format(cancer.keys()))
print("유방암 데이터의 행열 : {}".format(cancer.data.shape))


# ### feature 이름, class 이름

# In[10]:


print("특성이름(featuer_names) : {}".format(cancer['feature_names']))
print()
print("클래스 이름(target_names) : {}".format(cancer['target_names']))


# ### 데이터 셋 나누기

# In[11]:


from sklearn.model_selection import train_test_split


# In[12]:


X = cancer.data
y = cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                         stratify=cancer.target,     
                                         random_state=77)


# In[13]:


## target의 확인
print( (y_train == 1).sum(), (y_train == 0).sum() )
print( (y_test == 1).sum(), (y_test == 0).sum() )
print( 267/(267+90), 159/(159+53) )
print( 90/(267+90), 53/(159+53) )


# ### 04 머신러닝 모델 만들고 예측하기

# ### 작업 단계
# <pre>
# (1) 모델 만들기
# (2) 모델 학습 시키기(fit)
# (3) 모델을 이용한 값 예측(predict)
# (4) 훈련 데이터를 이용한 정확도 확인
# (5) 테스트 데이터를 이용한 정확도 확인
# </pre>

# In[14]:


from sklearn.neighbors import KNeighborsClassifier


# In[15]:


model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
pred = model.predict(X_test)
pred


# In[16]:


# 정확도 구하기
(pred == y_test).sum()/(len(pred))


# In[17]:


acc_tr = model.score(X_train, y_train)   # 주어진 데이터를 활용한 확인(자체 예측 후 확인 결과)
acc_test = model.score(X_test, y_test)   # 주어진 데이터를 활용한 확인(자체 예측 후 확인 결과)


# ### score를 이용한 결과 확인

# In[18]:


print("k : {}".format(3))
print("훈련 데이터셋 정확도 : {:.2f}".format(acc_tr))
print("테스트 데이터 셋 정확도 : {:.2f}".format(acc_test))


# ### 05 k의 값에 따른 정확도 확인해 보기

# In[19]:


training_accuracy = []
test_accuracy = []
neighbors_numbers = range(1,11)  # 1~10까지의 값
for n in neighbors_numbers:
    model = KNeighborsClassifier(n_neighbors=n)
    model.fit(X_train, y_train)
    
    acc_tr = model.score(X_train, y_train)
    acc_test = model.score(X_test, y_test)
    training_accuracy.append(acc_tr)
    test_accuracy.append(acc_test)
    
    print("k : {}".format(n))
    print("accuracy of training set : {:.2f}".format(acc_tr))
    print("accuracy of test set : {:.2f}".format(acc_test))


# ### 직접 해보기
# * k을 1부터 100까지 돌려보고 가장 높은 값을 갖는 k의 값을 구해보자.

# ### 실습해 보기
# * titanic 데이터 셋을 활용하여 knn 모델을 구현한다. 
# * 가장 높은 일반화 성능을 갖는 k의 값은 무엇인지 찾아보자.
# 
# ### 더 해보기
# * 이를 그래프로 표현해 보기
# * Bike 데이터 셋을 knn모델을 활용하여 예측해 보기
