#!/usr/bin/env python
# coding: utf-8

# ### 판다스를 활용한 데이터 이해

# In[1]:


myfood = ['banana', 'apple', 'candy']
print(myfood[0])
print(myfood[1])
print(myfood[2])
print(myfood[1:3]) # 첫번째 두번째 가져오기


# In[2]:


for item in myfood:
    print(item)


# ### 딕셔너리(Dictionary)

# In[3]:


dict1 = {'one':'하나', 'two':"둘", 'three':'셋'}
dict2 = {1:"하나", 2:"둘", 3:"셋"}
dict3 = {'col1':[1,2,3], 'col2':['a','b','c']}


# In[4]:


print(dict1)
print(dict2)
print(dict3)


# In[5]:


print(dict1['one'])
print(dict2[2])
print(dict3['col2'])


# ### 판다스 모듈 불러오기

# In[7]:


import pandas as pd  # pandas를 불러오고 pd로 약자로서 쓰겠다.


# In[8]:


from pandas import Series, DataFrame  # pandas안의 Series와 DataFrame 를 불러옴.


# In[9]:


print("pandas 버전 : ", pd.__version__)


# ### 홍길동 팀별 대항 게임 5일간의 점수

# In[10]:


score = Series( [1000,14000, 3000, 3000, 1000] )
print(score)


# In[11]:


print("자료형 확인 : ", type(score))


# In[12]:


## Series 인덱스 확인


# In[13]:


print(score.index)


# In[14]:


## Series 값 확인
print(score.values)


# ### 판다스 시리즈 인덱스 지정

# In[15]:


score = Series( [1000, 14000, 3000], 
                index=['2019-05-01', '2019-05-02', '2019-05-03'])
print(score)


# In[16]:


print(score['2019-05-01']) # 인덱스 이용 - 5월 1일 날짜 점수 확인
print("-------------")
print(score['2019-05-02':'2019-05-03']) # 5월 2일, 3일 날짜 팀 점수 확인


# In[17]:


for idx in score.index:
    print(idx)


# In[18]:


for value in score.values:
    print(value)


# ### 두 팀의 팀점수 합산해보기
#  * 길동팀의 3일간의 점수와 toto 팀의 3일간의 점수

# In[19]:


from pandas import Series


# In[20]:


gildong = Series([1500, 3000, 2500],
   index = ['2019-05-01', '2019-05-02', '2019-05-03'] )
toto = Series([3000, 3000, 2000],
   index = ['2019-05-01', '2019-05-03', '2019-05-02'] )


# In[21]:


gildong + toto


# ### 데이터 프레임의 이해

# * 데이터 프레임의 객체를 생성하는 가장 간단한 방법은 딕셔너리를 이용하는 방법
# * 데이터 프레임은 Series의 결합으로 이루어진 것으로 생각할 수 있음.
# * Pandas(판다스)의 대표적인 기본 자료형이다.
# * DataFrame 함수를 이용하여 객체 생성이 가능하다.

# In[22]:


from pandas import DataFrame


# In[23]:


dat = { 'col1' : [1,2,3,4],
        'col2' : [10,20,30,40],
        'col3' : ['A', 'B', 'C', 'D'] }
df = DataFrame(dat)
df


# ### 네 팀의 5일간의 팀별 점수

# * 팀은 toto, gildong, apple, catanddog 팀이다.

# In[24]:


from pandas import DataFrame
team_score = {  "toto":[1500,3000,5000,7000,5500],
                "apple":[4000,5000,6000,5500,4500],
                "gildong":[2000,2500,3000,4000,3000],
                "catanddog":[7000,5000,3000,5000,4000]}
team_df = DataFrame(team_score)
team_df


# In[25]:


date = ['19-05-01','19-05-02', '19-05-03', '19-05-04', '19-05-05']
team_df = DataFrame(team_score,
    columns=['catanddog', 'toto', 'apple', 'gildong'],
    index=date)
team_df


# In[26]:


team_df['toto']


# In[27]:


team_df[ ['toto', 'gildong'] ]


# ### loc와 iloc를 이용한 접근

# * loc는 데이터 프레임의 컬럼명(인덱스)를 사용하여 데이터 추출한다.
# * iloc는 데이터 프레임의 데이터 순서(번호)를 사용하여 데이터 추출(시작번호 : 0)
# * loc[ 행, 열] 접근이라고 쉽게 생각한다.

# In[28]:


print(team_df.loc[ '19-05-02' ] ) # 19-05-02 일
print("-----------")
print(team_df.loc[ ['19-05-02', '19-05-03'] ]) # 5월 2일, 3일
print("-----------")
print(team_df.loc[ '19-05-02': ]) # 5월 2일 이후 전체 데이터 가져오기


# ### loc를 이용한 접근

# In[29]:


## 컬럼명 확인
print(team_df.columns)
print("-----")
print(team_df.loc[:, 'toto']) # 전체행, toto팀
print("-----")
print(team_df.loc[:, ['toto', 'gildong'] ]) # 전체행, toto, gildong팀
print("-----")
print(team_df.loc[:, 'toto': ]) # 전체행, toto 부터 끝까지


# ### iloc 속성을 이용한 행, 열, 데이터 접근하기

# In[30]:


print(team_df.iloc[0]) # 첫번째 행 접근
print("------")
print(team_df.iloc[ [0,1] ]) # 첫번째 두번째 행 접근
print("------")
print(team_df.iloc[ 0:3:1] ) # 첫번째부터 세번째 행 접근
print("------")
range_num = list(range(0,3,1))
print(team_df.iloc[ range_num ] ) # 첫번째부터 세번째 행 접근


# In[31]:


print(team_df.iloc[:, 0]) # 첫번째 열 접근
print("------")
print(team_df.iloc[:, [0,1] ]) # 첫번째 두번째 열 접근
print("------")
print(team_df.iloc[:, 0:3:1] ) # 첫번째부터 세번째 열 접근
print("------")
range_num = list(range(0,3,1))
print(team_df.iloc[:, range_num ] ) # 첫번째부터 세번째 열 접근


# In[32]:


print(team_df.sum() )
print("----")
print(team_df.mean() )
print("----")


# ### 팀별 요약값을 보고 싶다.

# In[33]:


team_df.describe()


# In[34]:


## 날짜별 누적 통계
team_df.cumsum()


# ### 날짜별 합계

# In[35]:


## 날짜별 합계
print(team_df.sum(axis=1))


# In[36]:


rowsum = team_df.sum(axis=1)
print(type(rowsum))


# In[37]:


team_df['rowsum'] = team_df.sum(axis=1)
team_df


# ### 점수가 높은 날짜별로 확인해보자

# In[38]:


team_df.rowsum.sort_values(ascending=False)


# ### 조건을 걸어 일정 이상의 팀 점수의 날만 확인해 보자.

# * 17000이상인 날만 확인해 보기

# In[39]:


team_df[ team_df.rowsum >= 17000]


# In[40]:


team_df


# In[41]:


team_df.drop(['toto', 'gildong' ], axis=1)


# In[42]:


team_12 = team_df.drop(['toto', 'gildong' ], axis=1)
team_12


# In[43]:


team_12.to_csv("team_12.csv", index=False)
team_12.to_excel("team_12.xlsx", index=False)


# In[46]:


# window의 경우
get_ipython().system('dir *team*')

# 리눅스 OS의 경우
# !ls *team*


# ### REF
#  * pandas 공식 사이트 : https://pandas.pydata.org/ (https://pandas.pydata.org/)
#  * pandas 10 minute tutorial : https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html
# (https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html)

# In[ ]:




