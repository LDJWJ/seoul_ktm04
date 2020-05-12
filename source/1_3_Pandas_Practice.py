#!/usr/bin/env python
# coding: utf-8

# ### 라이브러리 불러오기

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os 


# In[3]:


import pandas as pd
data_tr = pd.read_csv("california_housing_train.csv")
data_test = pd.read_csv("california_housing_test.csv")

print("캘리포니아 데이터 행열(train) :", data_tr.shape)
print("캘리포니아 데이터 행열(test) :", data_test.shape)


# ### 일부 행만 보기

# In[4]:


print(data_tr.head())     # 5행만 보기
print(data_test.head(3)) # 3행만 보기
# tail를 이용하여 뒤에 5행을 볼 수 있음.
data_tr.tail()     # 뒤에서 5행만 보기


# ### 파일 만들기(csv, excel)

# In[5]:


### 구글 콜랩의 데이터 셋을 이용해서 csv, excel파일을 만들자.
data_tr.to_csv("data.csv", index=False)
data_tr.to_excel("data.xlsx", index=True)

### 파일을 불러와 보자.
excel_data = pd.read_excel("data.xlsx")
csv_data = pd.read_csv("data.csv")


# In[6]:


### 파일 생성 확인(리눅스 명령 이용)
get_ipython().system('ls')


# In[7]:


excel_data.head()


# In[8]:


csv_data.head()


# In[9]:


data_tr = pd.read_csv("sample_data/california_housing_train.csv")
data_test = pd.read_csv("sample_data/california_housing_test.csv")


# ### 데이터의 컬럼명 확인

# In[10]:


print("데이터 열의 제목(train) : ", data_tr.columns) 
print("데이터 열의 제목(test) : ", data_test.columns) 


# ### 데이터의 정보 확인
#   * 컬럼의 자료형, 행열, 비어있는 값의 개수

# In[11]:


print(data_tr.info() ) 


# In[12]:


print(data_test.info() ) 


# ### 데이터의 빈 값을 채우기

# In[13]:


import numpy as np
data_tr.iloc[0,1] = np.nan
data_test.iloc[2,1:8] = np.nan

print(data_tr.head() ) 
print(data_test.head() ) 


# In[14]:


print("train 데이터 셋 : ", data_tr.info() ) 
print()
print("test 데이터 셋 : ", data_test.info() ) 


# In[15]:


import pandas as pd


# ### 데이터 선택(iloc : 인덱스로 선택함)

# In[19]:


base_dir = "./"
data_tr = pd.read_csv(base_dir + "california_housing_train.csv")
data_test = pd.read_csv(base_dir + "california_housing_test.csv")

# 두번째 행 선택
print(data_tr.iloc[1, :])


# In[20]:


data_tr.shape


# In[21]:


print(data_tr.iloc[2:4])    # (2+1)행부터 4행까지 선택
print(data_tr.iloc[:4])     # 처음부터 4행까지 선택
print(data_tr.iloc[16997:]) # 16998행부터 끝까지 선택


# In[22]:


data_tr.head(3)


# ### 데이터의 열선택
#  * 컬럼명으로 데이터를 선택하기

# In[23]:


# 열 선택(열 컬럼명으로 선택)
print("latitude 컬럼명으로 선택")
col_sel = data_tr['latitude']
print(col_sel.head())


# ### 데이터의 열선택(iloc 이용)

# In[24]:


# 두번째 열 선택(열도 0부터 시작하므로 지정값의 +1열)
print("두번째 열 선택")
col_sel = data_tr.iloc[:,1]
print( col_sel.head() )


# In[25]:


# 복수열 선택 : 3개 열 선택(컬럼명으로)
print("컬럼명으로 선택_3개 컬럼")
column_name = ['latitude', 'total_rooms', 'population']
row_sel = data_tr[column_name]
print( row_sel.head() )


# In[26]:


# 복수열 선택 : 3개 열 선택(숫자 이용)
print("컬럼명으로 선택_3개 컬럼")
row_sel = data_tr.iloc[:,[1,3,5]]
print( row_sel.head() )


# In[27]:


# 일부 데이터 선택 (iloc 를 이용 행과 열에 접근)
# 1~10행, longitude, latitude, total_rooms, population에 접근 
print("데이터의 일부 가져오기")
dat_part = data_tr.iloc[0:10,[0,1,3,5]]
print( dat_part.head() )


# ### 지도 시각화
#  * folium 을 이용하기

# In[28]:


import folium


# ### 데이터 셋의 경도, 위도 정보를 이용해서 이에 대한 위치를 지도위에 표시

# In[29]:


lat_m = dat_part['latitude'].mean()  # 위도 위치의 평균
log_m = dat_part['longitude'].mean() # 경도 위치의 평균
rooms_m = dat_part['total_rooms'].mean() # 총 방수의 평균
pop_m = dat_part['population'].mean() # 인구의 평균

print("위도, 경도", lat_m , log_m )
print("방, 인구(평균", rooms_m , pop_m )

# 지도 중심위치 및 확대
map1 = folium.Map(location=[lat_m, log_m], zoom_start=7)
# Marker 설명(집, 인구) 
des = "room : " + str(rooms_m) + "<br>" + "pop :" + str(pop_m) 

folium.Marker([lat_m, log_m], popup=des).add_to(map1) # 마커 추가
map1


# In[31]:


## 컬럼명 변경
dat_part.columns = ['long', 'lat', 'tot_rooms', 'pop']
dat_part.columns


# In[32]:


df = dat_part.copy()
df.describe()


# In[33]:


# 여러개의 데이터 표시
# 데이터 셋 복사 및 컬럼명 변경
df = dat_part.copy()
df.columns = ['long', 'lat', 'tot_rooms', 'pop']
map2 = folium.Map(location=[lat_m, log_m], zoom_start=9)

# 추후 색 지정을 위한 함수
def color(pop_num): 
    if pop_num in range(0,1000): 
        col = 'green'
    elif pop_num in range(1001,1999): 
        col = 'blue'
    elif pop_num in range(2000,2999): 
        col = 'orange'
    else: 
        col='red'
    return col


# In[34]:


for lat,lan,room,pop in zip(df['lat'],df['long'],df['tot_rooms'],df['pop']): 
    # as a list as an argument 
    folium.Marker(location=[lat,lan],popup = "room:" + str(room), 
                  icon= folium.Icon(color=color(pop), 
                  icon_color='yellow',icon = 'cloud')).add_to(map2) 


# Save the file created above 
map2.save('test7.html')
map2


# ### 7.5 조건을 이용한 데이터 선택

# In[35]:


import seaborn as sns
sns.distplot(data_tr['longitude'])


# In[36]:


data_tr_long = data_tr[data_tr.longitude <= -120]
sns.distplot(data_tr_long['longitude'])


# In[37]:


data_tr_long = data_tr[ (data_tr.longitude >= -123) & (data_tr.longitude <= -121) ]
sns.distplot(data_tr_long['longitude'])


# In[38]:


data_tr_long = data_tr.loc[ (data_tr.longitude >= -123) & (data_tr.longitude <= -121) ]
sns.distplot(data_tr_long['longitude'])


# In[39]:


data_tr_long = data_tr[ (data_tr.longitude <= -122) | (data_tr.longitude >= -116) ]
sns.scatterplot(x="longitude", y="latitude", data=data_tr_long)


# ### 두개의 데이터 셋을 하나로 만들기
#   * append 함수를 이용

# In[41]:


base_dir = "./"

data_tr = pd.read_csv(base_dir + "california_housing_train.csv")
data_test = pd.read_csv(base_dir + "california_housing_test.csv")

data_all = data_tr.append(data_test)
print(data_all.shape)
print(data_all.iloc[16995:17005])


# ### 데이터의 인덱스 번호를 초기화
#   * [].reset_index() 함수를 사용

# In[42]:


data_all = data_all.reset_index(drop=True)
print(data_all.iloc[16995:17005])


# In[43]:


data_all.describe()


# In[44]:


sns.scatterplot(x="longitude", y="latitude", data=data_all)


# In[45]:


print(data_all.describe())
### -122미만, -122이상~-119미만, -119이상~-118미만, -118이상
data01 = data_all[data_all['longitude'] < -122]
data02 = data_all[ (data_all['longitude'] >= -122) &
                    (data_all['longitude'] < -119) ]
data03 = data_all[ (data_all['longitude'] >= -119) &
                    (data_all['longitude'] < -118) ]
data04 = data_all[data_all['longitude'] >= -118]

f, axes = plt.subplots(2, 2, figsize=(12, 12))
axes[0,0].set_ylim([30, 45])
axes[0,1].set_ylim([30, 45])
axes[1,0].set_ylim([30, 45])
axes[1,1].set_ylim([30, 45])

sns.scatterplot(x="longitude", y="latitude", data=data01, ax=axes[0, 0])
sns.scatterplot(x="longitude", y="latitude", data=data02, ax=axes[0, 1])
sns.scatterplot(x="longitude", y="latitude", data=data03, ax=axes[1, 0])
sns.scatterplot(x="longitude", y="latitude", data=data04, ax=axes[1, 1])


# In[46]:


data01.to_csv('data_01.csv', index=False)
data02.to_csv('data_02.csv', index=False)
data03.to_csv('data_03.csv', index=False)
data04.to_csv('data_04.csv', index=False)

print(os.listdir())


# ### 데이터를 그룹화하여 요약 값 확인
#  * 네 개의 데이터를 읽어온다.
#  * 각각의 데이터 셋에 새로운 변수(컬럼)을 생성.

# In[47]:


data01 = pd.read_csv("data_01.csv")
data02 = pd.read_csv("data_02.csv")
data03 = pd.read_csv("data_03.csv")
data04 = pd.read_csv("data_04.csv")

print("각 데이터 행열 : {}".format(data01.shape) )
print("각 데이터 행열 : {}".format(data02.shape) )
print("각 데이터 행열 : {}".format(data03.shape) )
print("각 데이터 행열 : {}".format(data04.shape) )


# In[48]:


# 새로운 컬럼 생성
data01['group_lat'] = 1
data02['group_lat'] = 2
data03['group_lat'] = 3
data04['group_lat'] = 4


# ### 한번에 여러개 파일 합치기

# In[49]:


data_all = data01.append([data02, data03, data04], ignore_index = True)
data_all.group_lat.unique()  # 새 컬럼의 유일한 값 확인


# ### 새로운 컬럼을 활용하여 집 값의 중위값을 표시

# In[50]:


print("전체 데이터 행열 :{}".format(data_all.shape))
print("전체 데이터 컬럼명 :{}".format(data_all.columns))
print("group_lat 컬럼의 값 : {}".format(data_all.group_lat.unique()) )
sns.barplot(x="group_lat", y="median_house_value", data=data_all)


# ### 값들의 데이터 개수 알아보기

# In[51]:


print(data_all['group_lat'].value_counts() )


# In[52]:


print(pd.value_counts(data_all['group_lat']) )


# ### 데이터를 그룹화하여 이에 대한 요약값을 확인하기

# In[53]:


### 지역별 집 값 알아보기
grouped = data_all.groupby('group_lat')  # 그룹화
print( "행정구역 인구 평균 :\n ", grouped.mean()['population'] )    # 행정 구역 인구 데이터의 평균
print( "소득 평균 :\n ",grouped.mean()['median_income'] ) # 소득 데이터의 평균
print( "방 개수 평균 :\n ",grouped.mean()['total_rooms'] )   # 방 개수 데이터의 평균
print( "세대 수 평균 :\n ",grouped.mean()['households'] )    # 세대 수 데이터의 평균


# ### 그룹화한 결과를 시각화하여 확인하기

# In[54]:


# 2행 2열의 그래프 만들고 전체 크기 지정 
fig ,axes = plt.subplots(nrows=2, ncols=2)  # 2행 2열의 구조 
fig.set_size_inches(12,12)  # 전체 크기

# 그룹화하기
grouped = data_all.groupby('group_lat').mean()

# 막대 그래프로 확인해 보기 
sns.barplot(x=grouped.index, y="population", data=grouped, ax= axes[0][0])
sns.barplot(x=grouped.index, y="median_income", data=grouped, ax= axes[0][1])
sns.barplot(x=grouped.index, y="total_rooms", data=grouped, ax= axes[1][0])
sns.barplot(x=grouped.index, y="households", data=grouped, ax= axes[1][1])

# 각각의 그래프에 제목을 넣기
axes[0][0].title.set_text('population')
axes[0][1].title.set_text('median_income')
axes[1][0].title.set_text('total_rooms')
axes[1][1].title.set_text('households')


# In[55]:


### 지역별 집 값 알아보기
grouped = data_all.groupby('group_lat')  # 그룹화
print( grouped.mean()['median_house_value'] )  # 지역별 집 값 평균
print( grouped.sum()['median_house_value'] )  # 지역별 집값의 전체 합 
print( grouped.std()['median_house_value'] )  # 지역별 집값의 표준편차

grouped = grouped.mean()
sns.barplot(x=grouped.index, y="median_house_value", data=grouped)

