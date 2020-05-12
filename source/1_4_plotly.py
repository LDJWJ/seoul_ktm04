#!/usr/bin/env python
# coding: utf-8

# ### Plotly 소개
#  * d3.js를 이용하여 interactive하게 그래프를 보여준다.

# ### 사전 파이썬 버전 확인
# ```
# (base) C:\Users\toto>python --version
# Python 3.7.7
# ```

# ### plotly를 pandas와 함께 사용하는 법
#  * cufflinks 설정과 .iplot()을 활용. pandas.plot()와 같이 판다스 데이터 시각화
#  * plotly.express 라이브러리 활용

# ### cuffilinks 는 무엇
#  * 판다스 데이터 프레임과 Plotly를 연결하여 사용자가 판다스로부터 직접 시각화를 할 수 있는 라이브러리

# ### 01 시작하기 - 설치(Plotly and Cufflinks)
#  * pip install plotly
#  * pip install cufflinks

# In[8]:


import plotly
import cufflinks as cf
import pandas as pd
import numpy as np


# ### 버전 확인

# In[9]:


print(plotly.__version__)
print(cf.__version__)
print(pd.__version__)
print(np.__version__)


# In[10]:


#Enabling the offline mode for interactive plotting locally
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
init_notebook_mode(connected=True)
cf.go_offline()


# ### 데이터 생성 및 plot

# In[15]:


#create Data
df = pd.DataFrame(np.random.randn(100,4),    # 100개 4개 컬럼 
                  columns='A B C D'.split())

print(df.shape)
df.head()


# In[22]:


df2 = pd.DataFrame({'items':['bag','apple','cap'],'Values':[32,43,50,]})
df2


# ### Line Plot

# In[23]:


df.iplot()


# ### Scatter Plot

# In[24]:


df.iplot(kind='scatter', x='A',y='B',mode='markers',size=20)


# ### Bar Plot

# In[26]:


df2.iplot(kind='bar',x='items',y='Values')


# In[29]:


df = pd.DataFrame(np.random.rand(10,4), 
                  columns=['A', 'B', 'C', 'D'])
df.head()


# In[30]:


df.iplot(kind='bar')


# ### A컬럼만 보기

# In[32]:


df['A'].iplot(kind='bar')


# ### stack plot

# In[31]:


df.iplot(kind='bar', barmode='stack')


# In[33]:


df.iplot(kind='barh', barmode='stack')


# ### Box Plot

# In[27]:


df.iplot(kind='box')


# ### 3D Surface Plot

# In[21]:


df3 = pd.DataFrame({'x':[1,2,3,4,5],
                    'y':[10,20,30,40,60],
                    'z':[5,4,3,2,1]})
df3


# In[28]:


df3.iplot(kind='surface',colorscale='rdylbu')


# ### Line Charts

# In[5]:


df = cf.datagen.lines()
df.head()


# In[6]:


df.iplot(kind='line')


# In[38]:


print(df.shape)
df.head(10)


# ### Plot Styling

# ### 테마(Theme) 설정

# In[39]:


themes = cf.getThemes()
themes


# In[40]:


data = pd.Series(range(10))
for theme in themes:
    data.iplot(kind='bar', theme=theme, title=theme)


# ### 테마 설정

# In[41]:


cf.set_config_file(theme='pearl')


# ## Plotly express 사용한 시각화

# * cufflinks보다 좀 더 다양하며, 사용방법은 seaborn과 비슷함.
# * plotly_express 이용. plotly 4.1 부터는 별도 설치 없어도 됨. 3.8.1의 경우 설치 필요

# In[42]:


import plotly.express as px


# In[43]:


# iris 데이터 불러오기
print(px.data.iris.__doc__)
px.data.iris().head()


# ### 산점도(scatter plot) and Line Plots(선 그래프)

# In[45]:


import plotly.express as px
df = px.data.iris()
fig = px.scatter(df, x="sepal_width", y="sepal_length")
fig.show()


# In[46]:


import plotly.express as px
df = px.data.iris()
fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species")
fig.show()


# In[47]:


import plotly.express as px
df = px.data.iris()
fig = px.scatter(df, 
           x="sepal_width", y="sepal_length", 
           color="species", marginal_y="violin",
           marginal_x="box", trendline="ols")
fig.show()


# In[50]:


import plotly.express as px
df = px.data.iris()
fig = px.scatter_matrix(df, dimensions=["sepal_width", 
                                        "sepal_length", 
                                        "petal_width", 
                                        "petal_length"], 
                        color="species")
fig.show()


# In[51]:


import plotly.express as px
df = px.data.tips()
fig = px.parallel_categories(df, color="size", color_continuous_scale=px.colors.sequential.Inferno)
fig.show()


# In[57]:


df = px.data.gapminder()
print(df.shape)
print(df.columns)
print(px.data.gapminder.__doc__)


# In[52]:


import plotly.express as px
df = px.data.gapminder()
fig = px.scatter(df.query("year==2007"), 
                 x="gdpPercap", 
                 y="lifeExp", 
                 size="pop", 
                 color="continent",
                 hover_name="country", log_x=True, size_max=60)
fig.show()


# In[58]:


import plotly.express as px
df = px.data.gapminder()
fig = px.scatter(df, x="gdpPercap", y="lifeExp", 
                 animation_frame="year", 
                 animation_group="country",
                 size="pop", 
                 color="continent", 
                 hover_name="country", 
                 facet_col="continent",
                 log_x=True, size_max=45, range_x=[100,100000], range_y=[25,90])
fig.show()


# ### Barplot

# In[63]:


import plotly.express as px
df = px.data.tips()
fig = px.bar(df, x="sex", y="total_bill", color="smoker", barmode="group")
fig.show()


# ### 3D

# In[65]:


df = px.data.election()
print(df.shape)
print(df.head())
print(df.columns)
print(px.data.election.__doc__)


# In[66]:


import plotly.express as px
df = px.data.election()
fig = px.line_3d(df, x="Joly", y="Coderre", z="Bergeron", color="winner", line_dash="winner")
fig.show()


# ### Maps

# In[68]:


import plotly.express as px
df = px.data.gapminder()
fig = px.scatter_geo(df, 
                     locations="iso_alpha", 
                     color="continent", 
                     hover_name="country", 
                     size="pop",
               animation_frame="year", projection="natural earth")
fig.show()


# In[69]:


import plotly.express as px
df = px.data.gapminder()
fig = px.line_geo(df.query("year==2007"), locations="iso_alpha", color="continent", projection="orthographic")
fig.show()


# ### REF
#  * cufflinks.datagen module
#  * https://jpoles1.github.io/cufflinks/html/cufflinks.datagen.html
#  
# * Plotly Express in Python
# * https://plot.ly/python/plotly-express/#plotly-express
