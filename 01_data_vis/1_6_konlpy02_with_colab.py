# -*- coding: utf-8 -*-
"""3_2_Konlpy02_with_Colab

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ZuFfWln7MFqi3UsHBFKC_-IUG1NWqV9b

## Colab 환경에서 자연어처리(2)

### colab에서 열때,
<a href="https://github.com/LDJWJ/PythonBasic/blob/master/3_2_Konlpy_Colab" target="_parent">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

### 01 사전 작업
### 02 텍스트 데이터 시각화
### 03 영화 댓글 시각화
"""



"""### 01 사전 작업 
 * 한글 폰트 적용, konlpy 설치
"""

### 나눔 고딕 설치
!apt-get update -qq   # 설치를 업데이트 
!apt-get install fonts-nanum* -qq  # 설치한다. fonts-nanum*

import matplotlib.font_manager as fm # 폰트 관련 용도
import matplotlib.pyplot as plt      # 그래프 그리는 용도

path = '/usr/share/fonts/truetype/nanum/NanumGothicEco.ttf' # 설치된 나눔 글꼴중 원하는 녀석의 전체
font_name = fm.FontProperties(fname=path, size=10).get_name()
print(font_name)
plt.rc('font', family=font_name)

# 우선 fm._rebuild() 를 해주고 # 폰트 매니저 재빌드가 필요하다.
fm._rebuild()

"""### 런타임 재기동 후, 다시 시작"""

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import matplotlib as mpl # 기본 설정 만지는 용도
import matplotlib.pyplot as plt # 그래프 그리는 용도
import matplotlib.font_manager as fm # 폰트 관련 용도
import numpy as np

path = '/usr/share/fonts/truetype/nanum/NanumGothicEco.ttf' # 설치된 나눔글꼴중 원하는 녀석의 전체
font_name = fm.FontProperties(fname=path, size=10).get_name()
print(font_name)
plt.rc('font', family=font_name)

## 음수 표시되도록 설정
mpl.rcParams['axes.unicode_minus'] = False

# 데이터 준비
data = np.random.randint(-200, 100, 50).cumsum()

# 그래프를 그려 한글 확인
plt.plot(range(50), data, 'r')
plt.title('시간별 가격 추이')

"""### 웹 환경이 아닌 개인 컴퓨터에서의 한글 폰트 설정
```
from matplotlib import font_manager, rc
import matplotlib.pyplot as plt
import platform

path = "C:/Windows/Fonts/malgun.ttf"  # 한글 폰트 위치 지정
if platform.system() == "Windows":  # 사용 OS가 Windows의 경우
    font_name = font_manager.FontProperties(fname=path).get_name()
    rc('font', family=font_name)
elif platform.system()=="Darwin":   # 사용 OS가 Mac인 경우
    rc('font', family='AppleGothic')
else:
    print("Unknown System")
```

### konlpy 설치
"""

pip install konlpy

import nltk
from konlpy.tag import Kkma      ### 꼬꼬마
from konlpy.tag import Hannanum  ### 한나눔

### wordcloud와 이미지 표시
from wordcloud import WordCloud, STOPWORDS
from PIL import Image

"""### 02 텍스트 데이터 시각화"""

### 데이터 읽기
text = open("alice.txt").read()
text

## 집합 확인
s2 = set("Hello")
s2

### 불용어 단어 확인 
print( type(STOPWORDS) )
print(STOPWORDS)

### 불용어 단어 추가
stopwords = set(STOPWORDS)
stopwords.add("said")
stopwords

"""### 앨리스 이미지 확인"""

# alice_mask = np.array(Image.open("alice_color.png"))
alice_mask = np.array(Image.open("alice_color.png"))

### 워드 클라우드 표현을 위한 데이터 생성
wc = WordCloud( background_color='white', 
               max_words=2000,
               mask=alice_mask,   
               contour_width=3,
               contour_color="steelblue" )
wc.generate(text)
wc.words_

"""* interpolation 참조 : https://matplotlib.org/3.2.1/gallery/images_contours_and_fields/interpolation_methods.html"""

plt.figure(figsize=(15,8))  # 크기
plt.imshow(alice_mask, cmap=plt.cm.gray, interpolation='bilinear')
plt.axis('off')
plt.show()

"""### 위에서 생성한 wc 데이터를 이용하여 그래프 표시"""

plt.figure(figsize=(15,8))
plt.imshow(wc, interpolation="bilinear") 
plt.axis("off")
plt.show()

"""### 03 영화 댓글 시각화

*  분노의 질주 - 댓글 분석
"""

doc_ko = open("15_TheExtreme_utf8.txt").read()
doc_ko[1:1000]

# OKT 클래스를 이용한 명사확인
from konlpy.tag import Okt       ### Okt

t = Okt()
doc_nouns = t.nouns(doc_ko)
doc_nouns

# nltk.Text()를 이용하여 nltk가 가지는 많은 기능을 사용 가능해짐.
ko = nltk.Text(doc_nouns, name="분노의 질주")
print(type(ko))
print(len(ko.tokens))

### 단어들의 사용 횟수 확인 - 빈도 분석
ko.vocab()

most_fre = new_ko.vocab().most_common(50)
most_fre

### 중복된 단어를 제거한 개수를 확인
print(len(set(ko.tokens)))

plt.figure(figsize=(12, 6))
ko.plot(50)
plt.show()

### 한글에서는 따로 불용어 사전이 없어, 따로 만들거나 또는 파일로 부터 불러올 수 있다.
stop_words = ['분노', '영화', '액션', '시리즈', '더', 
              '그', '이', '것', '또', '좀', 
              '돈', '것', '다음', '질주', '그냥', 
              '분노의질주', '말', '뭐', '애', '나', '듯', '편', '볼', '점', '중', '로']

new_ko = [ ]
for one_word in ko:
  if one_word not in stop_words:
    new_ko.append(one_word)

### nltk Text 객체 만들기
new_ko = nltk.Text(new_ko, name="분노의 질주2")
plt.figure(figsize=(12,6))
new_ko.plot(50)

"""### 텍스트의 단어어 분포 확인 (dispersion_plot)"""

plt.figure(figsize=(15,8))
new_ko.dispersion_plot(['스토리', '대박', '스트레스', '브라이언'])

from wordcloud import WordCloud, STOPWORDS

import numpy as np
from PIL import Image

Car_mask = np.array(Image.open("Draw_car1.png"))

data = new_ko.vocab().most_common(1000)

### 워드 클라우드 표현을 위한 데이터 생성
### 약간의 시간이 필요.
wc = WordCloud(background_color='white', 
               max_words=1000,
               mask=Car_mask,   
               contour_width=3, 
               contour_color="steelblue", 
               font_path=path).generate_from_frequencies(dict(data))

plt.figure(figsize=(12,8))
plt.imshow(wc)
plt.axis("off")
plt.show()

