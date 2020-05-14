# -*- coding: utf-8 -*-
"""3_2_Konlpy01_with_Colab

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1k6arn6UIggzmOde8Ug-dh9RYyogyTgbr

### Colab 환경에서의 자연어 처리 시작하기

#### 1-1 한글 폰트 설정
#### 1-2 한글 적용 확인
#### 1-3 konlpy 설치
#### 1-4 한글 엔진을 이용한 간단한 예제
"""

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import matplotlib as mpl             # 기본 설정 만지는 용도
import matplotlib.pyplot as plt      # 그래프 그리는 용도
import matplotlib.font_manager as fm # 폰트 관련 용도

"""### colab 환경에서 한글 적용을 위한 나눔 고딕 설치"""

### 나눔 고딕 설치  
!apt-get update -qq   # 설치를 업데이트   -qq  : 로그를 최소한으로
!apt-get install fonts-nanum* -qq # 설치한다. fonts-nanum*   =>  ttf-nanum, ttf-nanum-coding, ttf-nanum-extra ]

path = '/usr/share/fonts/truetype/nanum/NanumGothicEco.ttf' # 설치된 나눔 글꼴중 원하는 녀석의 전체
font_name = fm.FontProperties(fname=path, size=10).get_name()
print(font_name)
plt.rc('font', family=font_name)

# 우선 fm._rebuild() 를 해주고 # 폰트 매니저 재빌드가 필요하다.
fm._rebuild()

"""### 런타임 재기동 후, 
 * (방법 1) CTRL + M . 을 실행 
 * (방법 2) 메뉴의 런타임 선택 후, 런타임 다시 시작 선택
 * 데이터 준비
 * 라이브러리 import 
 * 폰트 설정 후, 확인
"""

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
data

# 그래프를 그려보자. 이번에는 정상
plt.plot(range(50), data, 'r')
plt.title('시간별 가격 추이')
plt.ylabel('주식 가격')
plt.xlabel('시간(분)')
plt.style.use('seaborn-pastel')
plt.show()

"""### 1-3 konlpy 설치
  * pip install konlpy
"""

!pip install konlpy

import nltk
import matplotlib.pyplot as plt
import numpy as np

"""#### 꼬꼬마를 이용한 분석 
  * 문장 분석
  * 명사 분석
  * 형태소 분석
"""

from konlpy.tag import Kkma
k = Kkma()
k.sentences("안녕하세요! 오늘은 한글 분석을 시작합니다.")

"""### 명사 분석"""

k.nouns("안녕하세요! 오늘은 한글 분석을 시작합니다.")

"""### 형태소 분석
 * http://kkma.snu.ac.kr/documents/index.jsp?doc=postag : 한글 형태소 분석기 품사 태그표
 * (예) NNG : 일반 명사, XSV : 동사 파생 접미사, EFN : 평서형 종결 어미
"""

k.pos("안녕하세요! 오늘은 한글 분석을 시작합니다.")

"""### 또 다른 한글 엔진 사용해 보기
 * 한나눔 한글 엔진 사용해 보기
 * Okt 엔진 사용해 보기
"""

from konlpy.tag import Hannanum
hannanum = Hannanum()
hannanum.nouns("안녕하세요! 오늘은 한글어 분석을 시작합니다.")

from konlpy.tag import Okt
okt = Okt()
print(okt.morphs("단독입찰보다 복수 입찰의 경우"))