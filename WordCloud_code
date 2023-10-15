## 데이터 청년 캠퍼스 kusf 협업(훈련용품 최적화) 시 사용한 word cloud 시각화 code ##
## 중구분에 대해서 세부구분,훈련용품 명들에 대해서 word cloud ##


!pip install konlpy wordcloud

import nltk
from konlpy.corpus import kobill
from wordcloud import WordCloud
from matplotlib import gridspec

# 텍스트 전처리
def text_preprocessor(s):

    # (1) 괄호와 괄호 안 문자 제거
    pattern = r'\([^)]*\)'  # ()
    s = re.sub(pattern=pattern, repl='', string=s)

    # (2) 특수문자 제거
    pattern = r'[^가-힣]'
    s = re.sub(pattern=pattern, repl='', string=s)

    # (3) 상의, 하의 제거
    s = s.replace('상의', '')
    s = s.replace('하의', '')
    s = s.strip()

    # (4) 볼 -> 공
    s = s.replace('볼', '공')

    return s

# 전처리한 리스트 반환 -> 세부 구분에 대한 정리
def product_sub_name(df):
  product_list = df['product_name'].to_list()
  for i in range(len(product_list)):
    product_list[i] = text_preprocessor(product_list[i])
  return product_list


product_list_sub_name = product_sub_name(eda_dt)
    
# 기존 데이터 훈련용품명 변경
eda_dt['product_name'] = product_list_sub_name
eda_dt.head()

# 빈도 그래프, 워드클라우드 시각화
def freq_plot(data, file_name):
  freq = [i for i in data if i] # 빈도수 데이터

  ko = nltk.Text(freq, name='test')

  freq_list = ko.vocab().most_common(100) # 빈도수 많은 상위 100개 키워드 추출

  wc = WordCloud(font_path=path,width=800, height=800,
                scale=2.0, max_font_size=200, max_words=100,
                background_color='white') # 워드 클라우드 설정
  wc.generate_from_frequencies(dict(freq_list)) # 빈도수 기반 워드 클라우드

  plt.show()

  # 빈도수 선 그래프
  plt.figure(figsize=(10, 5))
  fig = ko.plot(50).figure
  plt.show()

  # 워드 클라우드
  plt.figure(figsize=(5, 5))
  plt.imshow(wc)
  plt.axis('off')
  plt.show()
