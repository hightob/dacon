markdown.md
<!-- Heading -->
# Catboost 모델 진행

## 목차
1. 모델 전처리
2. 학습진행
3. 예측

## 1. 모델 전처리 
### 개발환경
- os: Linux-5.4.144+-x86_64-with-Ubuntu-18.04-bionic
- python: 3.7.12 (default, Sep 10 2021, 00:21:48) 
[GCC 7.5.0]
- pandas: 1.1.5
- numpy: 1.19.5
- sklearn: 1.0.1

colab에서 진행했습니다.

### 데이터 경로
```py
DATA_PATH = '/content/drive/MyDrive/dacon/jobcare/Jobcare_data/'
train_data = pd.read_csv(f'{DATA_PATH}train.csv')
test_data = pd.read_csv(f'{DATA_PATH}test.csv')
SEED = 42
code_d = pd.read_csv(f'{DATA_PATH}속성_D_코드.csv').iloc[:,:-1]
code_h = pd.read_csv(f'{DATA_PATH}속성_H_코드.csv')
code_l = pd.read_csv(f'{DATA_PATH}속성_L_코드.csv')

train_data.shape , test_data.shape
```
((501951, 35), (46404, 34))

코드 병합을 위해 col명 통일
```py
code_d.columns= ["attribute_d","attribute_d_d","attribute_d_s","attribute_d_m"]
code_h.columns= ["attribute_h","attribute_h_m","attribute_h_l"]
code_l.columns= ["attribute_l","attribute_l_d","attribute_l_s","attribute_l_m","attribute_l_l"]
```

















___
여기 부터는 실행 방법에 대한 설명
### 실행방법
1. 코드 실행 순서는 Eli5 -> Optuna -> Catboost 이다.
2. Eli5는 피쳐가 모델 학습에 미치는 영향을 위함. 따라서 선정된 피쳐의 지수가 낮은 하위 3개의 피쳐는 학습에서 제외
3. Optuna는 선정된 피쳐를 활용해 학습할때 설정하는 하이퍼 파라미터의 최적값을 찾기 위함 Best_Params의 리스트 값을 저장하고 Catboost모델에 적용
4. Catboost는 모델 학습을 위해 전처리 과정, 모델 선언 과정, 예측 값을 통한 y값 도출 과정이 있다. 이를 각 Readme파일에 맞게 선언하면 된다.


# 새로 작성을 해보자
## 제목 : 최종결과물: Catboost
범주형 변수에 강력한 성능을 보여주는 catboost를 이용하여 코드를 작성해 봤습니다. 
구글 코랩에서 작성하였고, GPU를 이용하여 학습 했습니다.
최종 예측 파일 threshold값을 조정해서 재현율을 끌어올려 LB 점수에서 이득을 봤습니다. 참고가 되었으면 좋겠습니다. 
### 1.Library & Data Load

### 2. Data Preprocess
속성_D_코드.csv,속성_L_코드.csv,속성_H_코드.csv를 학습데이터에 추가하기 위해 데이터 병합을 진행했고 Eli5에서 제외한 학습에 방해가 되는 코드는 학습 피쳐에서 제외했습니다.

### 3. 모델링
Eli5로 catboostclassifier모델을 진행하는데 있어 제외할 컬럼을 미리 선정했습니다. 또한 cat_feature에 범주형 칼럼리스트를 만들어 학습에 용이하게 했습니다.

Optuna로 Random Search를 통해 Catboost 최적의 파라미터를 사용하였습니다. objective 함수의 param에 파라미터를 넣고, 구간을 넣으면 랜덤한 값으로 학습되며 f1-score값이 반환되는 함수입니다. "trial"에 반복 횟수를 작성하면 됩니다.

Catboost 특성상 학습이 오래 걸리기 때문에 최적의 파라미터를 찾아 cat_param로 정의하였습니다.

(아래코드는 국경원 요원님의 잡케어 추천 알고리즘 경진대회 [Private 8위 0.66203] | Catboost 코드를 참고하여 수정하였습니다.)


# 결과값제출
train을 K-fold한 값의 평균을 구하다 보니 예측값의 극단값이 작아질 수 밖에 없습니다.

따라서 threshold_search 함수를 만들어 최대 threshold를 선정해 예측값이 제일 높은 것으로 진행했습니다.
