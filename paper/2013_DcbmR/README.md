## 2013_DcbmR [Deep content-based music recommendation]

![main](./image/main.PNG)

---

### Abstract    
* 자동 음악 추천  

* 협업 필터링(CF) 문제: cold-start  

* 제안: 음악 오디오의 잠재 요소 모델  
  * 심층 CNN  

---

### 1. Introduction
* 음악 추천 고려 요인: 다양한 스타일, 장르, 청취자 선호도(사회적/지리적 요인)  
  * 항목 수(음악) 多  

* 접근 방식: 
  * 협업 필터링(CF): 사용 패턴(power law) 의존  
    * cold-start 문제, 틈새 고객(niche audience) 데이터 부족으로 추천 난이도 ↑  
  * 콘텐츠 기반: 항목 콘텐츠 / 메타 데이터 > 사용자 선호도    
* 일반적 성능: 협업 필터링 > 콘텐츠 기반    

#### 1.1 Content-based music recommendation  
* Content-base  
  * 음악 메타 데이터(아티스트, 앨범 및 출시 연도) 이용    
    * 예측 가능한 권장 사항: 사용자 선호 아티스트 노래 추천 => 특별히 유용 X  
  * 오디오 신호 간 유사성 측정  
    * 유사성 메트릭 정의 필요(사전 지식 기반 임시 정의) => 반드시 최적 X  

#### 1.2 Collaborative filtering
* CF(2):    
  * 이웃 기반: 사용자 / 항목 간 유사성 측정 의존  
  * 모델 기반: 사용자 / 항목 잠재 특성 모델링  

#### 1.3 The semantic gap in music
* 잠재 인자 벡터: 다양한 사용자 취향 / 항목 특성  
  * ![T1](./image/T1.PNG)  
    * 데이터세트(↓) > 잠재 요인 계산 > 긍/부정값 가지는 아티스트    
      * 많은 노래가 데이터 부족으로 요인 벡터 안정적 추정 불가능  
      * ∴ 음악 오디오 콘텐츠 예측 가능 => 유용    
    
* 노래 특성(사용자 선호도 영향):   
  * 오디오 신호   
    * 추출 ○: (모델) 복잡한 계층 구조 캡처 > 높은 수준 속성(장르, 분위기, 악기, 테마)   
    * 추출 X: 속성(아티스트 인기, 명성, 위치)  

* 음악 정보 검색(MIR: music information retrieval) 분야 > 고수준 속성 추출 필요     

* MFCC(mel-frequency cepstral coefficients): 오디오 신호 feature  

---

### 2. The dataset 
* dataset: Million Song Dataset(MSD), Last.fm, 7digital.com, Taste Profile Subset    

---

### 3. Weighted matrix factorization

* WMF(Weighted Matrix Factorization): 모든 사용자 / 항목 잠재 요인 표현 학습  
  * 암시적 피드백 데이터세트; modified matrix factorization  

* <img src="https://latex.codecogs.com/gif.latex?p_%7Bui%7D%2C%20c_%7Bui%7D">: 선호도 변수(1이면 선호), 신뢰도 변수(선호도 확신 정도)   
  * ![(1+2)](./image/(1+2).PNG)  
    * <img src="https://latex.codecogs.com/gif.latex?r_%7Bui%7D">: 사용자 u / 노래 i - 재생 횟수(횟수 ∝ 선호도; 재생횟수=0 > 신뢰도 ↓)  
    * I(x): 표시기 함수  
    * <img src="https://latex.codecogs.com/gif.latex?%5Calpha%2C%20%5Cepsilon">: 하이퍼파라미터  

* WMF 목적 함수  
  * ![(3)](./image/(3).PNG)  
    * λ: regularization parameter  
    * : 사용자 u에 대한 latent factor vector 
    * <img src="https://latex.codecogs.com/gif.latex?y_i">: 노래 i에 대한 latent factor vector  
    * confidence-weighted mean squared error: L2 regularization  
* 데이터세트 크기 고려; SGD 보다 ALS 최적화 제안  

---

### 4. Predicting latent factors from music audio 
* (오디오 신호) 노래 잠재 요인 예측 => **회귀 문제**  
  * 함수(시계열> [map] 실수 벡터) 학습  

* 평가(2):  
  * 1. MIR 기존 접근 방식: 오디오 신호에서 로컬 특징 추출 > BoW(bag-of-words) 표현으로 집계 > 기존 회귀 기법(map)  
  * 2. CNN  

* 데이터 > [WMF] > 잠재 인자 벡터(훈련 ground truth)  
  * 효율적 최적화 절차 => WFM 사용    

#### 4.1 Bag-of-words representation
* MIR 시스템 feature 추출 파이프라인: 음악 오디오 신호 > [pipeline] > 고정 크기 표현 > classifier/regressor   
  * 오디오 신호 > MFCC 추출  
  * 벡터: MFCC 양자화(quantize)  
  * bag-of-words representatio 집계  
  * PCA(95%) > 선형회귀 / MLP > MLR > 콘텐츠 기반 추천 유사성 메트릭 학습   

#### 4.2 Convolutional neural networks
* CNN 성공요소(3):  
  * ReLU: 수렴 속도 ↑; 사라지는 경사 문제 ↓  
  * 병렬화: 훈련 속도 ↑;  
  * 데이터↑: 대형 모델(파라미터↑) 적합  
* dropout regularization: 큰 개선 X  
* 입력: 오디오 신호 > 중간 시간-주파수 표현(intermediate time-frequency representation) 추출 > NW  
* NW: 3s windows; average   
* 음악 오디오 CNN 잠재 인자 예측 이점:   
  * 다른 factors간 intermediate features 공유  
  * 계층구조(교대: feature 추출 레이어-풀링 레이어) > multiple timescales 작동    

#### 4.3 Objective functions 
* 목표:  
  * 잠재 인자 벡터(실수 값) > 평균 제곱 오차 (MSE)를 최소화 or  
  * WMF 목적 함수 > WPE(weighted prediction error; 가중 예측 오차) 최소화  
    * 목적 함수  
      * ![(5)](./image/(5).PNG)  
        * θ: model parameters  
        * <img src="https://latex.codecogs.com/gif.latex?y_i">: WMF > 노래 i에 대한 잠재 인자 벡터  
        * <img src="https://latex.codecogs.com/gif.latex?%7By%7D%27_i">: 모델에 의한 해당 예측  

---

### 5. Experiments
#### 5.1 Versatility of the latent factor representation
* 태그 예측 v.s 오디오 features  
  * 태그: 노래의 다양한 측면(장르, 악기, 템포, 분위기 및 출시 연도)   

#### 5.2 Latent factor prediction: quantitative evaluation
* 잠재 인자 벡터 예측 모델   
  * bag-of-words 훈련 선형 회귀  
  * 동일 bag-of-words 훈련 MLP  
  * CNN: log-scaled mel-spectrograms 학습; min MSE  
  * 동일 CNN; WMF 목적함수; min WPE(weighted prediction error)   
    * ![T2](./image/T2.PNG)
      * 평가 측도: AUC, mAP(mean average precision): 사용자마다 500개 자름  
      * latent factors > metric learning approach  
      * 한계: bag-of-words feature 표현(시간적 구조 반영 X )  
      * CNN > 성능 ↑  
      * WPE 목표: 성능 향상 X; 노래 중요성(가중치) ∝ 인기  

* 콘텐츠 예측 성능 상한선  
  * ![T3](./image/T3.PNG)  
    * 사용자 선호도: 오디오 신호에서만 추출 X  

#### 5.3 Latent factor prediction: qualitative evaluation
* 통찰력: 정확도 메트릭만으로 부족  
* 사용 패턴 간의 코사인 유사성 측정 > 유사한 노래 검색  
  * ![T4](./image/T4.PNG)
    * 곡과 가장 가까운 곡   
    * 예측 잠재 요인 활용 > 동일 청중 어필 가능성 ↑ / 다양성 ↑      
      * ![Fig1](./image/Fig1.PNG)
        *  t-SNE; 예측된 사용 패턴 분포  
---