## 2017_JT_RRN [JOINT TRAINING OF RATINGS AND REVIEWS WITH RECURRENT RECOMMENDER NETWORKS]

![main](./image/main.PNG)

---

### ABSTRACT
* ratings + reviews + temporal patterns => NN 모델  
  * recurrent architecture: user/itme dynamic components states   
  * numerical ratings + NL(natural language) reviews  

---

### 1. INTRODUCTION
* 목표: 이전 데이터 & 리뷰 DB > 특정 사용자 항목 선호도 예측  

* 이전 연구(bag-of-words 모델)의 한계(2):  
  * 1) 예측 정확도: 미래 등급 예측으로 거의 측정 X  
    * 무작위 분할된 데이터에서 평가 > 모델 유용성 이해 훼손  
    * 리뷰 시간적 패턴 탐구 ↓  
  * 2) 추천 시스템 리뷰 모델 기술(리뷰 이해 ↓) < 자연어 처리 최신 기술  
    * 리뷰: 일반 문서보다 다양, 구조화 X  
* LSTM, RNN 결합 제안  
  * Joint generative model: 상호작용 반복 네트워크(특히 LSTM) > 평가/리뷰 공동 모델 제안  
  * Nonlinear nonparametric review model: 사용자, 영화 state dynamics 함수 학습, 시간 경과에 따른 리뷰 진화 캡처    

---

### 2. MODEL
* Joint Review-Rating Recurrent Recommender Network
  * ![Fig1](./image/Fig1.PNG)  
    * 입력: user/movie history  
    * 알고리즘: LSTM/RNNs  
    * user/movie states temporal dynamics 캡처  

<br>

  * generator functions  
  * ![2-1](./image/2-1.PNG)
    * (input) i: user / j: movie의 고정/동적 상태(stationary and dynamic states)  
    * (output) time step(t),   
      * <img src="https://latex.codecogs.com/gif.latex?r_%7Bij%7D%7Ct">: 등급  
      * <img src="https://latex.codecogs.com/gif.latex?i_%7Bij%7D%7Ct">: 리뷰  
    * <img src="https://latex.codecogs.com/gif.latex?u_i%2C%20m_j">: 고정 상태  
    * <img src="https://latex.codecogs.com/gif.latex?u_%7Bit%7D%2C%20m_%7Bit%7D">: t에서 동적 상태   
    * states 직접 학습 X -> states 찾는 *함수* 학습   

#### Dynamic User and Movie State    
* 입력: user/movie rating history > states update   
* user-state RNN 각 단계, 네트워크:   
  * <img src="https://latex.codecogs.com/gif.latex?y_t%20%3A%3D%20W_%7Bembed%7D%5Bx_t%2C%201_%7Bnewbie%7D%2C%20%5Ctau_t%2C%20%5Ctau_%7Bt-1%7D%5D">    

    * <img src="https://latex.codecogs.com/gif.latex?x_t">: rating vector    
      * j-th 요소: 시간 t, 영화 j 사용자 등급; 없으면 0    
    * <img src="https://latex.codecogs.com/gif.latex?1_%7Bnewbie%7D">: indicator for new users    
    * <img src="https://latex.codecogs.com/gif.latex?%5Ctau_t">: wall-clock time    
* state update    
  * 표준 <img src="https://latex.codecogs.com/gif.latex?u_t%20%3A%3D%20LSTM%28u_%7Bt-1%7D%2C%20y_t%29">  
    * user index 생략  

#### Rating Emissions 
* 프로파일 벡터  
  * 시간에 따라 변화($ u_{it}, m_{jt} $) > [보완] > 고정(<img src="https://latex.codecogs.com/gif.latex?u_i%2C%20m_j">)<encode> > 시간 불변 속성(사용자 장기적 선호, 영화 장르)  
    * ∴ review rating: dynamic and stationary states 함수로 모델링  
    * ![(1)](./image/(1).PNG)  
      * <img src="https://latex.codecogs.com/gif.latex?%5Ctilde%20u_%7Bit%7D%2C%20%5Ctilde%20m_%7Bjt%7D">: <img src="https://latex.codecogs.com/gif.latex?u_%7Bit%7D%2C%20m_%7Bjt%7D">의 affine functions  
        * ![2-2](./image/2-2.PNG)  


#### Review Text Model  
* Review text: character-level LSTM network  
  * 동일한 user/movie latent states, rating model 공유  
* <img src="https://latex.codecogs.com/gif.latex?x_%7Bjoint%2C%20i%2Cj%7D">: 병목 레이어; 영화/사용자 동적 상태 + 고정 상태  
  * ![2-3](./image/2-3.PNG)  
    * <img src="https://latex.codecogs.com/gif.latex?o_%7Bij%2C%20k%7D">: 사용자 i, 영화 j 의 리뷰 k 위치 문자  
    * <img src="https://latex.codecogs.com/gif.latex?x_%7Biij%2C%20k%7D">: 문자 embedding    
    * φ: 비선형 함수  

* review text emission model ≒    
  * RNN, character-level LSTM generative model   
  * ![2-4](./image/2-4.PNG)  
    * softmax: LSTM 출력; 다음 문자 예측  

#### Training & Prediction 
* 목표: ratings, reviews 예측; 
  * (min) ![2-5](./image/2-5.PNG)     
    * <img src="https://latex.codecogs.com/gif.latex?D_%7Btrain%7D">: (i, j) 쌍 학습셋  
    * θ: 모든 모델 파라미터  
    * <img src="https://latex.codecogs.com/gif.latex?n_%7Bij%7D">: 사용자(i)가 영화(j)에 제공하는 리뷰 문자 수  
    * λ: 가중치 제어  
* [prediction time] 예측된 미래 상태 기반 등급 예측(≠ 기존 접근법)   
  * 최신 등급(입력) > 상태 업데이트 > 등급 예측    
  
---

### 3. EXPERIMENTS
* dataset: IMDb   
* model:   
  * PMF(Mnih & Salakhutdinov, 2007)  
  * Time-SVD++(Koren, 2010); the state-of-the-art temporal model  
  * AutoRec(Sedhain et al., 2015); state-of-the-art neural network-based model  

#### Rating prediction 
* ![T1](./image/T1.PNG)    
  * joint-modeling ratings and reviews > rating-only RRN  
    * 평점과 리뷰간에 통계적 강도 공유; 리뷰> 잠재요인 추정 ↑  

#### Text modeling 
* 난이도가 향상됨; 

---