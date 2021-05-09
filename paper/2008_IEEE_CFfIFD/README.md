## [2008_IEEE_CFfIFD] Collaborative_Filtering_for_Implicit_Feedback_Datasets

![main](./image/main.PNG)

---

### Abstract  
* 추천시스템: prior implicit feedback > personalized > improve UX  
  - 고객 dislike 정보 부족  
* 제안:  
  - 다양한 신뢰수준, 긍/부정 선호도 > 암시적 피드백 추천자 요인 모델   
  - 데이터 크기에 따라 선형 확장 가능한 최적화 절차  

---
### 1. Introduction
* 요구: 개인화된 사용자 취향/요구에 맞는 제품/서비스 추천  
* 기술: 사용자-제품 프로파일링, 관계 찾는것  
* 전략(2)/+조합
  - 컨텐스 기반 접근 방식: 사용자/제품 프로파일(구하기 어려운 explicit feedback 요구)  
  - 협업 필터링: 사용자 간의 관계와 제품 ​​간의 상호 종속성 분석(사용자의 과거 행동-explicit feedback 요구 X)   
      - 결과: 새로운 사용자-아이팀 연관성 식별  
      - 장점: 도메인 free, 컨텐츠 기반 기술이 처리하기 어려운 데이터 측면 처리 가능, 일반적으로 컨텐츠 기반 기술보다 정확  
      - 단점: cold start 
* Input(2)  
  - explicit: high quality, convenience, not always available  
    - 한계: 평가 꺼리거나 수집할 수 없는 상황  
  - implicit: 사용자 행동 추론, 간접 의견  
    - 극복: 사용자가 수집에 승인하면 추가적 피드백 불필요   
* Implicit feedback 특성  
  - 1) No negative feedback: dislike 추론 어려움(결측치와 구분), 근본적인 비대칭  
    - 부정 암시적 피드백 존재할 결측 데이터 처리 필요  
  - 2) inherently noisy  
    -  수동적 추적, 사용자 선호도/동기만 추측(물건을 샀다고 만족을 보장하는건 아님)  
  - 3) 명시적 피드백 수치 = 선호도 | 암시적 피드백 수치 = 신뢰도   
    - 명시적(1~5점), 암시적(행동 빈도-일회성 작업, 반복=> 사용자 의견 반영)(값-신뢰도 비례 보장 X)  
  - 4) 평가를 위한 적절한 조치 필요  
    - 가용성, 다른 item과 경쟁/반복 고려  

---
### 2. Preliminaries  
user: u, v  
item: i, j  
* <img src="https://latex.codecogs.com/gif.latex?r_%7Bui%7D">: observation, user-item association  
  * explicit feedback: item에 대한 user의 선호도 등급  
    * unknown user-item pair - ignore  
  * implicit feedback: user action에 대한 관찰  
    * assign all r_ui variables - 0  

---
### 3 Previous work  
##### 3.1 Neighborhood models - common approach   
* approch  
  * user-oriented: 같은 생각을 가진 사용자의 평가를 기반으로 추정  
  * item-oriented: 유사 항목에 대해 동일한 사용자가 만든 알려진 등급으초 추정   
  * => user < item : 확장성, 정확도 향상, 추론 설명 적합  
    * 선호 item 파악보다 like-mind user 특정이 어려움  
    * 암시적 피드백과 관련하여 단점 공유  
    * 사용자 선호도, 신뢰도 구별할 수 있는 유연성 제공 X  

* <img src="https://latex.codecogs.com/gif.latex?%5Chat%20r_%7Bui%7D">: item i에 대해 user u가 관찰하지 않은 값, 인접 항목에 대한 평점의 가중 평균   
<img src="https://latex.codecogs.com/gif.latex?%5Chat%7Br%7D_%7Bui%7D%20%3D%20%7B%7B%5Csum_%7Bj%5Cin%7BS%5Ek%28u%3Bi%29%7D%7Ds_%7Bij%7Dr_%7Buj%7D%7D%20%5Cover%20%7B%20%5Csum_%7Bj%5Cin%20S%5Ek%28i%3Bu%29%7D%20s_%7Bij%7D%20%7D%7D">
<img src="https://latex.codecogs.com/gif.latex?s_%7Bij%7D">: item i와 item j의 유사성(Pearson 상관계수 기반)  
<img src="https://latex.codecogs.com/gif.latex?s%5Ek%28i%3Bu%29">: item i와 가장 유사한 user u가 평가된 k개 item 이웃 집합  

##### 3.2 Latent factor model - alternative approach  
* 목표: 관찰된 등급을 설명하는 Latent factor를 발견하는것  
* pLSA, neural networks, Latent Dirichlet Allocation

* user-item 관찰 matrix의 SVD(Singular Value Decomposition) 유도 모델  
  * 정확성, 확장성  
  * typical 모델: 각각 연관, user u <img src="https://latex.codecogs.com/gif.latex?x_u%5Cin%20%5Cmathbb%7BR%7D%5Ef"> item i <img src="https://latex.codecogs.com/gif.latex?y_i%20%5Cin%20%5Cmathbb%7BR%7D%5Ef">  
    * 예측: 내적 <img src="https://latex.codecogs.com/gif.latex?%5Chat%20r_%7Bui%7D%20%3D%20x_u%5ET%20y_i">, parameter estimation(모수추정)  

* explicit feedback regularized 모델(과적합 피하며 관찰 등급 직접 모델링)    
<img src="https://latex.codecogs.com/gif.latex?min_%7Bx*%2C%20y*%7D%20%5Csum_%7Br_%7Bu%2Ci%7D%20is%20known%7D%20%28r_%7Bui%7D%20-%20x_u%5ETy_i%29%5E2%20&plus;%20%5Clambda%20%28%5Cleft%20%5C%7C%20x_u%20%5Cright%20%5C%7C%5E2%20&plus;%20%5Cleft%20%5C%7C%20y_i%20%5Cright%20%5C%7C%5E2%29">  
  * 파라미터: SGD 학습
  * 3.1 Neighborhood model 보다 성능 우수 경향  
  * => implicit feedback 모델로 접근 방식 차용  

---
### 4. Our model

---
### 5. Explaining recommendations

---  
### 6. Experimental study

---
### 7. Discussion