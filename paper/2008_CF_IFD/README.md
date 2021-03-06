## [2008_CF_IFD] Collaborative Filtering for Implicit Feedback Datasets

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
    * assign all <img src="https://latex.codecogs.com/gif.latex?r_ui"> variables - 0  

---
### 3 Previous work  
#### 3.1 Neighborhood models - common approach   
* approch  
  * user-oriented: 같은 생각을 가진 사용자의 평가를 기반으로 추정  
  * item-oriented: 유사 항목에 대해 동일한 사용자가 만든 알려진 등급으로 추정   
  * => user < item : 확장성, 정확도 향상, 추론 설명 적합  
    * 선호 item 파악보다 like-mind user 특정이 어려움  
    * 암시적 피드백과 관련하여 단점 공유  
    * 사용자 선호도, 신뢰도 구별할 수 있는 유연성 제공 X  

* <img src="https://latex.codecogs.com/gif.latex?%5Chat%20r_%7Bui%7D">: item i에 대해 user u가 관찰하지 않은 값, 인접 항목에 대한 평점의 가중 평균   

* <img src="https://latex.codecogs.com/gif.latex?%5Chat%7Br%7D_%7Bui%7D%20%3D%20%7B%7B%5Csum_%7Bj%5Cin%7BS%5Ek%28u%3Bi%29%7D%7Ds_%7Bij%7Dr_%7Buj%7D%7D%20%5Cover%20%7B%20%5Csum_%7Bj%5Cin%20S%5Ek%28i%3Bu%29%7D%20s_%7Bij%7D%20%7D%7D">  

  * <img src="https://latex.codecogs.com/gif.latex?s_%7Bij%7D">: item i와 item j의 유사성(Pearson 상관계수 기반)  

  * <img src="https://latex.codecogs.com/gif.latex?s%5Ek%28i%3Bu%29">: item i와 가장 유사한 user u가 평가된 k개 item 이웃 집합  

#### 3.2 Latent factor model - alternative approach  
* 목표: 관찰된 등급을 설명하는 Latent factor를 발견하는것  
* pLSA, neural networks, Latent Dirichlet Allocation

* user-item 관찰 matrix의 SVD(Singular Value Decomposition) 유도 모델  
  * 정확성, 확장성  
  * typical 모델: 각각 연관, user u <img src="https://latex.codecogs.com/gif.latex?x_u%5Cin%20%5Cmathbb%7BR%7D%5Ef"> item i <img src="https://latex.codecogs.com/gif.latex?y_i%20%5Cin%20%5Cmathbb%7BR%7D%5Ef">  
    * 예측: 내적 <img src="https://latex.codecogs.com/gif.latex?%5Chat%20r_%7Bui%7D%20%3D%20x_u%5ET%20y_i">, parameter estimation(모수추정)  

* explicit feedback regularized 모델(과적합 피하며 관찰 등급 직접 모델링)    

* <img src="https://latex.codecogs.com/gif.latex?min_%7Bx*%2C%20y*%7D%20%5Csum_%7Br_%7Bu%2Ci%7D%20is%20known%7D%20%28r_%7Bui%7D%20-%20x_u%5ETy_i%29%5E2%20&plus;%20%5Clambda%20%28%5Cleft%20%5C%7C%20x_u%20%5Cright%20%5C%7C%5E2%20&plus;%20%5Cleft%20%5C%7C%20y_i%20%5Cright%20%5C%7C%5E2%29">  

  * 파라미터: SGD 학습  
  * 3.1 Neighborhood model 보다 성능 우수 경향  
  * => implicit feedback 모델로 접근 방식 차용  

---
### 4. Our model

* <img src="https://latex.codecogs.com/gif.latex?p_%7Bui%7D">: <img src="https://latex.codecogs.com/gif.latex?r_%7Bui%7D"> 이진화 파생(이진 변수)  
* ![main](./image/4-1.PNG)  
  * user u가 item i를 소비하면 1(선호한다), 소비하지 않으면 0(선호하지 않는다)  
  * <img src="https://latex.codecogs.com/gif.latex?p_%7Bui%7D"> 의 낮은 신뢰도
    * 선호도 이외에 다른 요인으로 결과  
    * 같은 값(0/1) 이라도 다른 의미(다른 신뢰수준)  
    * => <img src="https://latex.codecogs.com/gif.latex?c_%7Bui%7D">: <img src="https://latex.codecogs.com/gif.latex?p_%7Bui%7D"> 의 신뢰도 측정 변수   
      * <img src="https://latex.codecogs.com/gif.latex?c_%7Bui%7D%20%3D%201%20&plus;%20%5Calpha%20r_%7Bui%7D">  

        * 더 많은 관찰 시 신뢰도 증가(α: 증가율 제어)  

* 목표: user u, item i에 대한 벡터를 찾는것  
  * <img src="https://latex.codecogs.com/gif.latex?x_u%20%5Cin%20%5Cmathbb%7BR%7D%5Ef%2C%20y_i%20%5Cin%20%5Cmathbb%7BR%7D%5Ef">  

    * 사용자 요인, 항목 요인, 직접 비교할 수 있는 공통 잠재 요인 공간에 매핑(<img src="https://latex.codecogs.com/gif.latex?%5Cmathbb%7BR%7D%5Ef">)    
  * 선호도 가정 => 내적  
    * <img src="https://latex.codecogs.com/gif.latex?p_%7Bui%7D%20%3D%20x_u%5ETy_i">  
    * SVD 와 차이점(explicit feedback)  
      * 다양한 신뢰수준 설명해야함  
      * 가능한 모든 (u, i) 쌍 고려 > 관찰 데이터   
  * 비용함수  
    * <img src="https://latex.codecogs.com/gif.latex?min_%7Bx*%2Cy*%7D%5Csum_%7Bu%2Ci%7Dc_%7Bui%7D%28p_%7Bui%7D-x_u%5ET%20y_i%29%5E2%20&plus;%20%5Clambda%28%5Csum_u%20%5Cleft%20%5C%7C%20x_u%20%5Cright%20%5C%7C%5E2%20&plus;%20%5Csum_i%20%5Cleft%20%5C%7C%20y_i%20%5Cright%20%5C%7C%5E2%29">  

      * (min) 신뢰도(이진 - 예측치) + 규제  
      * m: user 수, n: item 수 (수가 많아 SGD같은 최적화 기술 부적합)  
      * => 효율적 최적화 프로세스 필요  

* 비용함수 user-factors or the item-factors 고정  
  * quadratic > global minimum >> 교대 최소 제곱 최적화 '프로세스' 필요  

* 최적의 프로세스 제안  
  * 1) 모든 user factor 재계산  
    * <img src="https://latex.codecogs.com/gif.latex?x_u%20%3D%20%28Y%5ETC%5EuY%20&plus;%20%5Clambda%20I%29%5E%7B-1%7DY%5ET%20C%5Eu%20p%28u%29">    

      * Y (n*f matrix): 모든 item factor  
      * <img src="https://latex.codecogs.com/gif.latex?C%5Eu">: 각 user u, <img src="https://latex.codecogs.com/gif.latex?C_%7Bii%7D%5Eu%20%3D%20c_%7Bui%7D"> 인 대각행렬 n X n  
      * <img src="https://latex.codecogs.com/gif.latex?p%28u%29%20%5Cin%20%5Cmathbb%7BR%7D%5En">: 모든 user 선호도 포함 벡터(<img src="https://latex.codecogs.com/gif.latex?p_%7Bui%7D"> values)  
      * running time이 input 크기에 선형적(linear)  
  * 2) item 함수 재계산(병렬 방식)
    * <img src="https://latex.codecogs.com/gif.latex?y_i%20%3D%20%28X%5ET%20C%5Ei%20X%20&plus;%20%5Clambda%20I%29%5E%7B-1%7D%20X%5ET%20C%5Ei%20p%28i%29">  

      * A typical number of sweeps is 10  
      * whole process는 input 크기에 선형적  
  * 추가적인 대안
    * <img src="https://latex.codecogs.com/gif.latex?c_%7Bui%7D%20%3D%201%20&plus;%20%5Calpha%20log%20%281%20&plus;%20r_%7Bui%7D%20/%20%5Cepsilon%20%29">  

---
### 5. Explaining recommendations
* 좋은 추천 + **간략한 설명**  
  * user 신뢰도, 올바른 관점의 추천, 디버깅, 예기치 못한 동작 원인 추적   
  * 설명 제공을 위한 기술(2)  
    * a) 간단함> neighborhood-based(or, “memory-based”): 과거 user 행동 직접 추론  
    * b) 어려움> latent factor models : 과거 user 행동 직접 추상화(추천과 관계 차단됨)  
      * user 요소 대체 - ALS(alternating least squares model)  
        * <img src="https://latex.codecogs.com/gif.latex?x_u%20%3D%20%28Y%5ETC%5EuY%20&plus;%20%5Clambda%20I%29%5E%7B-1%7DY%5ET%20C%5Eu%20p%28u%29">  
        * item i에 대한 user u의 예측 선호도 <img src="https://latex.codecogs.com/gif.latex?%5Chat%20p_%7Bui%7D%20%3D%20y_i%5ETx_u"> 에서 user 요소 <img src="https://latex.codecogs.com/gif.latex?x_u"> 대체 
        * => <img src="https://latex.codecogs.com/gif.latex?y_i%5ET%28Y%5ETC%5EuY%20&plus;%20%5Clambda%20I%29%5E%7B-1%7D%20Y%5ETC%5Eup%28u%29">   
          * 간략화 <img src="https://latex.codecogs.com/gif.latex?W%5Eu%20%3D%20%28Y%5ETC%5EuY%20&plus;%20%5Clambda%20I%29%5E-1"> (가중치 f*f 행렬)  
        * user u의 관점에서 item i, j 사이의 가중치 유사성: <img src="https://latex.codecogs.com/gif.latex?s_%7Bij%7D%5Eu%20%3D%20y_i%5ETW%5Euy_j">  

* item i에 대한 user u의 예상 선호도 공식 업데이트  
  * <img src="https://latex.codecogs.com/gif.latex?%5Chat%20p%20_%7Bui%7D%20%3D%20%5Csum_%7Bj%3Ar_%7Buj%7D%20%3E%200%7Ds_%7Bij%7D%5Eu%20c_%7Buj%7D">   

    * 잠재 요인 모델을 선형 모델로 축소  

---  
### 6. Experimental study

#### Data description  
#### Evaluation methodology  
#### Evaluation results  

---
### 7. Discussion
* implicit feedback - collaborative filtering  
* 결과:  
  * user observations > 2 pair로 변환  
    * each user-item pair> “preference” infer> 신뢰수준과 결합  
* latent factor algorithm  
  * observations 긍정적 선호도 편향> user 프로필 반영 ↓> 모든 user-item 선호도 input 사용  
    * 확장성 문제 발생> 대수(algebraic) 구조 활용 해결  
  * 설명 가능    
    * item-oriented neighborhood approach>  insightful link  
* 동적 시간 변수 추가  
* standard training and test setup, 사용자 행동 예측 평가 설계  