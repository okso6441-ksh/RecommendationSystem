## 2015_IRSS [Image-based Recommendations on Styles and Substitutes]

![main](./image/main.PNG)

---

### Abstract
* 선택 요소: **appearance**(objects 간 관계, 상호작용)       
* object pairs: 대안(청바지 A-청바지B), 보완(청바지-어울리는 셔츠)   
* 접근: 가능한 가장 큰 데이터 세트 캡처 > 내부 시각적 관계 > 확장 가능한 방법 개발
  * 시각적 관계에 대한 인간의 개념을 발견   

---

### 1. Introduction
* 한 쌍의 객체 간 시각적 관계에 대한 인간 개념(human notion) 모델링  
* (기존) visual style of places objects[individual appearances] > [influence] visual attributes of another  

* 개체간 상위 수준 관계 모델 포함(시각적 유사성 계산 포함 X)  
  * ![Fig1](./image/Fig1.PNG)  
    * 상업적 응용: 사용자가 이미 관심을 보인 다른 항목을 기반 항목 추천    
      * 메타 데이터, 리뷰, 이전 구매 패턴 분석 구축  
        * 문제: cold-start problem, 인간의 시각적 선호도를 모델링     
             
* 인간 개념을 모델링   
  * 한 쌍의 객체: 본질적 연결 X, 인간의 개념만 존재(다른 쌍보다 적합)  
  * 접근방식: 수작업 레이블 이미지 활용      
    * 레이블 작업: 대부분 적은 데이터세트 > 과적합 피하기 위한 작업 필요   
      * 작업 노력 ↑ => 수동 주석 작업 X 제안   
        * 제안: 조금만 관련 있어도(loosely related) 더 큰 데이터 세트에서 작동하는 소스를 찾는것    

#### 1.1 A visual dataset of styles and substitutes
* Amazon 웹 스토어 기반 데이터 세트 개발  
  * Styles and Substitutes 데이터 셋  
    * ![T1](./image/T1.PNG)  
      * 관계: 호환성(‘compatibility’) 개념(2): 대체/보완  
      * 관계 카테고리(4)  
        * 1) X봄 > Y봄 (대체 가능하거나 noise)    
        * 2) X봄 > Y삼 (대체 가능하거나 noise)     
        * 3) X삼 > Y삼 (보완) 
        * 4) X+Y 동시에 삼 (보완)   
      * => 코사인 유사성에 따른 순위로 수집 가능, 외관이 아닌 기능 위주 공동 구매       

#### 1.2 Related work

#### 1.3 A visual and relational recommender system
* 프로세스: 시각적 / 관계형 추천 시스템  
  * 일반 추천(메타데이터, 리뷰) > object 외관 기반    

---

### 2. The Model
* notation  
  * ![T2](./image/T2.PNG)   

* 객체 > 다른 객체 시각적 appearance 대한 선호 표현 방법   
  * 데이터 양에 따라 확장되는 모델  
  * <img src="https://latex.codecogs.com/gif.latex?x%20%5Cin%20%5Cmathbb%20R%5EF">: F차원 feature 벡터  
  * <img src="https://latex.codecogs.com/gif.latex?r_%7Bij%7D%20%5Cin%20R">: objects i/j 관계 세트(관계 카테고리 중 하나에 속함)  
* 목표: 거리변환 파라미터(<img src="https://latex.codecogs.com/gif.latex?d%28x_i%2C%20x_j%29">) 학습  
  * d(·,·): <img src="https://latex.codecogs.com/gif.latex?P%20%28r_%7Bij%20%5Cin%20R%7D%29%2C%20-d%28x_i%2C%20x_j%29"> 단조 증가 하기 위해 찾음  


##### Distances and probabilities
* 거리와 확률을 연관시키기 위해 shifted sigmoid 함수 사용   
  * ![(1)](./image/(1).PNG)  
  * ![Fig2](./image/Fig2.PNG)  
    * cast logistic regression    
* item i/j 거리: (c; 예측 정확도 최대화 위해 미정)  
  * <img src="https://latex.codecogs.com/gif.latex?d%28x_i%2C%20x_j%29"> = c: 확률 0.5     
  * <img src="https://latex.codecogs.com/gif.latex?d%28x_i%2C%20x_j%29"> > c: 0.5 ↑      
  * <img src="https://latex.codecogs.com/gif.latex?d%28x_i%2C%20x_j%29"> > c: 0.5 ↓      

* 잠재적(potential) 거리 함수 세트  
  * Weighted nearest neighbor: 특정 관계와 관련된 특정 차원 학습  
    * ![(2)](./image/(2).PNG)  
  * Mahalanobis transform: 객체간 시각적 유사성 모델링 > feature 차원마다 강조 다름 제한   
    * Mahalanobis 거리 > 이미지 특징을 연결 > 서로 다른 특성 차원 관련(호환)  
    * ![(3)](./image/(3).PNG)  
      * M: full rank p.s.d. matrix(positive symmetric definite): 과적합 위험, 실용성 ↓ >   
      * <img src="https://latex.codecogs.com/gif.latex?M%20%5Csimeq%20YY%5ET"> 근사(Y: 차원 F x K 행렬):   
        * ![(4)](./image/(4).PNG)  

#### 2.1 Style space
* ![(4)](./image/(4).PNG)  
  * features <img src="https://latex.codecogs.com/gif.latex?x_i%2C%20x_j"> 저차원 임베딩 생성(Style space)  
  * K차원 벡터 <img src="https://latex.codecogs.com/gif.latex?s_i%20%3D%20x_iY">(Y: 시각적 유사 X, 관련 객체가 가깝게 식별)      
    * ![(5)](./image/(5).PNG)   

#### 2.2 Personalizing styles to individual users  
* 모델 + 각 개별 사용자가 중요하다고 생각하는 스타일의 차원 학습 => 개념 개인화   

* 개인화 된 거리 함수: 사용자 u - 항목 i/j 거리 측정  
  * ![(6)](./image/(6).PNG)
    * <img src="https://latex.codecogs.com/gif.latex?D%5E%7B%28u%29%7D%3A%20K%20%5Ctimes%20K">: diagonal(positive semidefinite) matrix  
    * <img src="https://latex.codecogs.com/gif.latex?D_%7Bkk%7D%5E%7B%28u%29%7D">: 사용자u가 k번째 스타일 차원에 대해 '관심'있는 정도   
    * <img src="https://latex.codecogs.com/gif.latex?D_%7Bkk%7D%5E%7B%28u%29%7D%20%3D%20X_%7Buk%7D">: 되도록 U X K 행렬 X 만듦  

* 개인화 거리(eq.5 > 단순화  > eq.7)  
    * ![(5)](./image/(5).PNG)  
    * ![(7)](./image/(7).PNG)  
      * <img src="https://latex.codecogs.com/gif.latex?X_u">: 스타일 공간 차원에 투영된 개인화 가중치  

* 개인화 된 공식(eq.6, eq.7)  
  * 의미 있는 경우: 데이터 세트 각 엣지와 관련된 사용자가 있는 경우  
  * 의미 없는 경우: 4가지 그래프 유형  

<br>

* 트리플 (i,j,u) 샘플링  
  * ![T1](./image/T1.PNG)  
  * 사용자 u가 구매한 제품 i,j(u가 i,j 모두 리뷰)   

#### 2.3 Features
* F/W: Caffe > 원본이미지 계산 > Features     

---

### 3. Training
* 각 관계 존재/부재 확률 최대화  
  * 무작위 negative set <img src="https://latex.codecogs.com/gif.latex?Q%20%3D%20%7Br_%7Bij%7D%20%7Cr_%7Bij%7D%20%5Cnotin%20R%7D">, |Q| = |R| log likelihood 최적화  
    * ![(8)](./image/(8).PNG)  
      * gradient ascent; Y와 c에 대해 l(Y,c|R, Q) 최적화 학습  
      * L-BFGS: *quasi-Newton method*; 변수 多; 비선형 최적화  
        * quasi-Newton method: 비선형 최적화, 각 반복에서 목적 함수에 대한 기울기만 필요   
          (이차 미분 필요한 Newton method보다 계산 부담 적음)  
      * Likelihood(eq.8), 미분 계산; 모든 쌍 <img src="https://latex.codecogs.com/gif.latex?r_%7Bij%7D%20%5Cin%20R%20%5Ccup%20Q">; na¨ıvely parallelized  

---

### 4. Experiments
* 비교  
  * WNN(Weighted Nearest Neighbor) 분류  
  * 카테고리 트리(CT): 레이블을 붙인 방법  
  * '관련됨' 표시 기준: 훈련 데이터; 범주 간 동시성 행렬 계산; b 범주가 a 범주의 가장 일반적 연결 상위 50% 중 하나에 속함  

* WNN(Weighted Nearest Neighbor) baseline; 각 제품의 리뷰에 대한 주제 모델을 훈련  
  * ![(9)](./image/(9).PNG)
    * <img src="https://latex.codecogs.com/gif.latex?%5Ctheta_i%2C%20%5Ctheta_j">: 제품 i,j 리뷰 파생 topic 벡터  
    * 단순 이미지 feature 보다 주제 벡터 사용하도록 조정  
    * 경쟁력 ↓  
      * 이유(2)  
        * 규모: 1M 주제 모델 효과적 학습 어려움  
        * 대부분 제품 리뷰 거의 없음  
          * 제품 당 리뷰 수(*power-law*)  
            * power-law(멱 법칙): 한 수가 다른 수 거듭제곱으로 표현되는 두 수 함수적 관계  
            * ![4](./image/4.PNG)  

* 제품 간 관계 예측을 위한 조건  
  * 두 제품 모두 신뢰할 수 있는 feature representations 존재  
    * = 두 제품 모두 여러 리뷰 존재   
* 리뷰 거의 없는 제품으로 야기되는 문제 => cold-start  
  * 리뷰 없는 new 제품 => 이미지 사용 가능 => visual features 기반 예측 변수 구축 주장   

#### 4.1 Experimental protocol
* 프로토콜 구성  
  * 1. 각 카테고리/그래프 유형 - 단일 실험 
  * 2. 목적: 관계 - 비 관계; (*link prediction*) 구별; 확률 > 0.5 식별   
    * link prediction: 네트워크 이론; 두 개체 간 링크 존재 예측  
  * 3. 모든 긍정 관계 & 비 관계 의 무작위 표본 고려(∴임의 분류기 성능= 50%)  
  * 4. 모든 결과 테스트 세트 보고    
  * 5. 결과 해석, 학습 된 모델 대상 이미지만 참조  

* 성능:  
  * 제안된 방법 > 범주 기반 방법 / WNN   
  * K 증가에도 균일하게 향상  
  * 대체/보완 거의 동일  

* 인간의 시각적 선호도와 관련된 근거 X  

#### 4.2 Personalized recommendations
* 사용자(구매 횟수 20회 이상), 각 사용자 50개 공동/비공동 구매 무작위 샘플링  
  * ![T3](./image/T3.PNG)  
    * 공동구매 예측 시 개선됨  

---

### 5. Visualizing Style Space
* 이미지 > <img src="https://latex.codecogs.com/gif.latex?s_i%20%3D%20x_iY"> [변환] > ‘style-space’ 투영  
  * 쌍별 거리 기반; 임베딩이 동형(isomorphism) 下 불변  
* <img src="https://latex.codecogs.com/gif.latex?s_i%2C%20s_j">: 회전/평행이동/반사 => 거리(eq.5) 유지  
  * ![(5)](./image/(5).PNG)    
* 임베딩 효과 시각화  
  * K-차원 임베디드 좌표; k-means clustering  
  * ![Fig3](./image/Fig3.PNG)  
    * 여러 범주(색상, 모양, 항목, 등 더 미묘한 특성)를 기반으로 클러스터(사용자 집합의 선호)  
* 카테고리 - 책  
  * 성능이 뛰어나지 X, 공동구매 예측 정확도가 높음  
  * ![Fig5](./image/Fig5.PNG)  
    * 데이터 파생 스타일 공간 클러스터 시각화  
    * 책 표지에서 정보 확인 가능(아동도서, 장르 등)  
* 모델로 관련 항목 사이 탐색 방법  
  * ![Fig6](./image/Fig6.PNG)  
    * 공동 탐색 가능성 없는 두 항목 무작위 선택 > 학습된 거리 기반 경로 탐색  
* source와 target items 사이 시각적 부드러운 전환 식별  
  * ![Fig7](./image/Fig7.PNG)  
    * co-browsing; K=2  

---

### 6. Generating Recommendations

* 목표: 쿼리항목 > 보완 항목 추천   
  * query item ex: (사용자) 현재 탐색 중 or 방금 구매 제품   
  * 시각적 스타일 기반 연결 가능성 높은 범주 항목 추천 > 선택  
* 각 범주(C)에 대한 recommendations 생성  
  * (given) 쿼리 항목(<img src="https://latex.codecogs.com/gif.latex?x_q">)
  * ![(10)](./image/(10).PNG)     
  * 원하는 범주 속하는 objects간 측정  
    * (min 거리) ![(4)](./image/(4).PNG)   
* ![Fig8](./image/Fig8.PNG)

---

### 7. Outfits in The Wild
* 데이터세트와 독립적 학습된 모델의 외부(wild) 검증 테스트  

---

### 8. Conclusion
* 시각적 관련된 인간의 개념 모델링 가능  
  * 단순한 시각적 유사성(대체) > 시각적 관계 모델링(보완)  
    * 유사성 이상의 측면에서 다른 객체의 외관에 대한 인간의 선호도 모델링 시도  

---