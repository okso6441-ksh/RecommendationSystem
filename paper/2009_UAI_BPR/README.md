## [2009_UAI_BPR] BPR: Bayesian Personalized Ranking from Implicit Feedback

![main](./image/main.PNG)

---

### Abstract  
* 항목 추천: item 집합에 대한 개인화 된 순위 예측   
  * implicit feedback 가장 일반적인 시나리오 조사  
    * 방법:  
      * MF(Matrix Factorization)  
      * kNN(Adaptive Knearest Neighbor)  
        * 개인화 된 랭킹 item 예측 ㅇ, *직접 최적화 ranking X*  
* BPR-Opt 제시  
  * Bayesian analysis 파생된 최대 사후 추정(maximum posterior estimator)  
  * => 개인화  된 랭킹에 대한 일반적인 최적화 기준  
  * BPR-Opt 모델 최적화 학습 알고리즘  
    * SGD with bootstrap sampling   

---
### 1. Introduction  
* Personalization recommendation - win-win : content providers + customers  
* **Item Recommendation**  
  * item set에 대한 user-specific ranking 만드는 것  
  * item에 대한 user preference: user-system 과거 상호작용으로 학습  
* scenarios  
  * explicit feedback   
  * implicit feedback: 실제 시나리오, 자동 추적(간접적), 상대적 수집 쉬움  

* 작업(4):    
  * 1) BPR-Opt: 최대 사후 추정기(maximum posterior estimator) 파생  
    - AUROC 최대화와 유사점  
  * 2) LearnBPR: BPR-Opt 최대화를 위한 SGD with Bootstrap sampling  
  * 3) state-of-the-art에 LearnBPR 적용 방법  
  * 4) BPR과 다른 학습 방법 비교  

--- 
### 2 Related Work  
* 가장 인기 있는 모델: kNN (k-nearest neighbor) 협업 필터링
  * 유사성 행렬: 휴리스틱(Pearson 상관 관계)> 모델 파라미터로 학습  
* 최근 인기 있는 모델: MF (Matrix Factorization) - implicit and explicit feedback  
  * 초기: SVD - [한계] overfitting > [극복] WR-MF(Weighted Regularized MF)> negative impact 줄일 수 있음 
* Hofmann - 확률론적 잠재 의미 모델(a probabilistic latent semantic model)
* Schmidt-Thieme - 다중 클래스 문제로 변환> 이진 분류기 집합으로 해결  
  
*▲ 모델 parameters 직접 최적화하지 않음*  
*▼ item 쌍 기반 최적화 순위 도출*  

--- 
### 3 Personalized Ranking  
--- 
#### 3.1 Formalization  

#### 3.2 Analysis of the problem setting  
--- 
### 4 Bayesian Personalized Ranking(BPR)  

#### 4.1 BPR Optimization Criterion  

#### 4.2 BPR Learning Algorithm  

#### 4.3 Learning models with BPR  

##### 4.3.1 Matrix Factorization  

##### 4.3.2 Adaptive k-Nearest-Neighbor  
--- 
### 5 Relations to other methods  

#### 5.1 Weighted Regularized Matrix Factorization (WR-MF)  

#### 5.2 Maximum Margin Matrix Factorization(MMMF)  
--- 
### 6. Evaluation  

#### 6.1 Datasets  

#### 6.2 Evaluation Methodology  

#### 6.3 Results and Discussion  

#### 6.4 Non-personalized ranking  
--- 
### 7 Conclusion  

