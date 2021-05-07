## [2009_IEEE] MATRIX_FACTORIZATION_TECHNIQUES_FOR_RECOMMENDER_SYSTEMS

![main](./image/main.PNG)

---

### Abstract
- In Producing product Recomeender System,  
　Matrix Factorization > Classic NN  
　(+) implict feedback  
　(+) temporal effect  
　(+) confidence levels  

---
### 추천 시스템 Strategies(2)
##### a. Content Filtering: user/item profile

##### b. Collaborative Filtering: relationship user-item interaction
* 장점: domain free, generally more accuracy
* 단점: cold start problem  
* Area(2): 
  - neighborhood methods  

![CP](./image/CP.PNG)

- latent factor models  

![matrix](./image/matrix.PNG) 

---
### Matrix Factorization Methods
explicit feedback(spease) -> + implicit feedback(dense)
* 강점: 추가 정보를 통합 할 수 있음 
---
### Matrix Factorization Basic Model

* 사용자와 아이템 잠재 요인 공간에 매핑, 상호작용이 공간에서 모델링  
<img src="https://latex.codecogs.com/gif.latex?f"/>: user-item latent space  
<img src="https://latex.codecogs.com/gif.latex?q_i%20%5Cin%20%5Cmathbb%7BR%7D%5Ef"/>: item vector   
<img src="https://latex.codecogs.com/gif.latex?p_u%20%5Cin%20%5Cmathbb%7BR%7D%5Ef"/>: user vector   
          
<img src="https://latex.codecogs.com/gif.latex?%5Chat%7Br%7D%3Dq_i%5ETp_u"/>: 추정치(dot product) => user-item interaction   

* SVD - user-item rating matrix(sparse)  
　imputation fill - expensive/distort  
　overfitting -> avoid => regularization  
  
* 모델 업데이트    
<img src="https://latex.codecogs.com/gif.latex?min%5Csum_%7B%28u%2Ci%29%5Cin%20%5Ckappa%7D%28r_%7Bui%7D-q_i%5ETp_u%29&plus;%20%5Clambda%20%28%5Cleft%20%5C%7C%20q_i%20%5Cright%20%5C%7C%5E2&plus;%5Cleft%20%5C%7C%20p_u%20%5Cright%20%5C%7C%5E2%29"/>
: (min) Error + Regulization  

<img src="https://latex.codecogs.com/gif.latex?%5Ckappa"/>: set of (u,i) pair training set

---
### Learning Algorithms
- 위 수식을 minimize 하기 위한 접근법(2)  
##### a. SGD(Stochastic Gradient Descent)
* 에러를 구하고  
<img src="https://latex.codecogs.com/gif.latex?e_%7Bui%7D%3Dr_%7Bui%7D-q_i%5ETp_u"/>  
* 경사를 업데이트  
<img src="https://latex.codecogs.com/gif.latex?q_i%20%5Cleftarrow%20q_i%20&plus;%20%5Cgamma%20%28e_%7Bui%7D*p_u-%5Clambda%20*q_i%29%2C%20p_u%20%5Cleftarrow%20p_u%20&plus;%20%5Cgamma%20%28e_%7Bui%7D*q_i-%5Clambda%20*p_u%29"/>  
* 강점: 구현 용이성, 빠른 러닝 타임  

##### b. ALS(Alternating least squares)
* <img src="https://latex.codecogs.com/gif.latex?q_i"/> 나 <img src="https://latex.codecogs.com/gif.latex?p_u"/> 모두 미지수이므로 convex 하지 않음  
* 둘 중 하나를 고정하면 convex 하여 풀 수 있음  
* 일반적으로 SGD가 쉽고 빠르지만 ALS 가 유리한 2가지 케이스 존재
  - case1) 병렬화 사용 가능 -> compute each  
  - case2) 암시적 데이터 중심(희소하지 않음)

---
### Adding Biases 
* 상호작용과 관계 없는 변동/쏠림 => 편향/절편  
  - 편향을 식별하여 실제 상호 작용 부분만 모델링 적용  

* Bias 식별:  
<img src="https://latex.codecogs.com/gif.latex?b_%7Bui%7D%20%3D%20%5Cmu%20&plus;%20b_i&plus;%20b_u"/>
<img src="https://latex.codecogs.com/gif.latex?%5Cmu"/>: global average
<img src="https://latex.codecogs.com/gif.latex?b_i"/>: item bias
<img src="https://latex.codecogs.com/gif.latex?b_u"/>: user bias  
  
* 편향을 반영하여 수식 수정 
<img src="https://latex.codecogs.com/gif.latex?%5Chat%7Br%7D_%7Bui%7D%20%3D%20%5Cmu&plus;%20b_i&plus;%20b_u%20&plus;%20q_i%5ETp_u"/>
 = bias + interaction  
* 모델 업데이트  
<img src="https://latex.codecogs.com/gif.latex?min_%7Bp%2Cq%2Cb%7D%5Csum_%7B%28u%2Ci%29%5Cin%20%5Ckappa%7D%28r_%7Bui%7D%20-%20%5Cmu-%20b_i-%20b_u%20-%20q_i%5ETp_u%29%5E2%20&plus;%20%5Clambda%20%28%5Cleft%20%5C%7C%20q_i%20%5Cright%20%5C%7C%5E2&plus;%5Cleft%20%5C%7C%20p_u%20%5Cright%20%5C%7C%5E2%20&plus;%20b_u%5E2%20&plus;%20b_i%5E2%29"/>: (min) Error + Regulization + Bias

---
### Additional Input Sources 
* cold start problem 완화를 위한 추가 정보 소스 
  * +implict feedback

<img src="https://latex.codecogs.com/gif.latex?N%28%5Cmu%29"/>: 사용자 u가 암시적 선호를 표현한 항목 집합 -> 프로파일링  
* item i의 새로운 item factors 집합 
<img src="https://latex.codecogs.com/gif.latex?%5Csum_%7Bi%5Cin%20N%28%5Cmu%29%7D%20x_i"/>  
* 정규화
<img src="https://latex.codecogs.com/gif.latex?%7C%20N%28%5Cmu%29%7C%5E%7B-0.5%7D%5Csum_%7Bi%20%5Cin%20N%28%5Cmu%29%7D%20x_i%5E%7B4.5%7D"/>  

* domographics 인구통계  
<img src="https://latex.codecogs.com/gif.latex?A%28u%29"/>: 사용자 u의 속성 boolean 속성 집합  
* 사용자 관련 속성 집합 
<img src="https://latex.codecogs.com/gif.latex?%5Csum_%7Ba%5Cin%7BA%28u%29%7D%7Dy_a"/>

* 수식 업데이트  
<img src="https://latex.codecogs.com/gif.latex?%5Chat%7Br%7D_%7Bui%7D%20%3D%20%5Cmu&plus;%20b_i&plus;%20b_u%20&plus;%20q_i%5ET%5Bp_u%20&plus;%20N%28%5Cmu%29%7C%5E%7B-0.5%7D%20&plus;%20%5Csum_%7Bi%5Cin%7BN%28u%29%7D%7Dx_i%20&plus;%20%5Csum_%7Ba%5Cin%7BA%28u%29%7D%7Dy_a%5D"/>
: bias + interaction + 사용자 implicit feedback + boolean 속성  

---
### Temporal Pynamics
* time-drifting> [Dynamic(User)] -> evolve -> redefine  
  * Matrix factorization 접근법은 시간 효과 모델링에 적합 -> 정확도 개선   
<img src="https://latex.codecogs.com/gif.latex?b_i%28t%29"/>: item's popularity change  
<img src="https://latex.codecogs.com/gif.latex?b_u%28t%29"/>: baseline rating change  
<img src="https://latex.codecogs.com/gif.latex?p_u%28t%29"/>: perference change  
* [Static(Itme)]  
<img src="https://latex.codecogs.com/gif.latex?q_i"/>: item characteristics  

* 수식 업데이트  
<img src="https://latex.codecogs.com/gif.latex?%5Chat%7Br%7D_%7Bui%7D%20%3D%20%5Cmu&plus;%20b_i%28t%29&plus;%20b_u%28t%29%20&plus;%20q_i%5ETp_u%28t%29"/>: bias + interaction + 사용자 implicit feedback + dynamic prediction rule  

---
### Input with Varying Confidence Levels
* 원인: 외부적 요인, 적대적 사용자, implicit feedback 사용  
* 방안: 예상선호도 + 신뢰도 점수  
  * 신뢰도: 사용 가능한 수치값 => 반복 이벤트로 사용자 의견 반영 가능성 높임  
  * 신뢰 수준별 가중치 부여 
* 모델 업데이트   
<img src="https://latex.codecogs.com/gif.latex?min_%7Bp%2Cq%2Cb%7D%5Csum_%7B%28u%2Ci%29%5Cin%20%5Ckappa%7Dc_%7Bui%7D%28r_%7Bui%20-%20%7D%5Cmu-%20b_i-%20b_u%20-%20q_i%5ETp_u%29%5E2%20&plus;%20%5Clambda%20%28%5Cleft%20%5C%7C%20q_i%20%5Cright%20%5C%7C%5E2&plus;%5Cleft%20%5C%7C%20p_u%20%5Cright%20%5C%7C%5E2%20&plus;%20b_u%5E2%20&plus;%20b_i%5E2%29"/>: (min) Error + Regulization + Bias + 가중치 부여  

<img src="https://latex.codecogs.com/gif.latex?c_%7Bui%7D"/> : confidence   