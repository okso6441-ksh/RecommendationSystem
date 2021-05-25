## 2015_CECS_AutoRec [AutoRec: Autoencoders Meet Collaborative Filtering]

![main](./image/main.PNG)

---
### ABSTRACT  
* AutoRec: CF 위한 새로운 autoencoder 프레임워크   

#### Keywords   
* Recommender Systems; Collaborative Filtering; Autoencoders   

---
### 1. INTRODUCTION
* CF: item에 대한 user 선호도 > personalised recommendations     
  * matrix factorisation, neighbourhood models  
* AutoRec: autoencoder paradigm 기반 CF  
  * 주장: representational, computational 이점 존재   

---
### 2. THE AUTOREC MODEL  
* user item rating 행렬  
  * <img src="https://latex.codecogs.com/gif.latex?R%20%5Cin%20%5Cmathbb%20R%20%5E%7Bm%20%5Ctimes%20n%7D">: user-item rating matrix  
  * m: user; u ∈ U = {1 . . . m}; 부분 관측 벡터 <img src="https://latex.codecogs.com/gif.latex?r%5E%7B%28u%29%7D%20%3D%20%7BR_%7Bu1%7D...R_%7Bun%7D%7D%20%5Cin%20%5Cmathbb%20R%5En">     
  * n: item; i ∈ I = {1. . . n}; 부분 관측 벡터 <img src="https://latex.codecogs.com/gif.latex?r%5E%7B%28i%29%7D%20%3D%20%7BR_%7B1i%7D...R_%7Bmi%7D%7D%20%5Cin%20%5Cmathbb%20R%5Em">       

* item-based/user-based autoencoder  
  * input <img src="https://latex.codecogs.com/gif.latex?r%5E%7B%28i%29%7D/r%5E%7B%28u%29%7D"> > project(저차원 잠재(은닉)공간) > <img src="https://latex.codecogs.com/gif.latex?r%5E%7B%28i%29%7D/r%5E%7B%28u%29%7D"> reconstruct(출력공간) > 누락된 등급 예측(추천)    

* 오토인코더  
  * ![(1)](./image/(1).PNG)  
    * S: <img src="https://latex.codecogs.com/gif.latex?%5Cmathbb%20R%20%5Ed"> 벡터 set   
    * <img src="https://latex.codecogs.com/gif.latex?k%20%5Cin%20%5Cmathbb%20N_&plus;">   
    * h(r; θ): <img src="https://latex.codecogs.com/gif.latex?%5Cmathbb%20R%20%5Ed"> 의 재구성  
      * ![2-1](./image/2-1.PNG)
        * f(·), g(·): 활성화 함수    
        * θ = {W, V, µ, b}; 변환 <img src="https://latex.codecogs.com/gif.latex?W%20%5Cin%20%5Cmathbb%20R%5E%7Bd%20%5Ctimes%20k%7D%2C%20V%20%5Cin%20%5Cmathbb%20R%5E%7Bk%20%5Ctimes%20d%7D">, 바이어스 <img src="https://latex.codecogs.com/gif.latex?%5Cmu%20%5Cin%20%5Cmathbb%20R%5Ek%2C%20b%20%5Cin%20%5Cmathbb%20R%5Ed">    
  * 목표: 단일 k-차원 은닉층 있는 auto-associative neural network    
    * θ: backpropagation 학습    

* item-based AutoRec model(I-AutoRec)  
  * ![Fig1](./image/Fig1.PNG)  
  * ![(1)](./image/(1).PNG) 변경사항(2):  
    * 벡터 set( <img src="https://latex.codecogs.com/gif.latex?%5Cleft%20%5C%7B%20r%5E%7B%28i%29%7D%20%5Cright%20%5C%7D_%7Bi%3D1%7D%5En"> ) 등식(1)에 적용  
    * 1) (matrix factorisation & RBM) <img src="https://latex.codecogs.com/gif.latex?r%5E%7B%28i%29%7D"> 부분적 관찰(입력/가중치 역전파 하는 동안만 업데이트)    
    * 2) 과적합 방지를 위한 매개변수 정규화  
  * 관찰된 ratings(shaded nodes), 입력(<img src="https://latex.codecogs.com/gif.latex?r%5E%7B%28i%29%7D">) 업데이트 된 가중치 solid 연결  
  * objective function(규제강도 λ > 0)  
    * ![(2)](./image/(2).PNG)  
      * <img src="https://latex.codecogs.com/gif.latex?%7C%7C%5Ccdot%7C%7C_O%5E2">: 관찰된 등급 기여도만 고려   

* User-based AutoRec model(U-AutoRec)  
  * <img src="https://latex.codecogs.com/gif.latex?%5Cleft%20%5C%7B%20r%5E%7B%28u%29%7D%20%5Cright%20%5C%7D_%7Bu%3D1%7D%5Em"> 작업 파생  
  * 2mk + m + k 파라미터 추정 필요  
  * user-item 예측 등급   
    * ![(3)](./image/(3).PNG)  
      * <img src="https://latex.codecogs.com/gif.latex?%5Chat%20%5Ctheta">: 학습된 매개변수  

* RBM-CF(RBM 기반 CF) v.s AutoRec  
  * 1) generative model v.s discriminative model     
  * 2) 파라미터 추정; maximising log likelihood v.s minimises RMSE  
  * 3) contrastive divergence v.s gradient-based backpropagation(더 빠름)  
  * 4) discrete ratings 적용 v.s agnostic to r  
* matrix factorisation v.s AutoRec  
  * 1) (user-item) embed shared latent space v.s (items) embeds  into latent space  
  * 2) linear latent representation v.s 활성화 함수 g(·) > non-linear latent representation  

---
### 3. EXPERIMENTAL EVALUATION  
* datasets: Movielens, Netflix   
* models: AutoRec, RBM-CF, Biased Matrix Factorisation(BiasedMF),  Local Low-Rank Matrix Factorisation(LLORMA)  

* average RMSE  
  * 95% confidence intervals: ±0.003 이하  

* 규제 강도: λ ∈ {0.001, 0.01, 0.1, 1, 100, 1000}  
* 잠재 차원: k ∈ {10, 20, 40, 80, 100, 200, 300, 400, 500}  

* autoencoders objective: non-convexity    
* 최적화 알고리즘: L-BFGS, RProp(빠름) => RProp 사용   

* ![T1a](./image/T1a.PNG)  
  * 성능: I-AutoRec > RBM   

* ![T1b](./image/T1b.PNG)  
  * f(·): identity,. g(·): sigmoid     

* ![Fig2](./image/Fig2.PNG)  
  * 은닉 유닛 수에 따른 성능  
  
* ![T1c](./image/T1c.PNG)  
  * AutoRec 지속적으로 성능 우수  

---