## [2009_IEEE] MATRIX_FACTORIZATION_TECHNIQUES_FOR_RECOMMENDER_SYSTEMS

![main](./image/main.PNG)

### Abstract
- In Producing product Recomeender System,
Matrix Factorization > Classic NN
+1) implict feedback
+2) temporal effect
+3) confidence levels

### Strategies(2)
1. Content Filtering
2. Collaborative Filtering
 장점: Domain free
 단점: cold start problem
 Area(2): neighborhood methon
          latent factor models

### Methods
explit feedback
implic feedback

### Basic Model
$\hat{r}=q_i^Tp_u$

regularized model 
$min\sum_{(u,i)\in K}(r_{ui}-q_i^Tp_u) + \lambda (\left \| q_i \right \|^2+\left \| p_u \right \|^2)$

learning algorithm
minimizing equation
a. SGD
$e_{ui}=r_{ui}-q_i^Tp_u$
- $q_i \leftarrow q_i + \gamma (e_{ui}*p_u-\lambda *q_i)$
- $p_u \leftarrow p_u + \gamma (e_{ui}*q_i-\lambda *p_u)$

b. ALS(Alternating least squares)
case1) parallelization -> compute each
case2) contered on implict data, not sparse

Adding biases: 쏠린 데이터 설명 

$b_{ui} = \mu  + b_i+ b_u$ 
($\mu$ : global average)

$\hat{r}_{ui} = \mu+ b_i+ b_u + q_i^Tp_u$
 = bias + interaction

$min_{p,q,b}\sum_{(u,i)\in K}(\mu- b_i- b_u - q_i^Tp_u)^2 + \lambda (\left \| q_i \right \|^2+\left \| p_u \right \|^2 + b_u^2 + b_i^2)$

Additional input sources - cold start problem
implict feedback - 사용자 행동 데이터 

$N(\mu)$

$\sum_{i\in N(\mu)} x_i$

$| N(\mu)|^{-0.5}\sum_{i \in N(\mu)} x_i^{4.5}$

Temporal Pynamics
effects
Dynamic(User)
$b_i(t)$: item's popularity change
$b_u(t)$: baseline rating change
$p_u(t)$: perference change
Static(Itme)
$q_i$: item characteristics

imput with varying confidence levels
observed rating $\neq$ same weight/confident

$c_{ui}$ : confidence 


$[]$