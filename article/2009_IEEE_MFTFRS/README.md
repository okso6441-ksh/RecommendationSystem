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

$min\sum_{(u,i)\in K}(r_{ui}-q_i^Tp_u) + \lambda (\left \| q_i \right \|^2+\left \| p_u \right \|^2)$

$e_{ui}=r_{ui}-q_i^Tp_u$

- $q_i \leftarrow q_i + \gamma (e_{ui}*p_u-\lambda *q_i)$
- $p_u \leftarrow p_u + \gamma (e_{ui}*q_i-\lambda *p_u)$

$b_{ui} = \mu  + b_i+ b_u$

$\hat{r}_{ui} = \mu+ b_i+ b_u + q_i^Tp_u$


$min_{p,q,b}\sum_{(u,i)\in K}(\mu- b_i- b_u - q_i^Tp_u)^2 + \lambda (\left \| q_i \right \|^2+\left \| p_u \right \|^2 + b_u^2 + b_i^2)$

$\sum_{i\in N(\mu)} x_i$

$| N(\mu)|^{-0.5}\sum_{i \in N(\mu)} x_i^{4.5}$

$[]$