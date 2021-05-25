## 2017_DAE_CF [Training Deep AutoEncoders for Collaborative Filtering]

![main](./image/main.PNG)

---

### ABSTRACT
* dataset: time-split Netflix     
* model: based deep autoencoder(6 layers), end-to-end(층별 pre-training X)  
  * a) 일반화: deep autoencoder > shallow  
  * b) training deep models, 비선형 활성화 함수(negative) 중요  
  * c) overfitting 방지: 규제 기술(dropout)  
* iterative output re-feeding 훈련모델 제안 > CF sparseness 극복  
* 코드ㅣ https://github.com/NVIDIA/DeepRecommender  

### 1. INTRODUCTION
* 추천시스템(2):  
  * context-based: contextual factors(지역, 날짜, 시간)   
  * personalized: CF 접근법; user(취향/선호도-implicit)=> 유사성 추론     

* 목표: 정확성 ↑  

* 고전적 CF 문제: m x n martix 누락 항목 추론(성능: RMSE)   

---

### 2. MODEL
* U-AutoRec 접근법 + deeper(*사전훈련 X*)    
  * a) scaled exponential linear units”(SELUs)  
  * b) dropout ↑  
  * d) iterative output re-feeding   

* autoencoder network  
  * 변환(2):   
    * $ encoder(x): R^n -> R^d $    
    * $ decoder(z): R^d -> R^n $    
  * 목표: f(x)=decode(encode(x)); 오류 최소화 - d차원 표현    
  * ![Fig1](./image/Fig1.PNG)  
    * 4-layer autoencoder network  
    * [encoding step] 노이즈 추가 => [autoencoder] de-noising(called)   

* encoder/decoder: feed-forward NN, classical fully connected layers   
  * l = f(W ∗ x + b) 계산    
    * f: 비선형 활성화 함수  
      * 활성화 함수 범위 < 데이터 범위; decoder 마지막 레이어: 선형 유지    
      * 은닉층 활성화 함수 : 0이 아닌 음수 포함; SELU 단위 사용  
  * decoder mirror encoder > 디코더 가중치는 전치된 인코더 가중치와 동일하게 제한/연결 가능 $ W_d^l - W_e^l $(free parameters 2배 적어짐)    

#### Forward pass and inference  
* 사용자 등급 벡터 $ x \in R^n $  
  * n: 항목 수   
  * x: sparse  
* decoder output $ f(x) \in R^n $  
  * dense, corpus 모든 항목 등급 예측 포함  

#### 2.1 Loss function
* ![(1)](./image/(1).PNG)  
  * $ r_i $: 실제 등급  
  * $ y_i $: 예측 등급  
  * $ m_i $: mask, $ r_i $ ≠ 0면 1, 아니면 0  
* RMSE = √MMSE  

#### 2.2 Dense re-feeding
During training and inference, an input x ∈ Rn is very sparse because no user can realistically rate but a tiny fractions of all items. 
2.2 조밀 한 재 공급
훈련 및 추론 중에 입력 x ∈ Rn은 사용자가 현실적으로 평가할 수 없지만 모든 항목의 극히 일부에 불과하기 때문에 매우 희박합니다.


On the other hand, autoencoder’s output f (x) is dense.  
반면에 오토 인코더의 출력 f (x)는 조밀합니다.


Lets consider an idealized scenario with a perfect f . 
완벽한 f로 이상적인 시나리오를 고려해 봅시다.


Then f(x)i = xi , ∀i : xi =/ 0 and f(x)i accurately predicts all user’s future ratings for items i : xi = 0. 
그러면 f (x) i = xi, ∀i : xi = / 0 및 f (x) i는 항목 i : xi = 0에 대한 모든 사용자의 향후 평가를 정확하게 예측합니다.


This means that if user rates new item k (thereby creating a new vector x0) then f (x)k = x0kand f (x) = f (x0). 
즉, 사용자가 새 항목 k (새로운 벡터 x0 생성)를 평가하면 f (x) k = x0k 및 f (x) = f (x0)가됩니다.


Hence, in this idealized scenario, y = f (x) should be a xed point of a well trained autoencoder: f (y) = y.
따라서이 이상화 된 시나리오에서 y = f (x)는 잘 훈련 된 오토 인코더의 xed 점이어야합니다. f (y) = y.


To explicitly enforce xed-point constraint and to be able to perform dense training updates, we augment every optimization iteration with an iterative dense re-feeding steps (3 and 4 below) as follows:
xed-point 제약 조건을 명시 적으로 적용하고 조밀 한 훈련 업데이트를 수행 할 수 있도록 다음과 같이 반복 조밀 한 재 공급 단계 (아래 3 및 4)로 모든 최적화 반복을 강화합니다.


(1) Given sparse x, compute dense f (x) and loss using equation 1 (forward pass)
(1) 희소 x가 주어지면 조밀 한 f (x)와 방정식 1 (순방향 통과)을 사용하여 손실을 계산합니다.


(2) Compute gradients and perform weight update (backward pass)
(2) 기울기 계산 및 가중치 업데이트 수행 (역방향 패스)

(3) Treat f (x) as a new example and compute f (f (x)). 
(3) f (x)를 새로운 예제로 취급하고 f (f (x))를 계산합니다.

Now both f (x) and f (f (x)) are dense and the loss from equation 1 has all m as non-zeros. (second forward pass)

(4) Compute gradients and perform weight update (second backward pass) 

Steps (3) and (4) can be also performed more than once for every iteration.

--- 

### 3 EXPERIMENTS AND RESULTS
#### 3.1 Experiment setup

For the rating prediction task, it is oen most relevant to predict future ratings given the past ones instead of predicting ratings missing at random. 

For evaluation purposes we followed exactly by spliing the original Netflix Prize training set into several training and testing intervals based on time. 

Training interval contains ratings which came in earlier than the ones from testing interval.

Testing interval is then randomly split into Test and Validation subsets so that each rating from testing interval has a 50% chance of appearing in either subset. 

Users and items that do not appear in the training set are removed from both test and validation subsets.
![T1](./image/T1.PNG)
Table 1 provides details on the data sets.

For most of our experiments we uses a batch size of 128, trained using SGD with momentum of 0.9 and learning rate of 0.001.

We used xavier initialization to initialize parameters. 

Note, that unlike we did not use any layer-wise pre-training. 

We believe that we were able to do so successfully because of choosing the right activation function (see Section 3.2).

3.2 Effects of the activation types

To explore the eects of using dierent activation functions, we tested some of the most popular choices in deep learning : sigmoid, “rectied linear units” (RELU),max(relu(x), 6) or RELU6, hyperbolic tangent (TANH), “exponential linear units” (ELU) [4], leaky relu (LRELU) [20] , and “scaled exponential linear units” [9] (SELU) on the 4 layer autoencoder with 128 units in each hidden layer.

Because ratings are on the scale from 1 to 5, we keep last layer of the decoder linear for sigmoid and tanh-based models. 

In all other models activation function is applied in all layers.

We found that on this task ELU, SELU and LRELU perform much beer than SIGMOID, RELU, RELU6 and TANH. 

![Fig2](./image/Fig2.PNG)

Figure 2 clearly demonstrates this. 

There are two properties which seems to separate activations which perform well from those which do not: 
a) non-zero negative part and b) unbounded positive part. 

Hence, we conclude, that in this seing these properties are important for successful training. 

Thus, we use SELU activation units and tune SELU-based networks for performance.

3.3 Over-tting the data

The largest data set we use for training, “Netflix Full” from Table 1, contains 98M ratings given by 477K users. 

Number of movies (e.g. items) in this set is n = 17, 768. 

Therefore, the first layer of encoder will have d ∗ n +d weights, where d is number of units in the layer.

For modern deep learning algorithms and hardware this is relatively small task. 

If we start with single layer encoders and decoders we can quickly overt to the training data even for d as small as 512. 

![Fig3](./image/Fig3.PNG)

Figure 3 clearly demonstrates this. 

Switching from unconstrained autoencoder to constrained reduces over-ing, but does not completely solve the problem.

#### 3.4 Going deeper

While making layers wider helps bring training loss down, adding more layers is oen correlated with a network’s ability to generalize.

In this set of experiments we show that this is indeed the case here. 

We choose small enough dimensionality (d = 128) for all hidden layers to easily avoid over-ing and start adding more layers. 
![T2](./image/T2.PNG)
Table 2 shows that there is a positive correlation between the number of layers and the evaluation accuracy.

Going from one layer in encoder and decoder to three layers in both provides good improvement in evaluation RMSE (from 1.146 to 0.9378). 

After that, blindly adding more layers does help, however it provides diminishing returns. 

Note that the model with single d = 256 layer in encoder and decoder has 9,115,240 parameters which is almost two times more than any of these deep models while having much worse evauation RMSE (above 1.0).

#### 3.5 Dropout
Section 3.4 shows us that adding too many small layers eventually hits diminishing returns. 

Thus, we start experimenting with model architecture and hyper-parameters more broadly. 

Our most promising model has the following architecture: n, 512, 512, 1024, 512, 512,n, which means 3 layers in encoder (512,512,1024), coding layer of 1024 and 3 layers in decoder of size 512,512,n. 

This model, however, quickly over-ts if trained with no regularization. 

To regularize it, we tried several dropout values and, interestingly, very high values of drop probability (e.g. 0.8) turned out to be the best. 
![Fig4](./image/Fig4.PNG)
See Figure 4 for evaluation RMSE. 

We apply dropout on the encoder output only, e.g. f (x) = decode(dropout(encode(x))). 

We tried applying dropout aer every layer of the model but that stied training convergence and did not improve generalization.

#### 3.6 Dense re-feeding

Iterative dense re-feeding (see Section 2.2) provides us with additional improvement in evaluation accuracy for our 6-layer-model: 
n, 512, 512, 1024,dp(0.8), 512, 512,n (referred to as Baseline below).

Here each parameter denotes the number of inputs, hidden units, or outputs and dp(0.8) is a dropout layer with a drop probability of 0.8. 

Just applying output re-feeding did not have signicant impact on the model performance. 

However, in conjunction with the higher learning rate, it did signicantly increase the model performance.

Note, that with this higher learning rate (0.005) but without dense re-feeding, the model started to diverge. 
![Fig5](./image/Fig5.PNG)
See Figure 5 for details.

Applying dense re-feeding and increasing the learning rate, allowed us to further improve the evaluation RMSE from 0.9167 to 0.9100. 

Picking a checkpoint with best evaluation RMSE and computing test RMSE gives as 0.9099, which we believe is signicantly beer than other methods.

#### 3.7 Comparison with other methods

We compare our best model with Recurrent Recommender Network from [19] which has been shown to outperform PMF, T-SVD [11] and I/U-AR [17] on the data we use (see Table 1 for data description). 

Note, that unlike T-SVD and RRN, our method does not explicitly take into account temporal dynamics of ratings.
![T3](./image/T3.PNG)
Yet, Table 3 shows that it is still capable of outperforming these methods on future rating prediction task. 

We train each model using only the training set and compute evaluation RMSE for 100 epochs. 

Then the checkpoint with the highest evaluation RMSE is tested on the test set.

“Netflix 3 months” has 7 times less training data compared to “Netflix full”, it is therefore, not surprising that the model’s performance is signicantly worse if trained on this data alone (0.9373 vs 0.9099). 

In fact, the model that performs best on “Netflix full” over-ts on this set, and we had to reduce the model’s complexity accordingly (see Table 4 for details).

### 4. CONCLUSION

Deep learning has revolutionized many areas of machine learning, and it is poised do so with recommender systems as well. 

In this paper we demonstrated how very deep autoencoders can be successfully trained even on relatively small amounts of data by using both well established (dropout) and relatively recent (“scaled exponential linear units”) deep learning techniques. 

Further, we introduced iterative output re-feeding - a technique which allowed us to perform dense updates in collaborative ltering, increase learning rate and further improve generalization performance of our model.  

On the task of future rating prediction, our model outperforms other approaches even without using additional temporal signals.

While our code supports item-based model (such asI-AutoRec) we argue that this approach is less practical than user-based model (UAutoRec). 

This is because in real-world recommender systems, there are usually much more users then items. 

Finally, when building personalized recommender system and faced with scaling problems, it can be acceptable to sample items but not users.
---