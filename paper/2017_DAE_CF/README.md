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
2.2 조밀 한 재 공급

During training and inference, an input x ∈ Rn is very sparse because no user can realistically rate but a tiny fractions of all items. 
훈련 및 추론 중에 입력 x ∈ Rn은 사용자가 현실적으로 평가할 수 없지만 모든 항목의 극히 일부에 불과하기 때문에 매우 희박합니다.


On the other hand, autoencoder’s output f (x) is dense.  
반면에 오토 인코더의 출력 f (x)는 조밀합니다.


Lets consider an idealized scenario with a perfect f . 
완벽한 f로 이상적인 시나리오를 고려해 봅시다.


Then f(x)i = xi , ∀i : xi =/ 0 and f(x)i accurately predicts all user’s future ratings for items i : xi = 0. 
그러면   $ f(x)_i = x_i, \forall _i : xi \neq 0 $   및 f (x) i는 항목 i : xi = 0에 대한 모든 사용자의 향후 평가를 정확하게 예측합니다.


This means that if user rates new item k (thereby creating a new vector x0) then f (x)k = x0kand f (x) = f (x0). 
즉, 사용자가 새 항목 k (새로운 벡터 x0 생성)를 평가하면 f (x) k = x0k 및 f (x) = f (x0)가됩니다.


Hence, in this idealized scenario, y = f (x) should be a fixed point of a well trained autoencoder: f (y) = y.
따라서 이 이상적인 시나리오에서 y = f (x)는 잘 훈련 된 오토 인코더의 고정 된 점이어야합니다. f (y) = y.


To explicitly enforce fixed-point constraint and to be able to perform dense training updates, we augment every optimization iteration with an iterative dense re-feeding steps (3 and 4 below) as follows:
고정 소수점 제약 조건을 명시 적으로 적용하고 조밀 한 훈련 업데이트를 수행 할 수 있도록 다음과 같이 반복 조밀 한 재 공급 단계 (아래 3 및 4)로 모든 최적화 반복을 강화합니다.


(1) Given sparse x, compute dense f (x) and loss using equation 1 (forward pass)
(1) 희소 x가 주어지면 조밀 한 f (x)와 방정식 1 (순방향 통과)을 사용하여 손실을 계산합니다.

(2) Compute gradients and perform weight update (backward pass)
(2) 기울기 계산 및 가중치 업데이트 수행 (역방향 패스)

(3) Treat f (x) as a new example and compute f (f (x)). 
(3) f (x)를 새로운 예제로 취급하고 f (f (x))를 계산합니다.

Now both f (x) and f (f (x)) are dense and the loss from equation 1 has all m as non-zeros. (second forward pass)
이제 f (x)와 f (f (x))는 모두 밀도가 높고 방정식 1의 손실은 모두 m이 0이 아닙니다. (두 번째 정방향 패스)


(4) Compute gradients and perform weight update (second backward pass) 
(4) 기울기 계산 및 가중치 업데이트 수행 (두 번째 역방향 패스)


Steps (3) and (4) can be also performed more than once for every iteration.
단계 (3) 및 (4)는 모든 반복에 대해 두 번 이상 수행 할 수도 있습니다.

--- 

### 3 EXPERIMENTS AND RESULTS
#### 3.1 Experiment setup

For the rating prediction task, it is often most relevant to predict future ratings given the past ones instead of predicting ratings missing at random. 
등급 예측 작업의 경우 무작위로 누락 된 등급을 예측하는 대신 과거 등급을 고려하여 미래 등급을 예측하는 것이 가장 적합합니다.


For evaluation purposes we followed exactly by spliing the original Netflix Prize training set into several training and testing intervals based on time. 
평가 목적으로 우리는 원래 Netflix Prize 교육 세트를 시간에 따라 여러 교육 및 테스트 간격으로 정확하게 분할했습니다.


Training interval contains ratings which came in earlier than the ones from testing interval.
훈련 간격에는 테스트 간격보다 이전에 제공된 등급이 포함됩니다.


Testing interval is then randomly split into Test and Validation subsets so that each rating from testing interval has a 50% chance of appearing in either subset. 
그런 다음 테스트 간격은 테스트 및 유효성 검사 하위 집합으로 임의로 분할되어 테스트 간격의 각 등급이 각 하위 집합에 나타날 확률이 50 %입니다.


Users and items that do not appear in the training set are removed from both test and validation subsets.
훈련 세트에 나타나지 않는 사용자와 항목은 테스트 및 검증 하위 집합에서 모두 제거됩니다.

![T1](./image/T1.PNG)

Table 1 provides details on the data sets.
표 1은 데이터 세트에 대한 세부 사항을 제공합니다.


For most of our experiments we uses a batch size of 128, trained using SGD with momentum of 0.9 and learning rate of 0.001.
대부분의 실험에서는 모멘텀이 0.9이고 학습률이 0.001 인 SGD를 사용하여 훈련 된 배치 크기 128을 사용합니다.


We used xavier initialization to initialize parameters. 
매개 변수를 초기화하기 위해 xavier 초기화를 사용했습니다.


Note, that unlike we did not use any layer-wise pre-training. 
우리와 달리 계층 별 사전 훈련은 사용하지 않았습니다.


We believe that we were able to do so successfully because of choosing the right activation function (see Section 3.2).
우리는 올바른 활성화 기능을 선택했기 때문에 성공적으로 그렇게 할 수 있다고 믿습니다 (섹션 3.2 참조).

#### 3.2 Effects of the activation types

To explore the effects of using different activation functions, we tested some of the most popular choices in deep learning : sigmoid, “rectified linear units” (RELU),max(relu(x), 6) or RELU6, hyperbolic tangent (TANH), “exponential linear units” (ELU) [4], leaky relu (LRELU) [20] , and “scaled exponential linear units” [9] (SELU) on the 4 layer autoencoder with 128 units in each hidden layer.
다양한 활성화 함수 사용의 효과를 조사하기 위해 딥 러닝에서 가장 인기있는 몇 가지 선택을 테스트했습니다. 시그 모이 드, "정류 된 선형 단위"(RELU), max (relu (x), 6) 또는 RELU6, 쌍곡 탄젠트 (TANH) , "지수 선형 단위"(ELU) [4], 누출 relu (LRELU) [20] 및 "스케일 된 지수 선형 단위"[9] (SELU)는 각 은닉 계층에 128 개의 단위가있는 4 계층 오토 인코더에 있습니다.


Because ratings are on the scale from 1 to 5, we keep last layer of the decoder linear for sigmoid and tanh-based models. 
등급은 1부터 5까지의 척도이므로 시그 모이 드 및 tanh 기반 모델에 대해 디코더의 마지막 레이어를 선형으로 유지합니다.


In all other models activation function is applied in all layers.
다른 모든 모델에서 활성화 기능은 모든 레이어에 적용됩니다.


We found that on this task ELU, SELU and LRELU perform much better than SIGMOID, RELU, RELU6 and TANH. 
이 작업에서 ELU, SELU 및 LRELU가 SIGMOID, RELU, RELU6 및 TANH보다 훨씬 더 나은 성능을 발휘한다는 것을 발견했습니다.

![Fig2](./image/Fig2.PNG)

Figure 2 clearly demonstrates this. 
그림 2는 이것을 명확하게 보여줍니다.


There are two properties which seems to separate activations which perform well from those which do not: 
a) non-zero negative part and b) unbounded positive part. 
잘 수행되는 활성화와 그렇지 않은 활성화를 구분하는 두 가지 속성이 있습니다.
a) 0이 아닌 음의 부분 및 b) 무한한 양의 부분.


Hence, we conclude, that in this seing these properties are important for successful training. 
따라서 우리는 이러한 속성이 성공적인 훈련에 중요하다는 결론을 내립니다.


Thus, we use SELU activation units and tune SELU-based networks for performance.
따라서 SELU 활성화 장치를 사용하고 성능을 위해 SELU 기반 네트워크를 조정합니다.


#### 3.3 Over-fitting the data


The largest data set we use for training, “Netflix Full” from Table 1, contains 98M ratings given by 477K users. 
우리가 학습에 사용하는 가장 큰 데이터 세트 인 표 1의 "Netflix Full"에는 477K 사용자가 제공 한 9800 만 등급이 포함되어 있습니다.


Number of movies (e.g. items) in this set is n = 17, 768. 
이 세트의 영화 (예 : 항목) 수는 n = 17, 768입니다.


Therefore, the first layer of encoder will have d ∗ n +d weights, where d is number of units in the layer.
따라서 인코더의 첫 번째 계층에는 d * n + d 가중치가 있으며 여기서 d는 계층의 단위 수입니다.


For modern deep learning algorithms and hardware this is relatively small task. 
최신 딥 러닝 알고리즘 및 하드웨어의 경우 이것은 비교적 작은 작업입니다.


If we start with single layer encoders and decoders we can quickly overfit to the training data even for d as small as 512. 
단일 레이어 인코더 및 디코더로 시작하면 d가 512만큼 작은 경우에도 훈련 데이터에 빠르게 과적 합할 수 있습니다.

![Fig3](./image/Fig3.PNG)

Figure 3 clearly demonstrates this. 
그림 3은 이것을 명확하게 보여줍니다.


Switching from unconstrained autoencoder to constrained reduces over-fitting, but does not completely solve the problem.
제한되지 않은 오토 인코더에서 제한됨으로 전환하면 과적 합이 줄어들지 만 문제가 완전히 해결되지는 않습니다.


#### 3.4 Going deeper
#### 3.4 자세히 알아보기


While making layers wider helps bring training loss down, adding more layers is often correlated with a network’s ability to generalize.
레이어를 넓게 만들면 학습 손실을 줄이는 데 도움이되지만 레이어를 더 추가하는 것은 종종 네트워크의 일반화 능력과 관련이 있습니다.


In this set of experiments we show that this is indeed the case here. 
이 일련의 실험에서 우리는 이것이 실제로 여기에 해당됨을 보여줍니다.


We choose small enough dimensionality (d = 128) for all hidden layers to easily avoid over-fitting and start adding more layers. 
모든 히든 레이어에 대해 충분히 작은 차원 (d = 128)을 선택하여 쉽게 과적 합을 피하고 더 많은 레이어를 추가하기 시작합니다.

![T2](./image/T2.PNG)

Table 2 shows that there is a positive correlation between the number of layers and the evaluation accuracy.
표 2는 레이어 수와 평가 정확도 사이에 양의 상관 관계가 있음을 보여줍니다.


Going from one layer in encoder and decoder to three layers in both provides good improvement in evaluation RMSE (from 1.146 to 0.9378). 
인코더와 디코더의 한 레이어에서 두 레이어 모두에서 3 개의 레이어로 이동하면 평가 RMSE가 향상됩니다 (1.146에서 0.9378로).


After that, blindly adding more layers does help, however it provides diminishing returns. 
그 후 맹목적으로 더 많은 레이어를 추가하면 도움이되지만 수익이 감소합니다.


Note that the model with single d = 256 layer in encoder and decoder has 9,115,240 parameters which is almost two times more than any of these deep models while having much worse evauation RMSE (above 1.0).
인코더 및 디코더에서 단일 d = 256 레이어를 가진 모델에는 9,115,240 개의 매개 변수가 있으며 이는 이러한 심층 모델보다 거의 두 배 더 많은 반면 평가 RMSE (1.0 이상)는 훨씬 더 나쁩니다.

#### 3.5 Dropout
Section 3.4 shows us that adding too many small layers eventually hits diminishing returns. 
섹션 3.4는 너무 많은 작은 레이어를 추가하면 결국 수익이 감소한다는 것을 보여줍니다.


Thus, we start experimenting with model architecture and hyper-parameters more broadly. 
따라서 모델 아키텍처와 하이퍼 파라미터를보다 광범위하게 실험하기 시작합니다.


Our most promising model has the following architecture: n, 512, 512, 1024, 512, 512,n, which means 3 layers in encoder (512,512,1024), coding layer of 1024 and 3 layers in decoder of size 512,512,n. 
우리의 가장 유망한 모델은 n, 512, 512, 1024, 512, 512, n 아키텍처를 가지고 있습니다. 즉, 인코더의 3 개 레이어 (512,512,1024), 1024의 코딩 레이어 및 512,512, n 크기의 디코더의 3 개 레이어를 의미합니다.


This model, however, quickly over-fits if trained with no regularization. 
그러나 이 모델은 정규화없이 훈련 된 경우 빠르게 과적 합됩니다.


To regularize it, we tried several dropout values and, interestingly, very high values of drop probability (e.g. 0.8) turned out to be the best. 
이를 정규화하기 위해 몇 가지 드롭 아웃 값을 시도했고 흥미롭게도 매우 높은 드롭 확률 값 (예 : 0.8)이 최고로 판명되었습니다.

![Fig4](./image/Fig4.PNG)

See Figure 4 for evaluation RMSE. 
RMSE 평가는 그림 4를 참조하십시오.


We apply dropout on the encoder output only, e.g. f (x) = decode(dropout(encode(x))). 
엔코더 출력에만 드롭 아웃을 적용합니다. f (x) = 디코드 (dropout (encode (x))).


We tried applying dropout after every layer of the model but that stifled training convergence and did not improve generalization.
우리는 모델의 모든 계층 이후에 드롭 아웃을 적용하려고 시도했지만 훈련 수렴을 억제하고 일반화를 개선하지 못했습니다.


#### 3.6 Dense re-feeding
#### 3.6 조밀 한 재 공급

Iterative dense re-feeding (see Section 2.2) provides us with additional improvement in evaluation accuracy for our 6-layer-model: 
n, 512, 512, 1024,dp(0.8), 512, 512,n (referred to as Baseline below).
반복적 인 고밀도 재 공급 (2.2 절 참조)은 6- 레이어 모델의 평가 정확도를 추가로 개선합니다.
n, 512, 512, 1024, dp (0.8), 512, 512, n (아래 기준선이라고 함).


Here each parameter denotes the number of inputs, hidden units, or outputs and dp(0.8) is a dropout layer with a drop probability of 0.8. 
여기서 각 매개 변수는 입력, 은닉 유닛 또는 출력의 수를 나타내며 dp (0.8)는 드롭 확률이 0.8 인 드롭 아웃 레이어입니다.


Just applying output re-feeding did not have signicant impact on the model performance. 
출력 재 공급을 적용하는 것만으로는 모델 성능에 큰 영향을 미치지 않았습니다.


However, in conjunction with the higher learning rate, it did signicantly increase the model performance.
그러나 더 높은 학습률과 함께 모델 성능이 크게 향상되었습니다.


Note, that with this higher learning rate (0.005) but without dense re-feeding, the model started to diverge. 
이 학습률 (0.005)이 높지만 조밀 한 재 공급없이 모델이 발산하기 시작했습니다.

![Fig5](./image/Fig5.PNG)

See Figure 5 for details.
자세한 내용은 그림 5를 참조하십시오.


Applying dense re-feeding and increasing the learning rate, allowed us to further improve the evaluation RMSE from 0.9167 to 0.9100. 
조밀 한 재 공급을 적용하고 학습률을 높임으로써 평가 RMSE를 0.9167에서 0.9100으로 더욱 향상시킬 수있었습니다.


Picking a checkpoint with best evaluation RMSE and computing test RMSE gives as 0.9099, which we believe is signicantly better than other methods.
최상의 평가 RMSE 및 컴퓨팅 테스트 RMSE로 체크 포인트를 선택하면 0.9099가 표시되며 이는 다른 방법보다 훨씬 낫다고 생각합니다.


#### 3.7 Comparison with other methods

We compare our best model with Recurrent Recommender Network from [19] which has been shown to outperform PMF, T-SVD [11] and I/U-AR [17] on the data we use (see Table 1 for data description). 
우리는 우리가 사용하는 데이터에서 PMF, T-SVD [11] 및 I / U-AR [17]보다 우수한 것으로 나타난 [19]의 Recurrent Recommender Network와 최상의 모델을 비교합니다 (데이터 설명은 표 1 참조).


Note, that unlike T-SVD and RRN, our method does not explicitly take into account temporal dynamics of ratings.
T-SVD 및 RRN과 달리 우리의 방법은 등급의 시간적 역학을 명시 적으로 고려하지 않습니다.

![T3](./image/T3.PNG)

Yet, Table 3 shows that it is still capable of outperforming these methods on future rating prediction task. 
그러나 표 3은 향후 등급 예측 작업에서 이러한 방법을 능가 할 수 있음을 보여줍니다.


We train each model using only the training set and compute evaluation RMSE for 100 epochs. 
훈련 세트만을 사용하여 각 모델을 훈련하고 100 epoch에 대한 평가 RMSE를 계산합니다.


Then the checkpoint with the highest evaluation RMSE is tested on the test set.
그런 다음 RMSE 평가가 가장 높은 체크 포인트가 테스트 세트에서 테스트됩니다.


“Netflix 3 months” has 7 times less training data compared to “Netflix full”, it is therefore, not surprising that the model’s performance is signicantly worse if trained on this data alone (0.9373 vs 0.9099). 
'Netflix 3 개월'은 'Netflix full'에 비해 학습 데이터가 7 배 적습니다. 따라서이 데이터만으로 학습하면 모델의 성능이 현저히 나빠지는 것은 놀라운 일이 아닙니다 (0.9373 vs 0.9099).


In fact, the model that performs best on “Netflix full” over-fits on this set, and we had to reduce the model’s complexity accordingly (see Table 4 for details).
실제로 "Netflix full"에서 가장 잘 수행되는 모델은이 세트에 과적 합하므로 그에 따라 모델의 복잡성을 줄여야했습니다 (자세한 내용은 표 4 참조).

### 4. CONCLUSION

Deep learning has revolutionized many areas of machine learning, and it is poised do so with recommender systems as well. 
딥 러닝은 머신 러닝의 많은 영역에 혁명을 가져 왔으며 추천 시스템도 마찬가지입니다.


In this paper we demonstrated how very deep autoencoders can be successfully trained even on relatively small amounts of data by using both well established (dropout) and relatively recent (“scaled exponential linear units”) deep learning techniques. 
이 백서에서 우리는 잘 확립 된 (드롭 아웃) 및 비교적 최근의 ( "스케일 된 지수 선형 단위") 딥 러닝 기술을 모두 사용하여 비교적 적은 양의 데이터에서도 매우 딥 오토 인코더를 성공적으로 훈련시킬 수있는 방법을 보여주었습니다.


Further, we introduced iterative output re-feeding - a technique which allowed us to perform dense updates in collaborative filtering, increase learning rate and further improve generalization performance of our model.  
또한 반복적 인 출력 재 공급을 도입했습니다.이 기술은 협업 필터링에서 고밀도 업데이트를 수행하고 학습률을 높이며 모델의 일반화 성능을 더욱 향상시킬 수있는 기술입니다.


On the task of future rating prediction, our model outperforms other approaches even without using additional temporal signals.
미래 등급 예측 작업에서 우리 모델은 추가적인 시간적 신호를 사용하지 않고도 다른 접근 방식을 능가합니다.


While our code supports item-based model (such as I-AutoRec) we argue that this approach is less practical than user-based model (U-AutoRec). 
우리 코드는 항목 기반 모델 (예 : I-AutoRec)을 지원하지만이 접근 방식은 사용자 기반 모델 (U-AutoRec)보다 실용적이지 않다고 주장합니다.

This is because in real-world recommender systems, there are usually much more users then items. 
이는 실제 추천 시스템에서는 일반적으로 항목보다 훨씬 더 많은 사용자가 있기 때문입니다.


Finally, when building personalized recommender system and faced with scaling problems, it can be acceptable to sample items but not users.
마지막으로 개인화 된 추천 시스템을 구축하고 확장 문제에 직면했을 때 샘플 항목은 허용되지만 사용자는 허용 할 수 없습니다.

---