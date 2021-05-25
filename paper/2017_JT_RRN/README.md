## 2017_JT_RRN [JOINT TRAINING OF RATINGS AND REVIEWS WITH RECURRENT RECOMMENDER NETWORKS]

![main](./image/main.PNG)

---

### ABSTRACT
Accurate modeling of ratings and text reviews is at the core of successful recommender systems. 
평가 및 텍스트 리뷰의 정확한 모델링은 성공적인 추천 시스템의 핵심입니다.


In this paper, we provide a neural network model that combines ratings, reviews, and temporal patterns to learn highly accurate recommendations. 
이 백서에서는 매우 정확한 권장 사항을 학습하기 위해 평점, 리뷰 및 시간 패턴을 결합한 신경망 모델을 제공합니다.


We co-train for prediction on both numerical ratings and natural language reviews, as well as using a recurrent architecture to capture the dynamic components of users’ and items’ states. 
우리는 사용자 및 항목 상태의 동적 구성 요소를 캡처하기 위해 반복 아키텍처를 사용하는 것뿐만 아니라 수치 평가 및 자연어 리뷰 모두에 대한 예측을 공동 훈련합니다.


We demonstrate that incorporating text reviews and temporal dynamic gives state-of-the-art results over the IMDb dataset. 
텍스트 리뷰와 시간적 역학을 통합하면 IMDb 데이터 세트에 대한 최첨단 결과를 얻을 수 있음을 보여줍니다.


1. INTRODUCTION
1. 소개


Designing highly accurate recommender systems has been the focus of research in many communities and at the center of many products for the past decade. 
매우 정확한 추천 시스템 설계는 지난 10 년 동안 많은 커뮤니티와 많은 제품의 중심에서 연구의 초점이었습니다.


The core goal is to predict which items a given user will like or dislike, typically based on a database of previous ratings and reviews (Salakhutdinov & Mnih, 2008; Beutel et al., 2015; McAuley & Leskovec, 2013; Diao et al., 2014).
핵심 목표는 일반적으로 이전 평가 및 리뷰 데이터베이스를 기반으로 특정 사용자가 좋아하거나 싫어할 항목을 예측하는 것입니다 (Salakhutdinov & Mnih, 2008; Beutel et al., 2015; McAuley & Leskovec, 2013; Diao et al. , 2014).


This previous research has been remarkably successful, but has two significant limitations that we discuss and address in this paper. 
이 이전 연구는 매우 성공적 이었지만이 백서에서 논의하고 해결하는 두 가지 중요한 한계가 있습니다.


First, prediction accuracy has rarely been measured by the ability of a model to predict future ratings. 
첫째, 예측 정확도는 미래의 등급을 예측하는 모델의 능력으로 거의 측정되지 않았습니다.


Rather, recommendation accuracy has been derived from a random split of the ratings data, which undermines our understanding of the models’ usefulness in practice. 
오히려 추천 정확도는 평점 데이터의 무작위 분할에서 파생 되었기 때문에 실제로 모델의 유용성에 대한 이해를 훼손합니다.


More recently, Recurrent Recommender Networks (RRN) use a recurrent neural network to capture changes in both user preferences and item perceptions, and extrapolate future ratings in an autoregressive way (Wu et al., 2017). 
최근에 RRN (Recurrent Recommender Networks)은 순환 신경망을 사용하여 사용자 선호도 및 항목 인식의 변화를 포착하고 자기 회귀 방식으로 미래 등급을 추정합니다 (Wu et al., 2017).


However, temporal patterns in reviews are largely unexplored. 
그러나 리뷰의 시간적 패턴은 대부분 탐구되지 않았습니다.


Second, models of reviews in recommender system fall significantly behind the state-of-the-art in natural language processing. 
둘째, 추천 시스템의 리뷰 모델은 자연어 처리의 최신 기술보다 훨씬 뒤떨어져 있습니다.


The bag-of-words model used in previous research improves over not using text, but is limited in the degree to which it can understand the review. 
이전 연구에서 사용 된 bag-of-words 모델은 텍스트를 사용하지 않는 것보다 향상되지만 리뷰를 이해할 수있는 정도에 제한이 있습니다.


In fact, the drawback of an underfitting model is especially salient in the case of reviews, because they are much more diverse and unstructured than regular documents.
사실, 리뷰의 경우 부족한 모델의 단점은 특히 일반 문서보다 훨씬 다양하고 구조화되지 않았기 때문에 두드러집니다.


Here, we combine these powerful neural-based language models with Long Short-Term Memory (LSTM) (Hochreiter & Schmidhuber, 1997) recurrent neural networks (RNN) to learn both accurate recommendations and accurate reviews. 
여기에서는 이러한 강력한 신경 기반 언어 모델을 LSTM (Long Short-Term Memory) (Hochreiter & Schmidhuber, 1997) RNN (순환 신경망)과 결합하여 정확한 권장 사항과 정확한 리뷰를 모두 학습합니다.


Our main contributions are as follows:
우리의 주요 기여는 다음과 같습니다.

• Joint generative model: We propose a novel joint model of ratings and reviews via interacting recurrent networks (particularly LSTM).
• Nonlinear nonparametric review model: By learning a function of user and movie state dynamics, we can capture the evolution of reviews (as well as ratings) over time.
• Experiments show that by jointly modeling ratings and reviews along with temporal patterns, our model achieves state-of-the-art results on IMDb dataset in terms of forward prediction, i.e. in the realistic scenario where we use only ratings strictly prior to prediction time to predict future ratings.
• 공동 생성 모델 : 상호 작용하는 반복 네트워크 (특히 LSTM)를 통해 평가 및 리뷰의 새로운 공동 모델을 제안합니다.
• 비선형 비모수 리뷰 모델 : 사용자 및 영화 상태 역학의 기능을 학습함으로써 시간 경과에 따른 리뷰 (평점 및 평가)의 진화를 포착 할 수 있습니다.
• 실험에 따르면 시간 패턴과 함께 등급 및 리뷰를 공동 모델링함으로써 우리 모델은 순방향 예측 측면에서 IMDb 데이터 세트에 대한 최첨단 결과를 달성합니다. 즉, 예측 시간 이전에 등급 만 엄격하게 사용하는 현실적인 시나리오에서 미래의 등급을 예측합니다.


2. MODEL

Figure 1 shows a depiction of our model: Joint Review-Rating Recurrent Recommender Network.
그림 1은 우리 모델 인 Joint Review-Rating Recurrent Recommender Network를 보여줍니다.


Here we use two LSTM RNNs that take user/movie history as input to capture the temporal dynamics in both user and movie states. 
여기에서는 사용자 / 영화 이력을 입력으로 사용하는 두 개의 LSTM RNN을 사용하여 사용자 및 영화 상태 모두에서 시간적 역학을 캡처합니다.


Given stationary and dynamic states of user i and movie j, we define generator functions that emit both rating rij |t and reviews oij |t at time step t. 
사용자 i와 영화 j의 고정 및 동적 상태가 주어지면 시간 단계 t에서 등급 rij | t와 리뷰 oij | t를 모두 내보내는 생성기 함수를 정의합니다.

Formally
공식적으로

2-1

where ui and mj denote stationary states, and uit and mit denote the dynamic state at t. 
여기서 ui 및 mj는 고정 상태를 나타내고 uit 및 mit은 t에서 동적 상태를 나타냅니다.


Note that here essentially we learn the functions that find the states instead of learning the states directly.
여기서는 본질적으로 상태를 직접 학습하는 대신 상태를 찾는 함수를 학습합니다.


Dynamic User and Movie State The key idea is to use user/movie rating history as inputs to update the states. 
동적 사용자 및 영화 상태 주요 아이디어는 사용자 / 영화 등급 기록을 입력으로 사용하여 상태를 업데이트하는 것입니다.


In this way we can model e.g. the change of user (movie) state caused by having watched and liked/disliked a movie (being liked/disliked by certain users). 
이런 식으로 우리는 예를 들어 모델링 할 수 있습니다. 영화를보고 좋아요 / 싫어요 (특정 사용자에 의해 좋아요 / 싫어요)되어 발생하는 사용자 (영화) 상태의 변화.


At each step of the user-state RNN, the network takes yt := Wembed [xt, 1newbie, τt, τt−1], where xt is the rating vector, 1newbie is the indicator for new users, and τt is wall-clock time. 
사용자 상태 RNN의 각 단계에서 네트워크는 yt : = Wembed [xt, 1newbie, τt, τt−1]를 사용합니다. 여기서 xt는 등급 벡터, 1newbie는 신규 사용자에 대한 표시기, τt는 벽시계입니다. 시각.


The jth element of xt is the rating the user gives for movie j at time t, and 0 otherwise. 
xt의 j 번째 요소는 사용자가 시간 t에서 영화 j에 대해 제공하는 등급이고 그렇지 않으면 0입니다.


The state update is given by standard ut := LSTM(ut−1, yt).
상태 업데이트는 표준 ut : = LSTM (ut−1, yt)에 의해 제공됩니다.


In the above we omit user index for clarity. 
위에서 우리는 명확성을 위해 사용자 색인을 생략했습니다.


The movie-state RNN is defined in the same way.
영화 상태 RNN도 같은 방식으로 정의됩니다.


Rating Emissions We supplement the time-varying profile vectors uit and mjt with stationary ones ui and mj respectively. 
방출 등급 평가 우리는 시간에 따라 변하는 프로파일 벡터 uit와 mjt를 각각 고정 된 ui와 mj로 보완합니다.


These stationary components encode time-invariant properties such as long-term preference of a user or the genre of a movie.
이러한 고정 구성 요소는 사용자의 장기적 선호 또는 영화 장르와 같은 시간 불변 속성을 인코딩합니다.


The review rating is thus modeled as a function of both dynamic and stationary states, i.e.
따라서 리뷰 평점은 동적 및 정지 상태의 함수로 모델링됩니다.

(1)

where u˜it and m˜ jt are affine functions of uit and mjt respectively. 
여기서 u ~ it 및 m ~ jt는 각각 uit 및 mjt의 아핀 함수입니다.


That is, we have
즉, 우리는

2-2


2-2

Review Text Model Review text is modeled by a character-level LSTM network. 
리뷰 텍스트 모델 리뷰 텍스트는 문자 수준 LSTM 네트워크에 의해 모델링됩니다.


This network shares the same user/movie latent states with the rating model. 
이 네트워크는 등급 모델과 동일한 사용자 / 영화 잠재 상태를 공유합니다.


We fuse the stationary and dynamic states of both user of movie by the bottleneck layer xjoint,ij given below:
우리는 아래에 주어진 병목 레이어 xjoint에 의해 영화 사용자의 고정 상태와 동적 상태를 융합합니다.


2-3
2-3


where oij,k denotes the character at position k for the review given by user i to movie j, and xoij,k denotes the embedding of the character. 
여기서 oij, k는 사용자 i가 영화 j에 제공 한 리뷰를 위해 위치 k에있는 문자를 나타내고 xoij, k는 문자 삽입을 나타냅니다.


φ here is some non-linear function. 
여기서 φ는 비선형 함수입니다.


The review text emission model is itself an RNN, specifically a character-level LSTM generative model. 
리뷰 텍스트 방출 모델은 그 자체로 RNN, 특히 문자 수준 LSTM 생성 모델입니다.


For character index k = 1, 2, . . . 
문자 인덱스 k = 1, 2,. . .

2-4


2-4

Here a softmax layer at output of LSTM is used to predict the next character.
여기서 LSTM의 출력에있는 소프트 맥스 레이어는 다음 문자를 예측하는 데 사용됩니다.

Training & Prediction Our goal is to predict both accurate ratings and accurate reviews, and thus we minimize L := P(i,j)∈Dtrain h(ˆrij (θ) − rij )2 − λPnijk=1 log (Pr(oij,k|θ))i, where Dtrain is the training set of (i, j) pairs, θ denotes all model parameters, nij is the number of characters in the review user i gives to movie j, and λ controls the weight between predicting accurate ratings and predicting accurate reviews.
훈련 및 예측 우리의 목표는 정확한 평가와 정확한 리뷰를 모두 예측하는 것이므로 L : = P (i, j) ∈Dtrain h (ˆrij (θ) − rij) 2 − λPnijk = 1 log (Pr (oij, k | θ)) i, 여기서 Dtrain은 (i, j) 쌍의 훈련 세트, θ는 모든 모델 매개 변수, nij는 리뷰 사용자 i가 영화 j에 제공하는 문자 수, λ는 예측 간의 가중치를 제어합니다. 정확한 평가 및 정확한 리뷰 예측.


In prediction time, we make rating predictions based on predicted future states. 
예측 시간에는 예측 된 미래 상태를 기반으로 등급을 예측합니다.


That is, we take the latest ratings as input to update the states, and use the newly predicted states to predict ratings. 
즉, 최신 등급을 입력으로 사용하여 상태를 업데이트하고 새로 예측 된 상태를 사용하여 등급을 예측합니다.


This differs from traditional approaches where embeddings are estimated instead of inferred.
이것은 임베딩이 추론되는 대신 추정되는 기존의 접근 방식과 다릅니다.


3. EXPERIMENTS


We evaluate our model on a k-core of IMDb dataset, first used in Diao et al. (2014), that is the only large-scale movie review dataset available. 
우리는 Diao 등에서 처음 사용 된 IMDb 데이터 세트의 k- 코어에서 모델을 평가합니다. (2014), 이것이 유일한 대규모 영화 리뷰 데이터 세트입니다.


The training set contains all ratings from July 1998 to December 2012, and the ratings from January to September 2013 are randomly split into a validation set and a test set. 
교육 세트에는 1998 년 7 월부터 2012 년 12 월까지의 모든 등급이 포함되며 2013 년 1 월부터 9 월까지의 등급은 무작위로 검증 세트와 테스트 세트로 나뉩니다.


We compare our model with PMF (Mnih & Salakhutdinov, 2007), the state-of-theart temporal model Time-SVD++ (Koren, 2010), and a state-of-the-art neural network-based model, AutoRec (Sedhain et al., 2015).
우리는 모델을 PMF (Mnih & Salakhutdinov, 2007), 최첨단 시간 모델 Time-SVD ++ (Koren, 2010) 및 최첨단 신경망 기반 모델 인 AutoRec (Sedhain et al., 2015).


Rating prediction The results are summarized in Table 1. 
등급 예측 결과는 표 1에 요약되어 있습니다.

For completeness, we include the results from Wu et al. (2017) on 6-month Netflix dataset that use ratings only to compare the behavior of different models on different datasets. 
완전성을 위해 Wu et al.의 결과를 포함합니다. (2017) 6 개월 Netflix 데이터 세트에서 등급을 사용하여 서로 다른 데이터 세트에서 서로 다른 모델의 동작을 비교합니다.


We see that rating-only RRN outperforms all baseline models in terms of rating prediction consistently in both datasets. 
등급 전용 RRN이 두 데이터 세트에서 일관되게 등급 예측 측면에서 모든 기준 모델을 능가합니다.


More importantly, joint-modeling ratings and reviews boosts the performance even more, compared to rating-only RRN. 
더 중요한 것은 공동 모델링 등급 및 리뷰가 등급 전용 RRN에 비해 성능을 더욱 향상 시킨다는 것입니다.


This implies that by sharing statistical strength between ratings and reviews, the rich information in reviews helps us estimate the latent factors better.
이는 평점과 리뷰간에 통계적 강도를 공유함으로써 리뷰의 풍부한 정보가 잠재 요인을 더 잘 추정하는 데 도움이된다는 것을 의미합니다.


Text modeling We also examine the impact of conditioning on user and item states for text modeling. 
텍스트 모델링 또한 텍스트 모델링을 위해 컨디셔닝이 사용자 및 항목 상태에 미치는 영향을 조사합니다.

Towards this end, we compare perplexity of characters in testing set with and without using the user/item factors. 
이를 위해 사용자 / 항목 요소를 사용하거나 사용하지 않고 테스트 세트에서 문자의 난이도를 비교합니다.


Perplexity is defined as exp  − 1 NcPc∈Dtestlog Pr(c), where Nc is thetotal number of characters in Dtest, and Pr(c) is the likelihood of character c. 
난이도는 exp − 1 NcPc∈Dtestlog Pr (c)로 정의됩니다. 여기서 Nc는 Dtest의 총 문자 수이고 Pr (c)는 문자 c의 가능성입니다.


Interestingly, we found that by jointly training with user and item states, the perplexity improves from 3.3442 to 3.3362. 
흥미롭게도 사용자 및 항목 상태와의 공동 교육을 통해 난이도가 3.3442에서 3.3362로 향상된다는 사실을 발견했습니다.


4. DISCUSSION & CONCLUSION
4. 토론 및 결론


We present a novel approach that jointly models ratings, reviews, and their temporal dynamics with RRN. 
RRN을 사용하여 평가, 리뷰 및 시간적 역학을 공동으로 모델링하는 새로운 접근 방식을 제시합니다.


We demonstrate that our joint model offers state-of-the-art results on rating prediction in real recommendation settings, i.e. predicting into the future.
우리는 우리의 공동 모델이 실제 추천 설정, 즉 미래 예측에서 등급 예측에 대한 최첨단 결과를 제공함을 보여줍니다.    

---