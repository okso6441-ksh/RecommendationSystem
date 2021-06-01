## Recommendation System

### Code  


---

### Paper
작성중> 2001_IB_CFR [Item-Based Collaborative Filtering Recommendation](https://github.com/okso6441-ksh/RecommendationSystem/tree/main/paper/2001_IB_CFR/README.md)
>   

2008_CF_IFD [Collaborative Filtering for Implicit Feedback Datasets](https://github.com/okso6441-ksh/RecommendationSystem/tree/main/paper/2008_CF_IFD/README.md)
> CF, Implicit, profile/관계(내적), cold-start, (+) 간단한 설명, 이웃, 잠재요인, ALS  

2009_BPR [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://github.com/okso6441-ksh/RecommendationSystem/tree/main/paper/2009_BPR/README.md)
> 개인화, Item Recommendation, BPR-Opt(최대 사후 추정), LearnBPR(Bootstrap, SGD)
  
2010_FM [Factorization Machines](https://github.com/okso6441-ksh/RecommendationSystem/tree/main/paper/2010_FM/README.md)
> SVM + Factorization, huge sparse, 상호작용 모델링, 일반 예측기  
  
2012_LARS [LARS: A Location-Aware Recommender System](https://github.com/okso6441-ksh/RecommendationSystem/tree/main/paper/2012_LARS/README.md)
> Location, 사용자 분할(적응형 피라미드 구조), 이동/여행 패널티, 선호 지역성  
  
2013_DcbmR [Deep content-based music recommendation](https://github.com/okso6441-ksh/RecommendationSystem/tree/main/paper/2013_DcbmR/README.md)
> 음악(power law), CF(cold-start) > 잠재요소모델(심층 CNN), 콘텐츠 기반, 오디오 신호, MIR, MFCC, WMF(SGD > ALS)  
  
2015_AutoRec [AutoRec: Autoencoders Meet Collaborative Filtering](https://github.com/okso6441-ksh/RecommendationSystem/tree/main/paper/2015_AutoRec/README.md)
> CF + autoencoder, I-AutoRec/U-AutoRec, 최적화(L-BFGS > RProp)  
  
2015_IRSS [Image-based Recommendations on Styles and Substitutes](https://github.com/okso6441-ksh/RecommendationSystem/tree/main/paper/2015_IRSS/README.md)
> appearance, 대안/보완, 시각적 관계에 대한 인간의 개념(수작업/loosely related)  
  
2015_SBRRNN [SESSION-BASED RECOMMENDATIONS WITH RECURRENT NEURAL NETWORKS](https://github.com/okso6441-ksh/RecommendationSystem/tree/main/paper/2015_SBRRNN/README.md)
> RS + RNN, short session-based data, GRU-based RNN, 1-of-N encoding, pairwise(TOP1), Item-KNN  
  
2015_VBPR [VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback](https://github.com/okso6441-ksh/RecommendationSystem/tree/main/paper/2015_VBPR/README.md)
> MF(선호 예측), BPR, 사전훈련 Deep CNN, 항목 visual appearance, 임베딩 공유  
  
2016_WDLRS [Wide & Deep Learning for Recommender Systems](https://github.com/okso6441-ksh/RecommendationSystem/tree/main/paper/2016_WDLRS/README.md)
> memorization + generalization, wide[(linear;일반화; cross-product 변환) FTRL + L1] + deep[(NN;임베딩) AdaGrad], joint training(학습결합), 웜 스타트 시스템(dry run, sanity check)  
  
2017_DAE_CF [Training Deep AutoEncoders for Collaborative Filtering](https://github.com/okso6441-ksh/RecommendationSystem/tree/main/paper/2017_DAE_CF/README.md)
> CF(sparseness/missing 극복), U-AutoRec + deeper(사전훈련 X), SELUs, dropout, iterative output e-feeding  
  
2017_DeepFM [DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://github.com/okso6441-ksh/RecommendationSystem/tree/main/paper/2017_DeepFM/README.md)
> CTR 최대화, low/high(FM/DNN) order feature interactions(end-to-end), feature 엔지니어링 X, 입력공유  
  
2017_JT_RRN [JOINT TRAINING OF RATINGS AND REVIEWS WITH RECURRENT RECOMMENDER NETWORKS](https://github.com/okso6441-ksh/RecommendationSystem/tree/main/paper/2017_JT_RRN/README.md)
> ratings + reviews + temporal patterns, Joint Review-Rating RNN(LSTM, RNN)  
  
2017_Neural_CF [Neural Collaborative Filtering](https://github.com/okso6441-ksh/RecommendationSystem/tree/main/paper/2017_Neural_CF/README.md)
> NCF, CF 내적 의존 극복, Implicit Data, DNN, F/W[User/Item Latent 벡터 > Multi-layers > NCF > 출력], NeuMF[GMF(선형) + MLP(비선형)]  
  
---  

### Article
2009_MFTFRS [MATRIX FACTORIZATION TECHNIQUES FOR RECOMMENDER SYSTEMS](https://github.com/okso6441-ksh/RecommendationSystem/tree/main/article/2009_MFTFRS/README.md)

---
