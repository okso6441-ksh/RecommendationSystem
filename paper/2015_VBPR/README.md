## 2015_VBPR [VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback]

![main](./image/main.PNG)

---

### Abstract

Modern recommender systems model people and items by discovering or ‘teasing apart’ the underlying dimensions that encode the properties of items and users’ preferences toward them. 
최신 추천 시스템은 항목의 속성과 이에 대한 사용자의 선호도를 인코딩하는 기본 차원을 발견하거나 '분리'하여 사람과 항목을 모델링합니다.


Critically, such dimensions are uncovered based on user feedback, often in implicit form (such as purchase histories, browsing logs, etc.); in addition, some recommender systems make use of side information, such as product attributes, temporal information, or review text. 
비판적으로 이러한 차원은 사용자 피드백을 기반으로, 종종 암시 적 형식 (구매 내역, 검색 로그 등)으로 밝혀집니다. 또한 일부 추천 시스템은 제품 속성, 시간 정보 또는 리뷰 텍스트와 같은 부가 정보를 사용합니다.


However one important feature that is typically ignored by existing personalized recommendation and ranking methods is the visual appearance of the items being considered. 
그러나 기존의 개인화 된 추천 및 순위 지정 방법에서 일반적으로 무시되는 중요한 기능 중 하나는 고려중인 항목의 시각적 모양입니다.


In this paper we propose a scalable factorization model to incorporate visual signals into predictors of people’s opinions, which we apply to a selection of large, real-world datasets. 
이 백서에서는 시각적 신호를 사람들의 의견 예측 변수에 통합하는 확장 가능한 인수 분해 모델을 제안합니다.이 모델은 엄선 된 대규모 실제 데이터 세트에 적용됩니다.


We make use of visual features extracted from product images using (pre-trained) deep networks, on top of which we learn an additional layer that uncovers the visual dimensions that best explain the variation in people’s feedback. 
우리는 (사전 훈련 된) 심층 네트워크를 사용하여 제품 이미지에서 추출한 시각적 특징을 활용하며, 그 위에 사람들의 피드백 변화를 가장 잘 설명하는 시각적 차원을 파악하는 추가 레이어를 학습합니다.


This not only leads to significantly more accurate personalized ranking methods, but also helps to alleviate cold start issues, and qualitatively to analyze the visual dimensions that influence people’s opinions.
이는 훨씬 더 정확한 개인화 된 순위 지정 방법으로 이어질뿐만 아니라 콜드 스타트 ​​문제를 완화하고 사람들의 의견에 영향을 미치는 시각적 차원을 질적으로 분석하는 데 도움이됩니다.

---

### Introduction

Modern Recommender Systems (RSs) provide personalized suggestions by learning from historical feedback and uncovering the preferences of users and the properties of the items they consume. 
Modern Recommender Systems (RS)는 과거 피드백을 통해 학습하고 사용자의 선호도와 사용자가 소비하는 항목의 속성을 밝혀 개인화 된 제안을 제공합니다.


Such systems play a central role in helping people discover items of personal interest from huge corpora, ranging from movies and music (Bennett and Lanning, 2007; Koenigstein, Dror, and Koren, 2011), to research articles, news and books (Das et al., 2007; Lu et al., 2015), to tags and even other users (Xu et al., 2011; Zhou et al., 2010; Zhu et al., 2011).
이러한 시스템은 사람들이 영화와 음악 (Bennett and Lanning, 2007; Koenigstein, Dror, Koren, 2011)에서 연구 기사, 뉴스 및 책에 이르기까지 거대한 말뭉치에서 개인적인 관심 항목을 찾는 데 중요한 역할을합니다 (Das et al., 2007; Lu et al., 2015), 태그 및 심지어 다른 사용자에게 (Xu et al., 2011; Zhou et al., 2010; Zhu et al., 2011).


The ‘historical feedback’ used to train such systems may come in the form of explicit feedback such as star ratings, or implicit feedback such as purchase histories, bookmarks, browsing logs, search patterns, mouse activities etc. (Yi et al., 2014). 
이러한 시스템을 교육하는 데 사용되는 '역사적 피드백'은 별표 평점과 같은 명시 적 피드백 또는 구매 내역, 북마크, 검색 로그, 검색 패턴, 마우스 활동 등과 같은 암시 적 피드백의 형태로 제공 될 수 있습니다 (Yi et al., 2014 ).


In order to model user feedback in large, realworld datasets, Matrix Factorization (MF) approaches have been proposed to uncover the most relevant latent dimensions in both explicit and implicit feedback settings (Bell, Koren, and Volinsky, 2007; Hu, Koren, and Volinsky, 2008; Pan et al., 2008; Rendle et al., 2009). 
대규모 실제 데이터 세트에서 사용자 피드백을 모델링하기 위해 명시 적 및 암시 적 피드백 설정 (Bell, Koren 및 Volinsky, 2007; Hu, Koren, and I)에서 가장 관련성이 높은 잠재 차원을 발견하기 위해 Matrix Factorization (MF) 접근법이 제안되었습니다. Volinsky, 2008; Pan et al., 2008; Rendle et al., 2009).


Despite the great success, they suffer from cold start issues due to the sparsity of real-world datasets.
큰 성공에도 불구하고 실제 데이터 세트의 희소성으로 인해 콜드 스타트 ​​문제가 발생합니다.

#### Visual personalized ranking. 
시각적 맞춤형 순위.

Although a variety of sources of data have been used to build hybrid models to make cold start or context-aware recommendations (Schein et al., 2002), from text (Bao, Fang, and Zhang, 2014), to a user’s physical location (Qiao et al., 2014), to the season or temperature (Brown, Bovey, and Chen, 1997), here we are interested in incorporating the visual appearance of the items into the preference predictor, a source of data which is typically neglected by existing RSs. 
다양한 데이터 소스를 사용하여 하이브리드 모델을 구축하여 텍스트 (Bao, Fang 및 Zhang, 2014)에서 사용자의 물리적 위치에 이르기까지 콜드 스타트 ​​또는 상황 인식 권장 사항 (Schein et al., 2002)을 만들었습니다. (Qiao et al., 2014), 계절이나 기온에 따라 (Brown, Bovey, Chen, 1997), 여기서는 항목의 시각적 모양을 일반적으로 무시되는 데이터 소스 인 선호도 예측 자에 통합하는 데 관심이 있습니다. 기존 RS에 의해.


One wouldn’t buy a t-shirt from Amazon without seeing the item in question, and therefore we argue that this important signal should not be ignored when building a system to recommend such products.
문제의 아이템을 보지 않고는 아마존에서 티셔츠를 사지 않을 것이기 때문에 이러한 제품을 추천하는 시스템을 구축 할 때이 중요한 신호를 무시해서는 안된다고 주장합니다.


Building on the success of Matrix Factorization methods at uncovering the latent dimensions/factors of people’s behavior, our goal here is to ask whether it is possible to uncover the visual dimensions that are relevant to people’s opinions, and if so, whether such ‘visual preference’ models shall lead to improved performance at tasks like personalized ranking. 
사람들의 행동의 잠재 차원 / 요인을 밝혀내는 매트릭스 분해 방법의 성공을 기반으로, 우리의 목표는 사람들의 의견과 관련된 시각적 차원을 밝혀 낼 수 있는지, 그렇다면 그러한 '시각적 선호도'를 묻는 것입니다. '모델은 개인화 된 순위와 같은 작업에서 향상된 성능으로 이어질 것입니다.


Answering these questions requires us to develop scalable methods and representations that are capable of handling millions of user actions, in addition to large volumes of visual data (e.g. product images) about the content they consume.
이러한 질문에 답하려면 사용자가 소비하는 콘텐츠에 대한 많은 양의 시각적 데이터 (예 : 제품 이미지) 외에도 수백만 명의 사용자 작업을 처리 할 수있는 확장 가능한 방법과 표현을 개발해야합니다.


In this paper, we develop models that incorporate visual features for the task of personalized ranking on implicit feedback datasets. 
이 백서에서는 암시 적 피드백 데이터 세트에 대한 개인화 된 순위 지정 작업을위한 시각적 기능을 통합하는 모델을 개발합니다.


By learning the visual dimensions people consider when selecting products we will be able to alleviate cold start issues, help explain recommendations in terms of visual signals, and produce personalized rankings that are more consistent with users’ preferences. 
사람들이 제품을 선택할 때 고려하는 시각적 차원을 학습함으로써 우리는 콜드 스타트 ​​문제를 완화하고 시각적 신호 측면에서 권장 사항을 설명하고 사용자의 선호도와 더 일치하는 개인화 된 순위를 생성 할 수 있습니다.


Methodologically we model visual aspects of items by using representations of product images derived from a (pre-trained) deep network (Jia et al., 2014), on top of which we fit an additional layer that uncovers both visual and latent dimensions that are relevant to users’ opinions. 
방법 론적으로 우리는 (사전 훈련 된) 딥 네트워크 (Jia et al., 2014)에서 파생 된 제품 이미지의 표현을 사용하여 항목의 시각적 측면을 모델링합니다. 그 위에 시각적 차원과 잠재 차원을 모두 표시하는 추가 레이어를 적용합니다. 사용자의 의견과 관련이 있습니다.


Although incorporating complex and domain-specific features often requires some amount of manual engineering, we found that visual features are read ily available out-of-the-box that are suitable for our task.
복잡하고 도메인 별 기능을 통합하려면 약간의 수동 엔지니어링이 필요한 경우가 많지만 시각적 기능을 즉시 사용할 수 있으며 작업에 적합하다는 사실을 발견했습니다.

Experimentally our model exhibits significant performance improvements on real-world datasets like Amazon clothing, especially when addressing item cold start problems. 
실험적으로 우리 모델은 특히 아이템 콜드 스타트 문제를 해결할 때 Amazon 의류와 같은 실제 데이터 세트에서 상당한 성능 향상을 보여줍니다.

Specifically, our main contributions are listed as follows:
• We introduce a Matrix Factorization approach that incorporates visual signals into predictors of people’s opinions while scaling to large datasets.
• Derivation and analysis of a Bayesian Personalized Ranking (BPR) based training procedure, which is suitable to uncover visual factors.
• Experiments on large and novel real-world datasets revealing our method’s effectiveness, as well as visualizations of the visual rating space we uncover.
특히, 우리의 주요 기여는 다음과 같이 나열됩니다.
• 대규모 데이터 세트로 확장하면서 시각적 신호를 사람들의 의견 예측 변수에 통합하는 Matrix Factorization 접근 방식을 도입했습니다.
• 시각적 요인을 발견하는 데 적합한 베이지안 개인화 순위 (BPR) 기반 교육 절차의 유도 및 분석.
• 우리가 발견 한 시각적 평가 공간의 시각화뿐만 아니라 방법의 효과를 보여주는 크고 새로운 실제 데이터 세트에 대한 실험.

---

### Related Work

Matrix Factorization (MF) methods relate users and items by uncovering latent dimensions such that users have similar representations to items they rate highly, and are the basis of many state-of-the-art recommendation approaches.(e.g. Bell, Koren, and Volinsky (2007); Bennett and Lanning (2007); Rendle et al. (2009)). 
MF (Matrix Factorization) 방법은 잠재 차원을 발견하여 사용자와 항목을 연관시켜 사용자가 높은 평가 항목과 유사한 표현을 가지고 있으며 많은 최첨단 권장 방법의 기초가됩니다 (예 : Bell, Koren 및 Volinsky). (2007); Bennett and Lanning (2007); Rendle et al. (2009)).


When it comes to personalized ranking from implicit feedback, traditional MF approaches are challenged by the ambiguity of interpreting ‘non-observed’ feedback. 
암시 적 피드백을 통한 개인화 된 순위와 관련하여 기존 MF 접근 방식은 '관찰되지 않은'피드백을 해석하는 모호함으로 인해 어려움을 겪습니다.


In recent years, point-wise and pairwise methods have been successful at adapting MF to address such challenges.
최근 몇 년 동안 포인트 방식 및 페어 방식 방식은 이러한 문제를 해결하기 위해 MF를 채택하는 데 성공했습니다.


Point-wise methods assume non-observed feedback to be inherently negative to some degree. 
포인트 방식 방법은 관찰되지 않은 피드백이 본질적으로 어느 정도 부정적이라고 가정합니다.


They approximate the task with regression which for each user-item pair predicts its affinity score and then ranks items accordingly. 
각 사용자 항목 쌍에 대해 선호도 점수를 예측 한 다음 그에 따라 항목의 순위를 매기는 회귀를 사용하여 작업을 근사합니다.


Hu, Koren, and Volinsky (2008) associate different ‘confidence levels’ to positive and non-observed feedback and then factorize the resulting weighted matrix, while Pan et al. (2008) sample non-observed feedback as negative instances and factorize a similar weighted matrix.
Hu, Koren 및 Volinsky (2008)는 서로 다른 '신뢰 수준'을 긍정 및 비 관찰 피드백에 연결 한 다음 결과 가중치 행렬을 인수 분해하는 반면 Pan et al. (2008)은 관찰되지 않은 피드백을 부정적인 사례로 샘플링하고 유사한 가중치 행렬을 분해합니다.


In contrast to point-wise methods, pairwise methods are based on a weaker but possibly more realistic assumption that positive feedback must only be ‘more preferable’ than non-observed feedback. 
점별 방법과 달리 쌍별 방법은 긍정적 인 피드백이 관찰되지 않은 피드백보다 '더 선호'되어야한다는 더 약하지만 더 현실적인 가정을 기반으로합니다.


Such methods directly optimize the ranking of the feedback and are to our knowledge state-of-the-art for implicit feedback datasets. 
이러한 방법은 피드백의 순위를 직접 최적화하며 암시 적 피드백 데이터 세트에 대한 최신 지식입니다.


Rendle et al.(2009) propose a generalized Bayesian Personalized Ranking (BPR) framework and experimentally show that BPRMF (i.e., with MF as the underlying predictor) outperforms a variety of competitive baselines. 
Rendle 등 (2009)은 일반화 된 Bayesian Personalized Ranking (BPR) 프레임 워크를 제안하고 BPRMF (즉, MF를 기본 예측 변수로 사용)가 다양한 경쟁 기준선을 능가한다는 것을 실험적으로 보여줍니다.

More recently BPR-MF has been extended to accommodate both users’ feedback and their social relations (Krohn-Grimberghe et al., 2012; Pan and Chen, 2013; Zhao, McAuley, and King, 2014). 
최근에 BPR-MF는 사용자의 피드백과 사회적 관계를 모두 수용하도록 확장되었습니다 (Krohn-Grimberghe et al., 2012; Pan and Chen, 2013; Zhao, McAuley, and King, 2014).


Our goal here is complementary as we aim to incorporate visual signals into BPR-MF, which presents a quite different set of challenges compared with other sources of data.
여기에서 우리의 목표는 시각적 신호를 BPR-MF에 통합하는 것을 목표로하기 때문에 보완적인 것입니다. BPR-MF는 다른 데이터 소스와 비교할 때 상당히 다른 문제를 제시합니다.


Others have developed content-based and hybrid models that make use of a variety of information sources, including text (and context), taxonomies, and user demographics (Bao,Fang, and Zhang, 2014; Kanagal et al., 2012; Lu et al., 2015; Qiao et al., 2014). 
다른 기업은 텍스트 (및 컨텍스트), 분류법 및 사용자 인구 통계를 포함한 다양한 정보 소스를 사용하는 콘텐츠 기반 및 하이브리드 모델을 개발했습니다 (Bao, Fang 및 Zhang, 2014; Kanagal et al., 2012; Lu et al. al., 2015; Qiao et al., 2014).


However, to our knowledge none of these works have incorporated visual signals into models of users’preferences and uncover visual dimensions as we do here.
그러나 우리가 아는 한 이러한 작업 중 어느 것도 사용자의 선호도 모델에 시각적 신호를 통합하지 않았으며 여기 에서처럼 시각적 차원을 발견하지 못했습니다.


Exploiting visual signals for the purpose of ‘in-style’ image retrieval has been previously proposed. 
'인스 타일'이미지 검색을 위해 시각적 신호를 악용하는 것이 이전에 제안되었습니다.


For example, Simo-Serra et al. (2014) predict the fashionability of a person in a photograph and suggest subtle improvements. Jagadeesh et al. (2014) use a street fashion dataset with detailed annotations to identify accessories whose style is consistent with a picture. 
예를 들어 Simo-Serra et al. (2014)는 사진 속 인물의 패션 가능성을 예측하고 미묘한 개선을 제안합니다. Jagadeesh et al. (2014)는 자세한 주석이 포함 된 스트리트 패션 데이터 세트를 사용하여 스타일이 그림과 일치하는 액세서리를 식별합니다.


Another method was proposed by Kalantidis, Kennedy, and Li (2013), which accepts a query image and uses segmentation to detect clothing classes before retrieving visually similar products from each of the detected classes. McAuley et al. (2015) use visual features extracted from CNNs and learn a visual similarity metric to identify visually complementary items to a query image.
Kalantidis, Kennedy 및 Li (2013)가 제안한 또 다른 방법은 쿼리 이미지를 받아들이고 감지 된 각 클래스에서 시각적으로 유사한 제품을 검색하기 전에 세분화를 사용하여 의류 클래스를 감지합니다. McAuley et al. (2015) CNN에서 추출한 시각적 특징을 사용하고 시각적 유사성 메트릭을 학습하여 쿼리 이미지에 대한 시각적 보완 항목을 식별합니다.


In contrast to our method, the above works focus on visual retrieval, which differs from recommendation in that such methods aren’t personalized to users based on historical feedback, nor do they take into account other factors besides visual dimensions, both of which are essential for a method to be successful at addressing one-class personalized ranking tasks. 
우리의 방법과 달리 위의 작업은 시각적 검색에 초점을 맞추고 있습니다. 권장 사항과 다른 방법은 이러한 방법이 역사적 피드백을 기반으로 사용자에게 개인화되지 않으며 시각적 차원 외에 필수적인 요소를 고려하지 않습니다. 한 클래스 개인화 된 순위 작업을 성공적으로 처리 할 수있는 방법을 제공합니다.

따라서 이전 작업과 우리의 접근 방식을 구별하는 것은 시각적 및 과거 사용자 피드백 데이터의 조합입니다.

Thus it is the combination of visual and historical user feedback data that distinguishes our approach from prior work.
따라서 이전 작업과 우리의 접근 방식을 구별하는 것은 시각적 및 과거 사용자 피드백 데이터의 조합입니다.

#### Visual Features. 
시각적 특징.


Recently, high-level visual features from Deep Convolutional Neural Networks (‘Deep CNNs’) have seen successes in tasks like object detection (Russakovsky et al., 2014), photographic style annotations (Karayev et al., 2014), and aesthetic quality categorization (Lu et al., 2014), among others. 
최근에 Deep Convolutional Neural Networks ( 'Deep CNN')의 높은 수준의 시각적 기능은 객체 감지 (Russakovsky et al., 2014), 사진 스타일 주석 (Karayev et al., 2014) 및 미적 품질과 같은 작업에서 성공을 거두었습니다. 분류 (Lu et al., 2014) 등이 있습니다.


Furthermore, recent transfer learning studies have demonstrated that CNNs trained on one large dataset (e.g. ImageNet) can be generalized to extract CNN features for other datasets, and outperform stateof-the-art approaches on these new datasets for different visual tasks (Donahue et al., 2014; Razavian et al., 2014). These successes demonstrate the highly generic and descriptive ability of CNN features for visual tasks and persuade us to exploit them for our recommendation task.
또한 최근 전이 학습 연구에 따르면 하나의 큰 데이터 세트 (예 : ImageNet)에서 훈련 된 CNN을 일반화하여 다른 데이터 세트에 대한 CNN 기능을 추출 할 수 있으며 다른 시각적 작업에 대해 이러한 새로운 데이터 세트에 대한 최첨단 접근 방식을 능가 할 수 있음이 입증되었습니다 (Donahue et al ., 2014; Razavian et al., 2014). 이러한 성공은 시각적 작업에 대한 CNN 기능의 매우 일반적이고 설명적인 능력을 보여주고 권장 작업에이를 활용하도록 설득합니다.


---

### VBPR: Visual Bayesian Personalized Ranking 

In this section, we build our visual personalized ranking model (VBPR) to uncover visual and latent (non-visual) dimensions simultaneously. 
이 섹션에서는 시각적 개인화 순위 모델 (VBPR)을 구축하여 시각적 차원과 잠재 (비 시각적) 차원을 동시에 발견합니다.


We first formulate the task in question and introduce our Matrix Factorization based predictor function. 
먼저 문제의 작업을 공식화하고 매트릭스 분해 기반 예측 기능을 소개합니다.


Then we develop our training procedure using a Bayesian Personalized Ranking (BPR) framework. 
그런 다음 베이지안 개인화 순위 (BPR) 프레임 워크를 사용하여 훈련 절차를 개발합니다.

![T1](./image/T1.PNG)

The notation we use throughout this paper is summarized in Table 1.
이 문서에서 사용하는 표기법은 표 1에 요약되어 있습니다.


#### Problem Formulation
문제 공식화

Here we focus on scenarios where the ranking has to be learned from users’ implicit feedback (e.g. purchase histories). 
여기서는 사용자의 암시 적 피드백 (예 : 구매 내역)에서 순위를 배워야하는 시나리오에 중점을 둡니다.


Letting U and I denote the set of users and items respectively, each user u is associated with an item set I+u about which u has expressed explicit positive feedback. 
U와 I가 각각 사용자와 항목의 집합을 나타내면 각 사용자 u는 u가 명시 적으로 긍정적 인 피드백을 표현한 항목 집합 I + u와 연관됩니다.


In addition, a single image is available for each item i ∈ I. 
또한 각 항목 i ∈ I에 대해 단일 이미지를 사용할 수 있습니다.


Using only the above data, our objective is to generate for each user u a personalized ranking of those items about which they haven’t yet provided feedback (i.e. I \ I+u).
위의 데이터 만 사용하여 우리의 목표는 각 사용자에게 아직 피드백을 제공하지 않은 항목 (예 : I \ I + u)의 개인화 된 순위를 생성하는 것입니다.

#### Preference Predictor 

Our preference predictor is built on top of Matrix Factorization (MF), which is state-of-the-art for rating prediction as well as modeling implicit feedback, whose basic formulation assumes the following model to predict the preference of a user u toward an item i (Koren and Bell, 2011):
우리의 선호도 예측기는 평가 예측 및 암시 적 피드백 모델링을위한 최첨단 매트릭스 분해 (MF)를 기반으로하며, 기본 공식은 다음 모델을 가정하여 사용자 u의 선호도를 예측합니다. 항목 i (Koren and Bell, 2011) :

![(1)](./image/(1).PNG)

where α is global offset, βu and βi are user/item bias terms, and γu and γi are K-dimensional vectors describing latent factors of user u and item i (respectively). 
여기서 α는 전역 오프셋, βu 및 βi는 사용자 / 항목 편향 항, γu 및 γi는 사용자 u 및 항목 i (각각)의 잠재 인자를 설명하는 K 차원 벡터입니다.


The inner product γTu γi then encodes the ‘compatibility’ between the user u and the item i, i.e., the extent to which the user’s latent ‘preferences’ are aligned with the products’ ‘properties’.
그런 다음 내적 γTu γi는 사용자 u와 항목 i 사이의 '호환성', 즉 사용자의 잠재 된 '선호'가 제품의 '속성'과 일치하는 정도를 인코딩합니다.


Although theoretically latent factors are able to uncover any relevant dimensions, one major problem it suffers from is the existence of ‘cold’ (or ‘cool’) items in the system, about which there are too few associated observations to estimate their latent dimensions. 
이론적으로 잠재적 인 요소가 관련 차원을 찾아 낼 수 있지만, 이로 인해 겪는 한 가지 주요 문제는 시스템에 '차가운'(또는 '멋진') 항목이 존재한다는 것입니다.이 항목에 대해서는 잠재적 인 차원을 추정하기에는 관련 관측치가 너무 적습니다.


Using explicit features can alleviate this problem by providing an auxiliary signal in such situations. 
명시 적 기능을 사용하면 이러한 상황에서 보조 신호를 제공하여이 문제를 완화 할 수 있습니다.

![Fig1](./image/Fig1.PNG)
In particular, we propose to partition rating dimensions into visual factors and latent (non-visual) factors, as shown in Figure 1. 
특히 그림 1과 같이 등급 차원을 시각적 요인과 잠재 (비 시각적) 요인으로 구분할 것을 제안합니다.


Our extended predictor takes the form 
확장 된 예측자는 다음과 같은 형식을 취합니다.

![(2)](./image/(2).PNG)

where α, β, and γ are as in Eq. 1. θu and θi are newly introduced D-dimensional visual factors whose inner product models the visual interaction between u and i, i.e., the extent to which the user u is attracted to each of D visual dimensions. 
여기서 α, β 및 γ는 Eq. 1. θu와 θi는 새롭게 도입 된 D 차원 시각적 인자로서, 내적은 u와 i 사이의 시각적 상호 작용, 즉 사용자 u가 각 D 시각적 차원에 끌리는 정도를 모델링합니다.


Note that we still use K to represent the number of latent dimensions of our model.
모델의 잠재 차원 수를 나타 내기 위해 여전히 K를 사용합니다.


One naive way to implement the above model would be to directly use Deep CNN features fi of item i as θi in the above equation. 
위 모델을 구현하는 순진한 방법 중 하나는 위 방정식에서 항목 i의 Deep CNN 기능 fi를 θi로 직접 사용하는 것입니다.


However, this would present issues due to the high dimensionality of the features in question, for example the features we use have 4096 dimensions. 
그러나 이것은 문제가되는 기능의 높은 차원 성으로 인해 문제를 야기 할 수 있습니다. 예를 들어 우리가 사용하는 기능에는 4096 차원이 있습니다.


Dimensionality reduction techniques like PCA pose a possible solution, with the potential downside that we would lose much of the expressive power of the original features to explain users’ behavior. 
PCA와 같은 차원 축소 기술은 사용자의 행동을 설명하기 위해 원래 기능의 표현력을 상당 부분 잃게 될 잠재적 인 단점이있는 가능한 솔루션을 제시합니다.


Instead, we propose to learn an embedding kernel which linearly transforms such high-dimensional features into a much lower-dimensional (say 20 or so) ‘visual rating’ space:
대신, 우리는 이러한 고차원 적 특징을 훨씬 낮은 차원 (예 : 20 개 정도)의 '시각적 등급'공간으로 선형 적으로 변환하는 임베딩 커널을 학습 할 것을 제안합니다.

![(3)](./image/(3).PNG)

Here E is a D × F matrix embedding Deep CNN feature space (F-dimensional) into visual space (D-dimensional), where fi is the original visual feature vector for item i. 
여기서 E는 Deep CNN 기능 공간 (F- 차원)을 시각적 공간 (D- 차원)에 포함하는 D × F 행렬이며, 여기서 fi는 항목 i에 대한 원래의 시각적 특징 벡터입니다.


The numerical values of the projected dimensions can then be interpreted as the extent to which an item exhibits a particular visual rating facet. 
투영 된 치수의 수치는 항목이 특정 시각적 등급 패싯을 나타내는 범위로 해석 될 수 있습니다.


This embedding is efficient in the sense that all items share the same embedding matrix which significantly reduces the number of parameters to learn.
이 임베딩은 모든 항목이 동일한 임베딩 행렬을 공유하므로 학습 할 매개 변수의 수가 크게 줄어든다는 점에서 효율적입니다.


Next, we introduce a visual bias term β 0 whose inner product with fi models users’ overall opinion toward the visual appearance of a given item. 
다음으로, fi가있는 내적은 주어진 항목의 시각적 외양에 대한 사용자의 전반적인 의견을 모델링하는 시각적 편향 용어 β 0을 소개합니다.


In summary, our final prediction model is
요약하면 최종 예측 모델은

![(4)](./image/(4).PNG)
#### Model Learning Using BPR 

Bayesian Personalized Ranking (BPR) is a pairwise ranking optimization framework which adopts stochastic gradient ascent as the training procedure. 
Bayesian Personalized Ranking (BPR)은 훈련 절차로 확률 적 경사 상승을 채택하는 쌍별 순위 최적화 프레임 워크입니다.


A training set DS consists of triples of the form (u, i, j), where u denotes the user together with an item i about which they expressed positive feedback, and a non-observed item j:
학습 세트 DS는 (u, i, j) 형식의 트리플로 구성됩니다. 여기서 u는 사용자가 긍정적 인 피드백을 표현한 항목 i 및 관찰되지 않은 항목 j와 함께 사용자를 나타냅니다.
![(5)](./image/(5).PNG)
Following the notation in Rendle et al. (2009), Θ is the parameter vector and xbuij (Θ) denotes an arbitrary function of Θ that parameterises the relationship between the components of the triple (u, i, j). 
Rendle et al.의 표기법에 따라. (2009), Θ는 매개 변수 벡터이고 xbuij (Θ)는 트리플 (u, i, j)의 구성 요소 간의 관계를 매개 변수화하는 Θ의 임의 함수를 나타냅니다.


The following optimization criterion is used for personalized ranking (BPR-OPT):
다음 최적화 기준은 개인화 순위 (BPR-OPT)에 사용됩니다.
![(6)](./image/(6).PNG)
where σ is the logistic (sigmoid) function and λΘ is a modelspecific regularization hyperparameter.
여기서 σ는 로지스틱 (시그 모이 드) 함수이고 λΘ는 모델 별 정규화 하이퍼 파라미터입니다.


When using Matrix Factorization as the preference predictor (i.e., BPR-MF), xbuij is defined as 
Matrix Factorization을 선호도 예측 변수 (즉, BPR-MF)로 사용할 때 xbuij는 다음과 같이 정의됩니다.


![(7)](./image/(7).PNG)

where xbu,i and xbu,j are defined by Eq. 1. BPR-MF can be learned efficiently using stochastic gradient ascent. 
여기서 xbu, i 및 xbu, j는 Eq에 의해 정의됩니다. 1. BPR-MF는 확률 적 경사 상승을 사용하여 효율적으로 학습 할 수 있습니다.


First a triple (u, i, j) is sampled from DS and then the learning algorithm updates parameters in the following fashion:
먼저 트리플 (u, i, j)이 DS에서 샘플링 된 다음 학습 알고리즘이 다음과 같은 방식으로 매개 변수를 업데이트합니다.

![(8)](./image/(8).PNG)

where η is the learning rate.
여기서 η는 학습률입니다.


One merit of our model is that it can be learned efficiently using such a sampling procedure with minor adjustments. 
우리 모델의 장점 중 하나는 약간의 조정으로 이러한 샘플링 절차를 사용하여 효율적으로 학습 할 수 있다는 것입니다.


In our case, xbuij is also defined by Eq. 7 but we instead use Eq. 4 as the predictor function for xbu,i and xbu,j in Eq. 7.
우리의 경우 xbuij도 Eq에 의해 정의됩니다. 그러나 우리는 대신 Eq를 사용합니다. 식에서 xbu, i 및 xbu, j에 대한 예측 함수로 4 7.


Compared to BPR-MF, there are now two sets of parameters to be updated: (a) the non-visual parameters, and (b) the newly-introduced visual parameters. 
BPR-MF에 비해 업데이트 할 매개 변수 세트는 (a) 비 시각적 매개 변수 및 (b) 새로 도입 된 시각적 매개 변수의 두 가지입니다.


Non-visual parameters can be updated in the same form as BPR-MF (therefore are suppressed for brevity), while visual parameters are updated according to:
비 시각적 매개 변수는 BPR-MF와 동일한 형식으로 업데이트 할 수 있으며 (따라서 간결하게 표시하지 않음), 시각적 매개 변수는 다음에 따라 업데이트됩니다.

![1](./image/1.PNG)

Note that our method introduces an additional hyperparameter λE to regularize the embedding matrix E. 
우리의 방법은 임베딩 행렬 E를 정규화하기 위해 추가 하이퍼 파라미터 λE를 도입합니다.


We sample users uniformly to optimize the average AUC across all users to be described in detail later. 
나중에 자세히 설명 할 모든 사용자의 평균 AUC를 최적화하기 위해 사용자를 균일하게 샘플링합니다.


All hyperparameters are tuned using a validation set as we describe in our experimental section later.
모든 하이퍼 파라미터는 나중에 실험 섹션에서 설명하는대로 검증 세트를 사용하여 조정됩니다.


#### Scalability

The efficiency of the underlying BPR-MF makes our models similarly scalable.  
기본 BPR-MF의 효율성으로 인해 모델을 유사하게 확장 할 수 있습니다.


Specifically, BPR-MF requires O(K) to finish updating the parameters for each sampled triple (u, i, j). 
특히 BPR-MF는 샘플링 된 각 트리플 (u, i, j)에 대한 매개 변수 업데이트를 완료하려면 O (K)가 필요합니다.


In our case we need to update the visual parameters as well. 
우리의 경우에는 시각적 매개 변수도 업데이트해야합니다.


In particular, updating θu takes O(D×F) = O(D),β0 takes O(F), and E takes O(D×F) = O(D), where F is the dimension of CNN features (fixed to 4096 in our case).
특히, θu 업데이트는 O (D × F) = O (D), β0은 O (F), E는 O (D × F) = O (D), 여기서 F는 CNN 기능의 차원 (고정 우리의 경우 4096으로).


Therefore the total time complexity of our model for updating each triple is O(K + D) (i.e. O(K) + O(D × F)), i.e., linear in the number of dimensions. 
따라서 각 트리플을 업데이트하는 모델의 총 시간 복잡도는 O (K + D) (즉, O (K) + O (D × F)), 즉 차원 수가 선형입니다.


Note that visual feature vectors (fi) from Deep CNNs are sparse, which significantly reduces the above worst-case running time.
Deep CNN의 시각적 특징 벡터 (fi)는 희소하므로 위의 최악의 실행 시간이 크게 줄어 듭니다.

---

### Experiments 

In this section, we perform experiments on multiple realworld datasets. 
이 섹션에서는 여러 실제 데이터 세트에 대한 실험을 수행합니다.


These datasets include a variety of settings where visual appearance is expected to play a role in consumers’ decision-making process.
이러한 데이터 세트에는 소비자의 의사 결정 과정에서 시각적 인 모습이 역할을 할 것으로 예상되는 다양한 설정이 포함됩니다.


#### Datasets 

The first group of datasets are from Amazon.com introduced by McAuley et al. (2015). 
첫 번째 데이터 세트 그룹은 McAuley 등이 소개 한 Amazon.com에서 가져온 것입니다. (2015).


We consider two large categories where visual features have already been demonstrated to be meaningful, namely Women’s and Men’s Clothing. 
시각적 기능이 이미 의미있는 것으로 입증 된 두 가지 큰 카테고리, 즉 여성 및 남성 의류를 고려합니다.


We also consider Cell Phones & Accessories, where we expect visual characteristics to play a smaller but possibly still significant role. 
또한 시각적 특성이 작지만 여전히 중요한 역할을 할 것으로 예상되는 휴대폰 및 액세서리도 고려합니다.


We take users’ review histories as implicit feedback and use one image per item to extract visual features.
사용자의 리뷰 기록을 암시 적 피드백으로 간주하고 항목 당 하나의 이미지를 사용하여 시각적 특징을 추출합니다.


We also introduce a new dataset from Tradesy.com, a second-hand clothing trading community. 
또한 중고 의류 거래 커뮤니티 인 Tradesy.com의 새로운 데이터 세트를 소개합니다.


It discloses users’purchase histories and ‘thumbs-up’, which we use together as positive feedback. 
사용자의 구매 내역과 '좋아요'를 공개하여 긍정적 인 피드백으로 함께 사용합니다.


Note that recommendation in this setting inherently involves cold start prediction due to the ‘oneoff’ trading characteristic of second-hand markets. 
이 설정의 권장 사항은 중고 시장의 '일회성'거래 특성으로 인해 본질적으로 콜드 스타트 ​​예측을 포함합니다.


Thus to design a meaningful recommender system for such a dataset it is critical that visual information be considered.
따라서 이러한 데이터 세트에 대한 의미있는 추천 시스템을 설계하려면 시각적 정보를 고려하는 것이 중요합니다.


We process each dataset by extracting implicit feedback and visual features as already described. 
이미 설명한대로 암시 적 피드백과 시각적 특징을 추출하여 각 데이터 세트를 처리합니다.


We discard users u where |I+u| < 5. 
| I + u | <5.

![T2](./image/T2.PNG)
Table 2 shows statistics of our datasets, all of which shall be made available at publication time.
표 2는 데이터 세트의 통계를 보여 주며, 모든 데이터는 게시 시점에 제공됩니다.

#### Visual Features 

For each item i in the above datasets, we collect one product image and extract visual features fi using the Caffe reference model (Jia et al., 2014), which implements the CNN architecture proposed by Krizhevsky, Sutskever, and Hinton (2012). 
위 데이터 세트의 각 항목 i에 대해 하나의 제품 이미지를 수집하고 Krizhevsky, Sutskever 및 Hinton (2012)이 제안한 CNN 아키텍처를 구현하는 Caffe 참조 모델 (Jia et al., 2014)을 사용하여 시각적 특징 fi를 추출합니다.


The architecture has 5 convolutional layers followed by 3 fully-connected layers, and has been pre-trained on 1.2 million ImageNet (ILSVRC2010) images. 
이 아키텍처에는 5 개의 컨벌루션 레이어와 3 개의 완전 연결 레이어가 있으며 120 만 ImageNet (ILSVRC2010) 이미지에 대해 사전 학습되었습니다.


In our experiments, we take the output of the second fully-connected layer (i.e. FC7), to obtain an F = 4096 dimensional visual feature vector fi.
실험에서 F = 4096 차원 시각적 특징 벡터 fi를 얻기 위해 두 번째 완전 연결 계층 (즉, FC7)의 출력을 가져옵니다.


#### Evaluation Methodology 

We split our data into training/validation/test sets by selecting for each user u a random item to be used for validation Vu and another for testing Tu. 
각 사용자에 대해 Vu 검증에 사용할 임의의 항목을 선택하고 Tu를 테스트하는 데 다른 항목을 선택하여 데이터를 훈련 / 검증 / 테스트 세트로 분할했습니다.


All remaining data is used for training. 
나머지 모든 데이터는 훈련에 사용됩니다.


The predicted ranking is evaluated on Tu with the widely used metric AUC (Area Under the ROC curve):
예측 순위는 널리 사용되는 메트릭 AUC (ROC 곡선 아래 영역)를 사용하여 Tu에서 평가됩니다.

![(9)](./image/(9).PNG)

where the set of evaluation pairs for user u is defined as
여기서 사용자 u에 대한 평가 쌍 세트는 다음과 같이 정의됩니다.

![(10)](./image/(10).PNG)

and δ(b) is an indicator function that returns 1 iff b is true.
δ (b)는 b가 참이면 1을 반환하는 표시기 함수입니다.


In all cases we report the performance on the test set T for the hyperparameters that led to the best performance on the validation set V.
모든 경우에 검증 세트 V에서 최고의 성능을 이끌어 낸 하이퍼 파라미터에 대한 테스트 세트 T의 성능을보고합니다.


#### Baselines 

Matrix Factorization (MF) methods are known to have stateof-the-art performance for implicit feedback datasets. 
MF (Matrix Factorization) 방법은 암시 적 피드백 데이터 세트에 대한 최신 성능을 제공하는 것으로 알려져 있습니다.


Since there are no comparable visual-aware MF methods, we mainly compare against state-of-the-art MF models, in addition to a recently proposed content-based method.
비교 가능한 시각적 인식 MF 방법이 없기 때문에 우리는 최근 제안 된 콘텐츠 기반 방법 외에도 주로 최첨단 MF 모델과 비교합니다.

• Random (RAND): This baseline ranks items randomly for all users.
• Most Popular (MP): This baseline ranks items according to their popularity and is non-personalized.
• MM-MF: A pairwise MF model from Gantner et al.(2011), which is optimized for a hinge ranking loss on xuij and trained using SGA as in BPR-MF.
• BPR-MF: This pairwise method was introduced by Rendle et al. (2009) and is the state-of-the-art of personalized ranking for implicit feedback datasets.
• 무작위 (RAND) :이 기준은 모든 사용자에 대해 항목의 순위를 무작위로 지정합니다.
• 최고 인기 (MP) :이 기준은 인기도에 따라 항목의 순위를 매기 며 개인화되지 않습니다.
• MM-MF : Gantner 등 (2011)의 쌍별 MF 모델로 xuij의 힌지 순위 손실에 최적화되고 BPR-MF에서와 같이 SGA를 사용하여 훈련되었습니다.
• BPR-MF :이 쌍 방식 방법은 Rendle et al. (2009)는 암시 적 피드백 데이터 세트에 대한 최신 개인화 된 순위입니다.

We also include a ‘content-based’ baseline for comparison against another method which makes use of visual data, though which differs in terms of problem setting and data (it does not make use of feedback but rather graphs encoding relationships between items as input):
또한 시각적 데이터를 사용하는 다른 방법과 비교하기 위해 '콘텐츠 기반'기준선을 포함하지만 문제 설정 및 데이터 측면에서 다릅니다 (피드백을 사용하지 않고 항목 간의 관계를 입력으로 인코딩하는 그래프) :


• Image-based Recommendation (IBR): Introduced by McAuley et al. (2015), it learns a visual space and retrieves stylistically similar items to a query image. 
• IBR (이미지 기반 권장 사항) : McAuley et al. (2015), 시각적 공간을 학습하고 쿼리 이미지와 스타일 적으로 유사한 항목을 검색합니다.


Prediction is then performed by nearest-neighbor search in the learned visual space.
그런 다음 학습 된 시각 공간에서 가장 가까운 이웃 검색에 의해 예측이 수행됩니다.


Though significantly different from the pairwise methods considered above, for comparison we also compared against a point-wise method, WRMF (Hu, Koren, and Volinsky, 2008).
위에서 고려한 쌍별 방법과는 크게 다르지만 비교를 위해 점별 방법 인 WRMF 와도 비교했습니다 (Hu, Koren 및 Volinsky, 2008).


Most baselines are from MyMediaLite (Gantner et al.,2011). 
대부분의 기준선은 MyMediaLite에서 가져온 것입니다 (Gantner et al., 2011).


For fair comparison, we use the same total number of dimensions for all MF based methods. 
공정한 비교를 위해 모든 MF 기반 방법에 대해 동일한 총 차원 수를 사용합니다.


In our model, visual and non-visual dimensions are fixed to a fifty-fifty split for simplicity, though further tuning may yield better performance. 
우리 모델에서 시각적 및 비 시각적 차원은 단순성을 위해 50 분할로 고정되지만 추가 조정은 더 나은 성능을 제공 할 수 있습니다.


All experiments were performed on a standard desktop machine with 4 physical cores and 32GB main memory.
모든 실험은 4 개의 물리적 코어와 32GB 메인 메모리가있는 표준 데스크톱 컴퓨터에서 수행되었습니다.


#### Reproducibility. 
재현성.

All hyperparameters are tuned to perform the best on the validation set. 
모든 하이퍼 파라미터는 검증 세트에서 최상의 성능을 발휘하도록 조정됩니다.


On Amazon, regularization hyperparamter λΘ = 10 works the best for BPR-MF, MM-MF and VBPR in most cases. 
Amazon에서 정규화 하이퍼 파라미터 λΘ = 10은 대부분의 경우 BPR-MF, MM-MF 및 VBPR에 가장 적합합니다.


While on Tradesy.com, λΘ = 0.1 is set for BPR-MF and VBPR and λΘ = 1 for MM-MF. 
Tradesy.com에서 BPR-MF 및 VBPR에 대해 λΘ = 0.1이 설정되고 MM-MF에 대해 λΘ = 1이 설정됩니다.


λE is always set to 0 for VBPR. 
λE는 VBPR의 경우 항상 0으로 설정됩니다.


For IBR, the rank of the Mahalanobis transform is set to 100, which is reported to perform very well on Amazon data. 
IBR의 경우 Mahalanobis 변환 순위는 100으로 설정되어 Amazon 데이터에서 매우 잘 수행되는 것으로보고됩니다.


All of our code and datasets shall be made available at publication time so that our experimental evaluation is completely reproducible.
우리의 모든 코드와 데이터 세트는 우리의 실험 평가가 완전히 재현 될 수 있도록 공개 시점에 제공되어야합니다.


#### Performance 
![T3](./image/T3.PNG)
Results in terms of the average AUC on different datasets are shown in Table 3 (all with 20 total factors). 
서로 다른 데이터 세트의 평균 AUC 측면에서 결과가 표 3에 나와 있습니다 (모두 20 개 요소 포함).


For each dataset, we report the average AUC on the full test set T (denoted by ‘All Items’), as well as a subset of T which only consists of items that had fewer than five positive feedback instances in the training set (i.e., cold start). 
각 데이터 세트에 대해 전체 테스트 세트 T ( '모든 항목'으로 표시)에 대한 평균 AUC와 훈련 세트에서 긍정적 인 피드백 인스턴스가 5 개 미만인 항목으로 만 구성된 T의 하위 집합 (예 : , 콜드 스타트).


These cold start items account for around 60% of the test set for the two Amazon datasets, and 80% for Tradesy.com; this means for such sparse real-world datasets, a model must address the inherent cold start nature of the problem and recommend items accurately in order to achieve acceptable performance. 
이러한 콜드 스타트 ​​항목은 두 개의 Amazon 데이터 세트에 대한 테스트 세트의 약 60 %를 차지하고 Tradesy.com에 대해 80 %를 차지합니다. 이는 이러한 희소 한 실제 데이터 세트의 경우 모델이 문제의 고유 한 콜드 스타트 ​​특성을 해결하고 수용 가능한 성능을 달성하기 위해 항목을 정확하게 추천해야 함을 의미합니다.


The main findings from Table 3 are summarized as follows:
표 3의 주요 결과는 다음과 같이 요약됩니다.

1. Building on top of BPR-MF, VBPR on average improves on BPR-MF by over 12% for all items, and more than 28% for cold start. 
1. BPR-MF 위에 구축 된 VBPR은 평균적으로 모든 항목에서 BPR-MF를 12 % 이상, 콜드 스타트에서 28 % 이상 향상시킵니다.


This demonstrates the significant benefits of incorporating CNN features into our ranking task.
이는 순위 작업에 CNN 기능을 통합 할 때 얻을 수있는 중요한 이점을 보여줍니다.


2. As expected, IBR outperforms BPR-MF & MM-MF in cold start settings where pure MF methods have trouble learning meaningful factors. Moreover, IBR loses to MF methods for warm start since it is not trained on historical user feedback.
2. 예상대로 IBR은 순수 MF 방법이 의미있는 요소를 학습하는 데 어려움이있는 콜드 스타트 ​​설정에서 BPR-MF 및 MM-MF를 능가합니다. 더욱이 IBR은 과거 사용자 피드백에 대한 교육을받지 않았기 때문에 웜 스타트를위한 MF 방법을 잃었습니다.


3. By combining the strengths of both MF and content-based methods, VBPR outperforms all baselines in most cases.
3. MF 및 콘텐츠 기반 방법의 장점을 결합함으로써 VBPR은 대부분의 경우 모든 기준을 능가합니다.


4. Our method exhibits particularly large improvements on Tradesy.com, since it is an inherently cold start dataset due to the ‘one-off’ nature of trades.
4. 트레이드의 '일회성'특성으로 인해 본질적으로 콜드 스타트 ​​데이터 셋이기 때문에 우리의 방법은 Tradesy.com에서 특히 크게 개선되었습니다.


5. Visual features show greater benefits on clothing than cellphone datasets. Presumably this is because visual factors play a smaller (though still significant) role when selecting cellphones as compared to clothing.
5. 시각적 기능은 휴대폰 데이터 세트보다 의류에 더 큰 이점을 보여줍니다. 아마도 이것은 의류에 비해 휴대폰을 선택할 때 시각적 요소가 더 작은 (여전히 중요하지만) 역할을하기 때문일 것입니다.


6. Popularity-based methods are particularly ineffective here, as cold items are inherently ‘unpopular’.
6. 콜드 아이템은 본질적으로 '인기없는'것이기 때문에 인기 기반 방법은 특히 효과적이지 않습니다.


Finally, we found that pairwise methods indeed outperform point-wise methods (WRMF in our case) on our datasets, consistent with our analysis in Related Work. 
마지막으로, 우리는 데이터 세트에서 쌍별 방법이 실제로 포인트 별 방법 (우리의 경우 WRMF)을 능가한다는 것을 발견했습니다. 이는 Related Work의 분석과 일치합니다.


We found that on average, VBPR beats WRMF by 14.3% for all items and 20.3% for cold start items.
평균적으로 VBPR은 모든 항목에서 WRMF를 14.3 %, 콜드 시작 항목에서 20.3 % 앞섰습니다.

#### Sensitivity. 
![Fig2](./image/Fig2.PNG)
As shown in Figure 2, MM-MF, BPR-MF, and VBPR perform better as the number of factors increases, which demonstrates the ability of pairwise methods to avoid overfitting. 
그림 2에서 볼 수 있듯이 MM-MF, BPR-MF 및 VBPR은 요인 수가 증가할수록 더 나은 성능을 발휘하며, 이는 과적 합을 방지하는 쌍별 방법의 능력을 보여줍니다.


Results for other Amazon categories are similar and suppressed for brevity.
다른 아마존 카테고리의 결과는 유사하며 간결성을 위해 표시되지 않습니다.


#### Training Efficiency. 
훈련 효율성.

![Fig3](./image/Fig3.PNG)
In Figure 3 we demonstrate the AUC (on the test set) with increasing training iterations. 
그림 3에서는 훈련 반복이 증가하는 AUC (테스트 세트에서)를 보여줍니다.


Generally speaking, our proposed model takes longer to converge than MM-MF and BPR-MF, though still requires only around 3.5 hours to train to convergence on our largest dataset (Women’s Clothing).
일반적으로 제안 된 모델은 MM-MF 및 BPR-MF보다 수렴하는 데 시간이 더 오래 걸리지 만 가장 큰 데이터 세트 (Women 's Clothing)에서 수렴하는 데는 약 3.5 시간 만 필요합니다.


#### Visualizing Visual Space 

VBPR maps items to a low-dimensional ‘visual space,’ such that items with similar styles (in terms of how users evaluate them) are mapped to nearby locations. 
VBPR은 항목을 저 차원의 '시각적 공간'에 매핑하여 유사한 스타일 (사용자가 평가하는 방식)을 가진 항목을 가까운 위치에 매핑합니다.

![Fig4](./image/Fig4.PNG)
We visualize this space (for Women’s Clothing) in Figure 4. 
이 공간 (여성 의류 용)을 그림 4에 시각화합니다.


We make the following two observations: 
우리는 다음 두 가지 관찰을합니다.

(1) although our visual features are extracted from a CNN pre-trained on a different dataset, by using the embedding we are nevertheless able to learn a ‘visual’ transition (loosely) across different subcategories, which confirms the expressive power of the extracted features; and 
(1) 우리의 시각적 특징은 다른 데이터 세트에 대해 사전 훈련 된 CNN에서 추출되었지만 임베딩을 사용하여 그럼에도 불구하고 다른 하위 범주에서 '시각적'전환을 (느슨하게) 학습 할 수 있으며, 이는 추출 된 항목의 표현력을 확인합니다. 풍모; 과


(2) VBPR not only helps learn the hidden taxonomy, but also more importantly discovers the most relevant underlying visual dimensions and maps items and users into the uncovered space.
(2) VBPR은 숨겨진 분류법을 배우는 데 도움이 될뿐만 아니라 가장 관련성이 높은 기본 시각적 차원을 발견하고 항목과 사용자를 숨겨진 공간으로 매핑합니다.

---

### Conclusion & Future Work 

Visual decision factors influence many of the choices people make, from the clothes they wear to their interactions with each other.  
시각적 의사 결정 요소는 사람들이 입는 옷부터 서로 상호 작용에 이르기까지 많은 선택에 영향을 미칩니다.


In this paper, we investigated the usefulness of visual features for personalized ranking tasks on implicit feedback datasets. 
이 백서에서는 암시 적 피드백 데이터 집합에 대한 개인화 된 순위 지정 작업에 대한 시각적 기능의 유용성을 조사했습니다.


We proposed a scalable method that incorporates visual features extracted from product images into Matrix Factorization, in order to uncover the ‘visual dimensions’ that most influence people’s behavior. 
우리는 사람들의 행동에 가장 큰 영향을 미치는 '시각적 차원'을 밝히기 위해 제품 이미지에서 추출한 시각적 특징을 Matrix Factorization에 통합하는 확장 가능한 방법을 제안했습니다.


Our model is trained with Bayesian Personalized Ranking (BPR) using stochastic gradient ascent. 
우리의 모델은 확률 적 경사 상승을 사용하여 베이지안 개인화 순위 (BPR)로 훈련되었습니다.


Experimental results on multiple large real-world datasets demonstrate that we can significantly outperform state-of-the-art ranking techniques and alleviate cold start issues.
여러 개의 대규모 실제 데이터 세트에 대한 실험 결과는 최첨단 순위 기술을 크게 능가하고 콜드 스타트 ​​문제를 완화 할 수 있음을 보여줍니다.


As part of future work, we will further extend our model with temporal dynamics to account for the drifting of fashion tastes over time. 
향후 작업의 일환으로 시간에 따른 패션 취향의 표류를 설명하기 위해 시간 역학으로 모델을 더욱 확장 할 것입니다.


Additionally, we are also interested in investigating the efficacy of our proposed method in the setting of explicit feedback.
또한 우리는 명시 적 피드백 설정에서 제안 된 방법의 효과를 조사하는 데 관심이 있습니다.
    

---