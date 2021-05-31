2001_IB_CFR [Item-Based Collaborative Filtering Recommendation]

![main](./image/main.PNG)

---

ABSTRACT
Recommender systems apply knowledge discovery techniques to the problem of making personalized recommendations for information, products or services during a live interaction. 

These systems, especially the k-nearest neighbor collaborative filtering based ones, are achieving widespread success on the Web. 

The tremendous growth in the amount of available information and the number of visitors to Web sites in recent years poses some key challenges for recommender systems. 

These are: producing high quality recommendations, performing many recommendations per second for millions of users and items and achieving high coverage in the face of data sparsity. 

In traditional collaborative filtering systems the amount of work increases with the number of participants in the system. 

New recommender system technologies are needed that can quickly produce high quality recommendations, even for very large-scale problems. 

To address these issues we have explored item-based collaborative filtering techniques. 

Item-based techniques first analyze the user-item matrix to identify relationships between different items, and then use these relationships to indirectly compute recommendations for users.


요약
Recommender 시스템은 실시간 상호 작용 중에 정보, 제품 또는 서비스에 대한 개인화 된 권장 사항을 만드는 문제에 지식 검색 기술을 적용합니다.

이러한 시스템, 특히 k- 최근 접 이웃 협업 필터링 기반 시스템은 웹에서 광범위한 성공을 거두고 있습니다.

최근 몇 년 동안 사용 가능한 정보의 양과 웹 사이트 방문자 수가 엄청나게 증가함에 따라 추천 시스템에 몇 가지 주요 문제가 발생합니다.

여기에는 고품질 권장 사항 생성, 수백만 명의 사용자 및 항목에 대해 초당 많은 권장 사항 수행, 데이터 희소성에 직면 한 높은 적용 범위 달성 등이 있습니다.

기존의 협업 필터링 시스템에서 작업량은 시스템 참여자 수에 따라 증가합니다.

매우 큰 규모의 문제에서도 고품질 추천을 신속하게 생성 할 수있는 새로운 추천 시스템 기술이 필요합니다.

이러한 문제를 해결하기 위해 항목 기반 협업 필터링 기술을 탐색했습니다.

항목 기반 기술은 먼저 사용자 항목 매트릭스를 분석하여 서로 다른 항목 간의 관계를 식별 한 다음 이러한 관계를 사용하여 사용자에 대한 권장 사항을 간접적으로 계산합니다.





In this paper we analyze different item-based recommendation generation algorithms. 

We look into different techniques for computing item-item similarities (e.g., item-item correlation vs. cosine similarities between item vectors) and different techniques for obtaining recommendations from them (e.g., weighted sum vs. regression model). 

Finally, we experimentally evaluate our results and compare them to the basic k-nearest neighbor approach. 

Our experiments suggest that item-based algorithms provide dramatically better performance than user-based algorithms, while at the same time providing better quality than the best available userbased algorithms.

이 논문에서는 다양한 항목 기반 추천 생성 알고리즘을 분석합니다.

항목-항목 유사성을 계산하기위한 다양한 기술 (예 : 항목-항목 상관 대 항목 벡터 간의 코사인 유사성)과 권장 사항을 얻기위한 다양한 기술 (예 : 가중치 합계 대 회귀 모델)을 조사합니다.

마지막으로 결과를 실험적으로 평가하고 기본 k- 최근 접 이웃 접근법과 비교합니다.

우리의 실험에 따르면 항목 기반 알고리즘은 사용자 기반 알고리즘보다 훨씬 더 나은 성능을 제공하는 동시에 사용 가능한 최상의 사용자 기반 알고리즘보다 더 나은 품질을 제공합니다.








1. INTRODUCTION

The amount of information in the world is increasing far more quickly than our ability to process it. 

All of us have known the feeling of being overwhelmed by the number of new books, journal articles, and conference proceedings coming out each year. 

Technology has dramatically reduced the barriers to publishing and distributing information. 

Now it is time to create the technologies that can help us sift through all the available information to find that which is most valuable to us.

One of the most promising such technologies is col laborative filtering [19, 27, 14, 16]. 

Collaborative filtering works by building a database of preferences for items by users. 

A new user, Neo, is matched against the database to discover neighbors, which are other users who have historically had similar taste to Neo. 

Items that the neighbors like are then recommended to Neo, as he will probably also like them. 

Collaborative filtering has been very successful in both research and practice, and in both information filtering applications and E-commerce applications. 

However, there remain important research questions in overcoming two fundamental challenges for collaborative filtering recommender systems. 

The first challenge is to improve the scalability of the collaborative filtering algorithms. 

These algorithms are able to search tens of thousands of potential neighbors in real-time, but the demands of modern systems are to search tens of millions of potential neighbors. 

Further, existing algorithms have performance problems with individual users for whom the site has large amounts of information. 

For instance, if a site is using browsing patterns as indications of content preference, it may have thousands of data points for its most frequent visitors. 

These "long user rows" slow down the number of neighbors that can be searched per second, further reducing scalability. 

The second challenge is to improve the quality of the recommendations for the users. 

Users need recommendations they can trust to help them find items they will like. 

세계의 정보량은 우리가 처리하는 능력보다 훨씬 더 빠르게 증가하고 있습니다.

우리 모두는 매년 나오는 새로운 책, 저널 기사 및 회의 절차에 압도당하는 느낌을 알고 있습니다.

기술은 정보 게시 및 배포의 장벽을 극적으로 줄였습니다.

이제 우리에게 가장 가치있는 정보를 찾기 위해 사용 가능한 모든 정보를 살펴볼 수있는 기술을 만들 때입니다.

이러한 가장 유망한 기술 중 하나는 협업 필터링입니다 [19, 27, 14, 16].

협업 필터링은 사용자가 항목에 대한 선호도 데이터베이스를 구축하여 작동합니다.

새로운 사용자 인 Neo는 이전에 Neo와 비슷한 취향을 가진 다른 사용자 인 이웃을 찾기 위해 데이터베이스와 대조됩니다.

이웃 사람들이 좋아하는 아이템은 아마 좋아할 것이므로 Neo에게 추천됩니다.

협업 필터링은 연구와 실습, 정보 필터링 애플리케이션과 전자 상거래 애플리케이션 모두에서 매우 성공적이었습니다.

그러나 협업 필터링 추천 시스템의 두 가지 근본적인 문제를 극복하는 데있어 중요한 연구 질문이 남아 있습니다.

첫 번째 과제는 협업 필터링 알고리즘의 확장 성을 향상시키는 것입니다.

이러한 알고리즘은 수만 개의 잠재적 인 이웃을 실시간으로 검색 할 수 있지만 현대 시스템의 요구 사항은 수천만 개의 잠재적 인 이웃을 검색하는 것입니다.

또한 기존 알고리즘은 사이트에 많은 양의 정보가있는 개별 사용자에게 성능 문제가 있습니다.

예를 들어, 사이트에서 검색 패턴을 콘텐츠 선호도의 지표로 사용하는 경우 가장 빈번한 방문자에 대한 수천 개의 데이터 포인트가있을 수 있습니다.

이러한 "긴 사용자 행"은 초당 검색 할 수있는 인접 항목 수를 줄여 확장 성을 더욱 감소시킵니다.

두 번째 과제는 사용자를위한 권장 사항의 품질을 개선하는 것입니다.

사용자는 좋아할 항목을 찾는 데 도움이되도록 신뢰할 수있는 권장 사항이 필요합니다.







Users will "vote with their feet" by refusing to use recommender systems that are not consistently accurate for them.

In some ways these two challenges are in conflict, since the less time an algorithm spends searching for neighbors, the more scalable it will be, and the worse its quality. 

For this reason, it is important to treat the two challenges simultaneously so the solutions discovered are both useful and practical.

In this paper, we address these issues of recommender systems by applying a different approach-item-based algorithm. 

The bottleneck in conventional collaborative filtering algorithms is the search for neighbors among a large user population of potential neighbors [12]. 

Item-based algorithms avoid this bottleneck by exploring the relationships between items first, rather than the relationships between users. 

Recommendations for users are computed by finding items that are similar to other items the user has liked. 

Because the relationships between items are relatively static, item-based algorithms may be able to provide the same quality as the user-based algorithms with less online computation.

1.1 Related Work
In this section we briefly present some of the research literature related to collaborative filtering, recommender systems, data mining and personalization.

Tapestry [10] is one of the earliest implementations of collaborative filtering-based recommender systems. 

This system relied on the explicit opinions of people from a close-knit community, such as an oce workgroup. 

However, recommender system for large communities cannot depend on each person knowing the others. 

사용자는 지속적으로 정확하지 않은 추천 시스템 사용을 거부함으로써 "발로 투표"합니다.

알고리즘이 이웃을 검색하는 데 소요되는 시간이 적을수록 확장 성이 높아지고 품질이 저하되기 때문에 어떤면에서는이 두 가지 문제가 충돌합니다.

따라서 발견 된 솔루션이 유용하고 실용적 이도록 두 가지 문제를 동시에 처리하는 것이 중요합니다.

이 논문에서 우리는 다른 접근법-항목 기반 알고리즘을 적용하여 추천 시스템의 이러한 문제를 해결합니다.

기존 협업 필터링 알고리즘의 병목 현상은 잠재적 인 이웃의 대규모 사용자 집단 중에서 이웃을 검색하는 것입니다 [12].

항목 기반 알고리즘은 사용자 간의 관계가 아닌 항목 간의 관계를 먼저 탐색하여 이러한 병목 현상을 방지합니다.

사용자에 대한 추천은 사용자가 좋아 한 다른 항목과 유사한 항목을 찾아 계산됩니다.

항목 간의 관계가 비교적 정적이기 때문에 항목 기반 알고리즘은 적은 온라인 계산으로 사용자 기반 알고리즘과 동일한 품질을 제공 할 수 있습니다.

1.1 관련 작업
이 섹션에서는 협업 필터링, 추천 시스템, 데이터 마이닝 및 개인화와 관련된 몇 가지 연구 문헌을 간략하게 소개합니다.

Tapestry [10]는 협업 필터링 기반 추천 시스템의 초기 구현 중 하나입니다.

이 시스템은 사무실 작업 그룹과 같은 긴밀한 커뮤니티의 사람들의 명시적인 의견에 의존했습니다.

그러나 대규모 커뮤니티를위한 추천 시스템은 다른 사람을 아는 사람 각자에게 의존 할 수 없습니다.






Later, several ratings-based automated recommender systems were developed. 

The GroupLens research system [19, 16] provides a pseudonymous collaborative filtering solution for Usenet news and movies. 

Ringo [27] and Video Recommender [14] are email and webbased systems that generate recommendations on music and movies, respectively. 

A special issue of Communications of the ACM [20] presents a number of different recommender systems.

Other technologies have also been applied to recommender systems, including Bayesian networks, clustering, and Horting. 

Bayesian networks create a model based on a training set with a decision tree at each node and edges representing user information. 

The model can be built off-line over a matter of hours or days. 

The resulting model is very small, very fast, and essentially as accurate as nearest neighbor methods [6]. 

Bayesian networks may prove practical for environments in which knowledge of user preferences changes slowly with respect to the time needed to build the model but are not suitable for environments in which user preference models must be updated rapidly or frequently. 

Clustering techniques work by identifying groups of users who appear to have similar preferences. 

Once the clusters are created, predictions for an individual can be made by averaging the opinions of the other users in that cluster. 

Some clustering techniques represent each user with partial participation in several clusters. 

The prediction is then an average across the clusters, weighted by degree of participation.

Clustering techniques usually produce less-personal recommendations than other methods, and in some cases, the clusters have worse accuracy than nearest neighbor algorithms [6]. 

Once the clustering is complete, however, performance can be very good, since the size of the group that must be analyzed is much smaller. 

Clustering techniques can also be applied as a "first step" for shrinking the candidate set in a nearest neighbor algorithm or for distributing nearestneighbor computation across several recommender engines. 

나중에 여러 등급 기반 자동 추천 시스템이 개발되었습니다.

GroupLens 연구 시스템 [19, 16]은 Usenet 뉴스 및 영화에 대한 익명의 협업 필터링 솔루션을 제공합니다.

Ringo [27]와 Video Recommender [14]는 각각 음악과 영화에 대한 추천을 생성하는 이메일 및 웹 기반 시스템입니다.

Communications of the ACM [20]의 특별 호는 다양한 추천 시스템을 제시합니다.

Bayesian 네트워크, 클러스터링 및 Horting을 포함한 다른 기술도 추천 시스템에 적용되었습니다.

베이지안 네트워크는 각 노드의 의사 결정 트리와 사용자 정보를 나타내는 에지가있는 훈련 세트를 기반으로 모델을 만듭니다.

이 모델은 몇 시간 또는 며칠 동안 오프라인으로 구축 할 수 있습니다.

결과 모델은 매우 작고 빠르며 기본적으로 최근 접 이웃 방법만큼 정확합니다 [6].

베이지안 네트워크는 모델을 구축하는 데 필요한 시간과 관련하여 사용자 선호도에 대한 지식이 느리게 변하는 환경에서 실용적 일 수 있지만 사용자 선호도 모델을 빠르게 또는 자주 업데이트해야하는 환경에는 적합하지 않습니다.

클러스터링 기술은 비슷한 선호도를 가진 것으로 보이는 사용자 그룹을 식별하여 작동합니다.

클러스터가 생성되면 해당 클러스터에있는 다른 사용자의 의견을 평균하여 개인에 대한 예측을 할 수 있습니다.

일부 클러스터링 기술은 여러 클러스터에 부분적으로 참여하는 각 사용자를 나타냅니다.

그런 다음 예측은 참여 정도에 따라 가중치가 부여 된 클러스터 전체의 평균입니다.

클러스터링 기술은 일반적으로 다른 방법보다 덜 개인적인 권장 사항을 생성하며 경우에 따라 클러스터는 최근 접 이웃 알고리즘보다 정확도가 떨어집니다 [6].

그러나 클러스터링이 완료되면 분석해야하는 그룹의 크기가 훨씬 더 작기 때문에 성능이 매우 좋을 수 있습니다.

클러스터링 기술은 가장 가까운 이웃 알고리즘에서 후보 세트를 축소하거나 여러 추천 엔진에 가장 가까운 이웃 계산을 배포하기위한 "첫 번째 단계"로 적용될 수도 있습니다.






While dividing the population into clusters may hurt the accuracy or recommendations to users near the fringes of their assigned cluster, pre-clustering may be a worthwhile trade-off between accuracy and throughput.

Horting is a graph-based technique in which nodes are users, and edges between nodes indicate degree of similarity between two users [1]. 

Predictions are produced by walking the graph to nearby nodes and combining the opinions of the nearby users. 

Horting differs from nearest neighbor as the graph may be walked through other users who have not rated the item in question, thus exploring transitive relationships that nearest neighbor algorithms do not consider.

In one study using synthetic data, Horting produced better predictions than a nearest neighbor algorithm [1].

Schafer et al., [26] present a detailed taxonomy and examples of recommender systems used in E-commerce and how they can provide one-to-one personalization and at the same can capture customer loyalty. 

Although these systems have been successful in the past, their widespread use has exposed some of their limitations such as the problems of sparsity in the data set, problems associated with high dimensionality and so on. 

Sparsity problem in recommender system has been addressed in [23, 11]. 

The problems associated with high dimensionality in recommender systems have been discussed in [4], and application of dimensionality reduction techniques to address these issues has been investigated in [24].

Our work explores the extent to which item-based recommenders, a new class of recommender algorithms, are able to solve these problems.

모집단을 클러스터로 나누면 할당 된 클러스터 주변에있는 사용자에 대한 정확성이나 권장 사항이 손상 될 수 있지만 사전 클러스터링은 정확도와 처리량 사이의 적절한 절충안이 될 수 있습니다.

Horting은 노드가 사용자이고 노드 사이의 경계는 두 사용자 간의 유사도를 나타내는 그래프 기반 기술입니다 [1].

그래프를 주변 노드로 이동하고 주변 사용자의 의견을 결합하여 예측을 생성합니다.

Horting은 해당 항목을 평가하지 않은 다른 사용자를 통해 그래프가 표시 될 수 있으므로 가장 가까운 이웃과 다릅니다. 따라서 가장 가까운 이웃 알고리즘이 고려하지 않는 전 이적 관계를 탐색 할 수 있습니다.

합성 데이터를 사용한 한 연구에서 Horting은 최근 접 이웃 알고리즘보다 더 나은 예측을 내놓았습니다 [1].

Schafer et al., [26]은 전자 상거래에 사용되는 추천 시스템의 자세한 분류법과 예를 제시하고 이들이 일대일 개인화를 제공하고 동시에 고객 충성도를 포착 할 수있는 방법을 제시합니다.

이러한 시스템은 과거에는 성공적 이었지만 널리 사용되면서 데이터 세트의 희소성 문제, 높은 차원과 관련된 문제 등과 같은 몇 가지 한계가 드러났습니다.

추천 시스템의 희소성 문제는 [23, 11]에서 해결되었습니다.

추천 시스템의 고차원 성과 관련된 문제는 [4]에서 논의되었으며, 이러한 문제를 해결하기위한 차원 축소 기술의 적용은 [24]에서 조사되었습니다.

우리의 연구는 새로운 종류의 추천 알고리즘 인 항목 기반 추천자가 이러한 문제를 해결할 수있는 정도를 탐구합니다.




1.2 Contributions
This paper has three primary research contributions:

1. Analysis of the item-based prediction algorithms and identification of different ways to implement its subtasks.

2. Formulation of a precomputed model of item similarity to increase the online scalability of item-based recommendations.

3. An experimental comparison of the quality of several different item-based algorithms to the classic user-based (nearest neighbor) algorithms.

1.2 기여
이 백서에는 세 가지 주요 연구 공헌이 있습니다.

1. 항목 기반 예측 알고리즘의 분석 및 하위 작업을 구현하는 다양한 방법 식별.

2. 항목 기반 권장 사항의 온라인 확장 성을 높이기 위해 항목 유사성의 사전 계산 된 모델을 공식화합니다.

3. 기존의 사용자 기반 (가장 가까운 이웃) 알고리즘에 대한 여러 가지 항목 기반 알고리즘의 품질을 실험적으로 비교합니다.








1.3 Organization
The rest of the paper is organized as follows. 

The next section provides a brief background in collaborative filtering algorithms. 

We first formally describe the collaborative filtering process and then discuss its two variants memorybased and model-based approaches. 

We then present some challenges associated with the memory-based approach. 

In section 3, we present the item-based approach and describe different sub-tasks of the algorithm in detail. 

Section 4 describes our experimental work. 

It provides details of our data sets, evaluation metrics, methodology and results of different experiments and discussion of the results. 

The final section provides some concluding remarks and directions for future research.

1.3 조직
나머지 논문은 다음과 같이 구성됩니다.

다음 섹션에서는 협업 필터링 알고리즘에 대한 간략한 배경 정보를 제공합니다.

먼저 협업 필터링 프로세스를 공식적으로 설명한 다음 두 가지 변형 메모리 기반 및 모델 기반 접근 방식에 대해 논의합니다.

그런 다음 메모리 기반 접근 방식과 관련된 몇 가지 문제를 제시합니다.

섹션 3에서는 항목 기반 접근 방식을 제시하고 알고리즘의 여러 하위 작업을 자세히 설명합니다.

섹션 4는 우리의 실험 작업을 설명합니다.

데이터 세트, 평가 지표, 방법론 및 다양한 실험의 결과 및 결과에 대한 토론에 대한 세부 정보를 제공합니다.

마지막 섹션에서는 향후 연구를위한 몇 가지 결론 및 방향을 제공합니다. 





2. COLLABORATIVE FILTERING BASED RECOMMENDER SYSTEMS

Recommender systems systems apply data analysis techniques to the problem of helping users find the items they would like to purchase at E-Commerce sites by producing a predicted likeliness score or a list of top{N recommended items for a given user. 

Item recommendations can be made using different methods. 

Recommendations can be based on demographics of the users, overall top selling items, or past buying habit of users as a predictor of future items.

Collaborative Filtering (CF) [19, 27] is the most successful recommendation technique to date. 

The basic idea of CF-based algorithms is to provide item recommendations or predictions based on the opinions of other like-minded users. 

The opinions of users can be obtained explicitly from the users or by using some implicit measures.

2.0.1 Overview of the Collaborative Filtering Process

The goal of a collaborative filtering algorithm is to suggest new items or to predict the utility of a certain item for a particular user based on the user's previous likings and the opinions of other like-minded users. 

In a typical CF scenario, there is a list of m users U = {u1, u2 ;::: ;um} and a list of n items I = {i1; i2 ;::: ;in}. 

Each user ui has a list of items Iui , which the user has expressed his/her opinions about. 

Opinions can be explicitly given by the user as a rating score, generally within a certain numerical scale, or can be implicitly derived from purchase records, by analyzing timing logs, by mining web hyperlinks and so on [28, 16].

Note that Iui공식I and it is possible for Iui, to be a null-set. 

2. 협업 필터링 기반 추천 시스템

Recommender 시스템 시스템은 사용자가 전자 상거래 사이트에서 구매하고 싶은 항목을 찾도록 돕는 문제에 데이터 분석 기법을 적용하여 특정 사용자에 대해 예측 가능성 점수 또는 상위 {N 개 권장 항목 목록을 생성합니다.

항목 추천은 다른 방법을 사용하여 만들 수 있습니다.

권장 사항은 사용자의 인구 통계, 전체적으로 가장 많이 팔린 항목 또는 미래 항목의 예측 변수로서 사용자의 과거 구매 습관을 기반으로 할 수 있습니다.

CF (Collaborative Filtering) [19, 27]는 현재까지 가장 성공적인 추천 기법입니다.

CF 기반 알고리즘의 기본 아이디어는 같은 생각을 가진 다른 사용자의 의견을 기반으로 항목 추천 또는 예측을 제공하는 것입니다.

사용자의 의견은 사용자로부터 명시 적으로 또는 일부 암시 적 조치를 사용하여 얻을 수 있습니다.

2.0.1 협업 필터링 프로세스 개요

협업 필터링 알고리즘의 목표는 사용자의 이전 선호도와 같은 생각을 가진 다른 사용자의 의견을 기반으로 특정 사용자에게 새로운 항목을 제안하거나 특정 항목의 유용성을 예측하는 것입니다.

일반적인 CF 시나리오에는 m 명의 사용자 목록 U = {u1, u2; :::; um} 및 n 개의 항목 목록 I = {i1; i2; :::; in}.

각 사용자 ui에는 사용자가 자신의 의견을 표현한 Iui 항목 목록이 있습니다.

의견은 일반적으로 특정 수치 척도 내에서 평점 점수로 사용자에 의해 명시 적으로 제공되거나, 타이밍 로그 분석, 웹 하이퍼 링크 마이닝 등을 통해 구매 기록에서 암시 적으로 도출 될 수 있습니다 [28, 16].

Iui 공식 I 및 Iui가 null 집합이 될 수 있습니다.




There exists a distinguished user ua 2 U called the active user for whom the task of a collaborative filtering algorithm is to find an item likeliness that can be of two forms. 

* Prediction is a numerical value, Pa;j , expressing the predicted likeliness of item ij 62 Iua for the active user ua. 

This predicted value is within the same scale (e.g., from 1 to 5) as the opinion values provided by ua. 

* Recommendation is a list of N items, Ir  I, that the active user will like the most. 

Note that the recommended list must be on items not already purchased by the active user, i.e., Ir집합Iua = . 

This interface of CF algorithms is also known as Top-N recommendation. 

Figure 1 shows the schematic diagram of the collaborative filtering process. 

CF algorithms represent the entire m x n user-item data as a ratings matrix, A. 

Each entry ai;j in A represents the preference score (ratings) of the ith user on the jth item. 

Each individual ratings is within a numerical scale and it can as well be 0 indicating that the user has not yet rated that item. 

Researchers have devised a number of collaborative filtering algorithms that can be divided into two main categories|Memory-based (user-based) and Model-based (item-based) algorithms [6]. 

In this section we provide a detailed analysis of CF-based recommender system algorithms.

Memory-based Collaborative Filtering Algorithms. 

Memory-based algorithms utilize the entire user-item database to generate a prediction. 

공동 필터링 알고리즘의 작업이 두 가지 형태가 될 수있는 항목 유사성을 찾는 것 인 활성 사용자라고하는 고유 사용자 ua 2 U가 있습니다.

* 예측은 활성 사용자 ua에 대한 항목 ij 62 Iua의 예측 가능성을 나타내는 숫자 값 Pa; j입니다.

이 예측 값은 ua에서 제공하는 의견 값과 동일한 척도 (예 : 1 ~ 5) 내에 있습니다.

* 추천 항목은 활성 사용자가 가장 좋아할 N 개 항목 Ir I의 목록입니다.

권장 목록은 활성 사용자가 아직 구매하지 않은 항목 (예 : Ir 집합 Iua =)에 있어야합니다.

이 CF 알고리즘 인터페이스는 Top-N 권장 사항이라고도합니다.

그림 1은 협업 필터링 프로세스의 개략도를 보여줍니다.

CF 알고리즘은 전체 m x n 사용자 항목 데이터를 등급 행렬 A로 나타냅니다.

A의 각 항목 ai; j는 j 번째 항목에 대한 i 번째 사용자의 선호도 점수 (등급)를 나타냅니다.

각 개별 등급은 숫자 척도 내에 있으며 사용자가 아직 해당 항목을 평가하지 않았 음을 나타내는 0 일 수도 있습니다.

연구자들은 메모리 기반 (사용자 기반) 및 모델 기반 (항목 기반) 알고리즘 [6]의 두 가지 주요 범주로 나눌 수있는 여러 협업 필터링 알고리즘을 고안했습니다.

이 섹션에서는 CF 기반 추천 시스템 알고리즘에 대한 자세한 분석을 제공합니다.

메모리 기반 협업 필터링 알고리즘.

메모리 기반 알고리즘은 전체 사용자 항목 데이터베이스를 활용하여 예측을 생성합니다.







These systems employ statistical techniques to find a set of users, known as neighbors, that have a history of agreeing with the target user (i.e., they either rate different items similarly or they tend to buy similar set of items). 

Once a neighborhood of users is formed, these systems use different algorithms to combine the preferences of neighbors to produce a prediction or top-N recommendation for the active user. 

The techniques, also known as nearest-neighbor or user-based collaborative filtering, are more popular and widely used in practice.

Model-based Collaborative Filtering Algorithms. 

Model-based collaborative filtering algorithms provide item recommendation by first developing a model of user ratings. 

Algorithms in this category take a probabilistic approach and envision the collaborative filtering process as computing the expected value of a user prediction, given his/her ratings on other items. 

The model building process is performed by different machine learning algorithms such as Bayesian network, clustering, and rule-based approaches. 

The Bayesian network model [6] formulates a probabilistic model for collaborative filtering problem. 

Clustering model treats collaborative filtering as a classification problem [2, 6, 29] and works by clustering similar users in same class and estimating the probability that a particular user is in a particular class C, and from there computes the conditional probability of ratings. 

The rule-based approach applies association rule discovery algorithms to find association between co-purchased items and then generates item recommendation based on the strength of the association between items[25].

2.0.2 Challenges of User-based Collaborative Filtering Algorithms
User-based collaborative filtering systems have been very successful in past, but their widespread use has revealed some potential challenges such as: 

이러한 시스템은 통계 기법을 사용하여 대상 사용자와 동의 한 기록이있는 이웃이라고 알려진 사용자 집합을 찾습니다 (즉, 서로 다른 항목을 비슷하게 평가하거나 비슷한 항목 집합을 구입하는 경향이 있음).

사용자의 이웃이 형성되면 이러한 시스템은 서로 다른 알고리즘을 사용하여 이웃의 선호도를 결합하여 활성 사용자에 대한 예측 또는 top-N 추천을 생성합니다.

가장 가까운 이웃 또는 사용자 기반 협업 필터링이라고도하는이 기술은 더 널리 사용되고 실제로 널리 사용됩니다.

모델 기반 협업 필터링 알고리즘.

모델 기반 협업 필터링 알고리즘은 먼저 사용자 평가 모델을 개발하여 항목 추천을 제공합니다.

이 범주의 알고리즘은 확률 적 접근 방식을 취하고 다른 항목에 대한 평가를 고려하여 사용자 예측의 예상 값을 계산하는 협업 필터링 프로세스를 구상합니다.

모델 구축 프로세스는 베이지안 네트워크, 클러스터링 및 규칙 기반 접근 방식과 같은 다양한 기계 학습 알고리즘에 의해 수행됩니다.

베이지안 네트워크 모델 [6]은 협업 필터링 문제에 대한 확률 모델을 공식화합니다.

클러스터링 모델은 협업 필터링을 분류 문제로 취급하고 [2, 6, 29] 동일한 클래스의 유사한 사용자를 클러스터링하고 특정 사용자가 특정 클래스 C에 속할 확률을 추정하는 방식으로 작동하며 여기에서 등급의 조건부 확률을 계산합니다.

규칙 기반 접근법은 연관 규칙 발견 알고리즘을 적용하여 공동 구매 품목 간의 연관성을 찾은 다음 품목 간의 연관 강도에 따라 품목 추천을 생성합니다 [25].

2.0.2 사용자 기반 협업 필터링 알고리즘의 과제
사용자 기반 협업 필터링 시스템은 과거에 매우 성공적 이었지만 광범위하게 사용되면서 다음과 같은 몇 가지 잠재적 인 문제가 드러났습니다.





* Sparsity. 

In practice, many commercial recommender systems are used to evaluate large item sets (e.g., Amazon.com recommends books and CDnow.com recommends music albums). 

In these systems, even active users may have purchased well under 1% of the items (1% of 2 million books is 20; 000 books). 

Accordingly, a recommender system based on nearest neighbor algorithms may be unable to make any item recommendations for a particular user. 

As a result the accuracy of recommendations may be poor. 

* Scalability. 

Nearest neighbor algorithms require computation that grows with both the number of users and the number of items. 

With millions of users and items, a typical web-based recommender system running existing algorithms will suffer serious scalability problems.

The weakness of nearest neighbor algorithm for large, sparse databases led us to explore alternative recommender system algorithms. 

Our first approach attempted to bridge the sparsity by incorporating semi-intelligent filtering agents into the system [23, 11]. 

These agents evaluated and rated each item using syntactic features. 

By providing a dense ratings set, they helped alleviate coverage and improved quality. 

The filtering agent solution, however, did not address the fundamental problem of poor relationships among like-minded but sparse-rating users. 


* 확장 성.

최근 접 이웃 알고리즘에는 사용자 수와 항목 수에 따라 증가하는 계산이 필요합니다.

수백만 명의 사용자와 항목이있는 기존 알고리즘을 실행하는 일반적인 웹 기반 추천 시스템은 심각한 확장 성 문제를 겪게됩니다.

대규모 희소 데이터베이스에 대한 최근 접 이웃 알고리즘의 약점으로 인해 대체 추천 시스템 알고리즘을 탐색하게되었습니다.

우리의 첫 번째 접근 방식은 반 지능적인 필터링 에이전트를 시스템에 통합하여 희소성을 연결하려고 시도했습니다 [23, 11].

이러한 에이전트는 구문 기능을 사용하여 각 항목을 평가하고 평가했습니다.

밀도가 높은 등급 세트를 제공함으로써 적용 범위를 줄이고 품질을 개선하는 데 도움이되었습니다.

그러나 필터링 에이전트 솔루션은 생각이 비슷하지만 드문 사용자 간의 관계 불량이라는 근본적인 문제를 해결하지 못했습니다.






To explore that we took an algorithmic approach and used Latent Semantic Indexing (LSI) to capture the similarity between users and items in a reduced dimensional space [24, 25]. 

In this paper we look into another technique, the model-based approach, in addressing these challenges, especially the scalability challenge. 

The main idea here is to analyze the user-item representation matrix to identify relations between different items and then to use these relations to compute the prediction score for a given user-item pair. 

The intuition behind this approach is that a user would be interested in purchasing items that are similar to the items the user liked earlier and would tend to avoid items that are similar to the items the user didn't like earlier. 

These techniques don't require to identify the neighborhood of similar users when a recommendation is requested; as a result they tend to produce much faster recommendations. 

A number of different schemes have been proposed to compute the association between items ranging from probabilistic approach [6] to more traditional item-item correlations [15, 13]. 

We present a detailed analysis of our approach in the next section.

우리는 알고리즘 접근 방식을 취하고 LSI (Latent Semantic Indexing)를 사용하여 축소 된 차원 공간에서 사용자와 항목 간의 유사성을 포착했습니다 [24, 25].

이 백서에서는 이러한 문제, 특히 확장 성 문제를 해결하기위한 또 다른 기술인 모델 기반 접근 방식을 살펴 봅니다.

여기서 주요 아이디어는 사용자 항목 표현 행렬을 분석하여 서로 다른 항목 간의 관계를 식별 한 다음 이러한 관계를 사용하여 주어진 사용자 항목 쌍에 대한 예측 점수를 계산하는 것입니다.

이 접근 방식의 직관은 사용자가 이전에 좋아했던 항목과 유사한 항목을 구매하는 데 관심이 있고 사용자가 이전에 좋아하지 않았던 항목과 유사한 항목을 피하는 경향이 있다는 것입니다.

이러한 기술은 추천이 요청 될 때 유사한 사용자의 이웃을 식별 할 필요가 없습니다. 결과적으로 훨씬 더 빠른 권장 사항을 생성하는 경향이 있습니다.

확률 론적 접근 [6]에서보다 전통적인 항목-항목 상관 [15, 13]에 이르는 항목 간의 연관성을 계산하기 위해 여러 가지 다른 방식이 제안되었습니다.

다음 섹션에서 우리의 접근 방식에 대한 자세한 분석을 제시합니다.





3. ITEM-BASED COLLABORATIVE FILTERING ALGORITHM
In this section we study a class of item-based recommendation algorithms for producing predictions to users. 

Unlike the user-based collaborative filtering algorithm discussed in Section 2, the item-based approach looks into the set of items the target user has rated and computes how similar they are to the target item i and then selects k most similar items fi1; i2;::: ;ikg. 

At the same time their corresponding similarities {si1; si2;::: ;sik} are also computed.

Once the most similar items are found, the prediction is then computed by taking a weighted average of the target user's ratings on these similar items. 

We describe these two aspects, namely, the similarity computation and the prediction generation in details here. 

3. 항목 기반 협업 필터링 알고리즘
이 섹션에서는 사용자에게 예측을 생성하기위한 항목 기반 추천 알고리즘 클래스를 연구합니다.

섹션 2에서 논의 된 사용자 기반 협업 필터링 알고리즘과 달리, 항목 기반 접근 방식은 대상 사용자가 평가 한 항목 세트를 조사하고 대상 항목 i와 얼마나 유사한 지 계산 한 다음 k 개의 가장 유사한 항목 fi1을 선택합니다. i2; :::; ikg.

동시에 그들의 유사성 {si1; si2; :::; sik}도 계산됩니다.

가장 유사한 항목이 발견되면 이러한 유사한 항목에 대한 대상 사용자 평가의 가중 평균을 취하여 예측을 계산합니다.

이 두 가지 측면, 즉 유사성 계산과 예측 생성에 대해 자세히 설명합니다.




3.1 Item Similarity Computation 

One critical step in the item-based collaborative filtering algorithm is to compute the similarity between items and then to select the most similar items. 

The basic idea in similarity computation between two items i and j is to first isolate the users who have rated both of these items and then to apply a similarity computation technique to determine the similarity si;j . 

Figure 2 illustrates this process; here the matrix rows represent users and the columns represent items.

There are a number of different ways to compute the similarity between items. 

Here we present three such methods.

These are cosine-based similarity, correlation-based similarity and adjusted-cosine similarity.

3.1 항목 유사성 계산

항목 기반 협업 필터링 알고리즘의 중요한 단계 중 하나는 항목 간의 유사성을 계산 한 다음 가장 유사한 항목을 선택하는 것입니다.

두 항목 i 및 j 간의 유사성 계산의 기본 아이디어는 먼저 이러한 항목을 모두 평가 한 사용자를 분리 한 다음 유사성 계산 기술을 적용하여 유사성 si; j를 결정하는 것입니다.

그림 2는이 프로세스를 보여줍니다. 여기서 행렬 행은 사용자를 나타내고 열은 항목을 나타냅니다.

항목 간의 유사성을 계산하는 방법에는 여러 가지가 있습니다.

여기서 우리는 그러한 세 가지 방법을 제시합니다.

이들은 코사인 기반 유사성, 상관 기반 유사성 및 조정 된 코사인 유사성입니다.





3.1.1 Cosine-based Similarity
In this case, two items are thought of as two vectors in the m dimensional user-space. 

The similarity between them is measured by computing the cosine of the angle between these two vectors. 

Formally, in the m  n ratings matrix in Figure 2, similarity between items i and j, denoted by
sim(i; j) is given by
sim(i; j)공식
where "$ \cdot $" denotes the dot-product of the two vectors.

3.1.2 Correlation-based Similarity
In this case, similarity between two items i and j is measured by computing the Pearson-r correlation corri;j . 

To make the correlation computation accurate we must first isolate the co-rated cases (i.e., cases where the users rated both i and j) as shown in Figure 2. 

Let the set of users who both rated i and j are denoted by U then the correlation similarity is given by
sim(i; j) 공식
Here R_ui denotes the rating of user u on item i, Ri is the average rating of the i-th item.


3.1.1 코사인 기반 유사성
이 경우 두 항목은 m 차원 사용자 공간에서 두 벡터로 간주됩니다.

이들 사이의 유사성은이 두 벡터 사이 각도의 코사인을 계산하여 측정됩니다.

공식적으로 그림 2의 m n 등급 행렬에서 항목 i와 j 사이의 유사성은 다음과 같이 표시됩니다.
sim (i; j)는 다음과 같이 주어진다.
sim (i; j) 공식
여기서 "$ \ cdot $"는 두 벡터의 내적을 나타냅니다.

3.1.2 상관 기반 유사성
이 경우 두 항목 i와 j 사이의 유사성은 Pearson-r 상관 corri; j를 계산하여 측정됩니다.

상관 관계 계산을 정확하게하기 위해 먼저 그림 2와 같이 공동 평가 된 케이스 (즉, 사용자가 i와 j를 모두 평가 한 케이스)를 분리해야합니다.

i와 j를 모두 평가 한 사용자 세트를 U로 표시하면 상관 유사성은 다음과 같이 표시됩니다.
sim (i; j) 공식
여기서 R_ui는 항목 i에 대한 사용자 u의 등급을 나타내고 Ri는 i 번째 항목의 평균 등급입니다.







3.1.3 Adjusted Cosine Similarity
One fundamental difference between the similarity computation in user-based CF and item-based CF is that in case of user-based CF the similarity is computed along the rows of the matrix but in case of the item-based CF the similarity is computed along the columns, i.e., each pair in the co-rated set corresponds to a different user (Figure 2). 

Computing similarity using basic cosine measure in item-based case has one important drawback|the differences in rating scale between different users are not taken into account.

The adjusted cosine similarity offsets this drawback by subtracting the corresponding user average from each co-rated pair. 

Formally, the similarity between items i and j using this scheme is given by
sim(i; j)공식 
Here Ru is the average of the u-th user's ratings.

3.1.3 조정 된 코사인 유사성
사용자 기반 CF와 항목 기반 CF의 유사성 계산의 근본적인 차이점 중 하나는 사용자 기반 CF의 경우 유사성이 행렬의 행을 따라 계산되지만 항목 기반 CF의 경우 유사성이 함께 계산된다는 것입니다. 즉, 공동 등급 세트의 각 쌍은 서로 다른 사용자에 해당합니다 (그림 2).

항목 기반 사례에서 기본 코사인 척도를 사용하여 유사성을 계산하는 것은 한 가지 중요한 단점이 있습니다. 다른 사용자 간의 평가 척도 차이를 고려하지 않았습니다.

조정 된 코사인 유사성은 각 공동 등급 쌍에서 해당 사용자 평균을 빼서이 단점을 상쇄합니다.

공식적으로,이 체계를 사용하는 항목 i와 j 사이의 유사성은 다음과 같이 주어진다.
sim (i; j) 공식
여기서 Ru는 u 번째 사용자의 평점 평균입니다.






3.2 Prediction Computation
The most important step in a collaborative filtering system is to generate the output interface in terms of prediction.

Once we isolate the set of most similar items based on the similarity measures, the next step is to look into the target users ratings and use a technique to obtain predictions. 

Here we consider two such techniques.

3.2.1 Weighted Sum
As the name implies, this method computes the prediction on an item i for a user u by computing the sum of the ratings given by the user on the items similar to i. 

Each ratings is weighted by the corresponding similarity si;j between items i and j. 

Formally, using the notion shown in Figure 3 we can denote the prediction Pu;i as 
Pui공식
Basically, this approach tries to capture how the active user rates the similar items. 

The weighted sum is scaled by the sum of the similarity terms to make sure the prediction is within the predefined range.

3.2 예측 계산
협업 필터링 시스템에서 가장 중요한 단계는 예측 측면에서 출력 인터페이스를 생성하는 것입니다.

유사성 측정 값을 기반으로 가장 유사한 항목 집합을 분리 한 후 다음 단계는 대상 사용자 등급을 조사하고 기술을 사용하여 예측을 얻는 것입니다.

여기서 우리는 그러한 두 가지 기술을 고려합니다.

3.2.1 가중 합계
이름에서 알 수 있듯이이 방법은 i와 유사한 항목에 대해 사용자가 부여한 등급의 합계를 계산하여 사용자 u에 대한 항목 i에 대한 예측을 계산합니다.

각 등급은 항목 i와 j 사이의 해당 유사성 si; j에 의해 가중치가 부여됩니다.

공식적으로 그림 3의 개념을 사용하여 예측 Pu; i를 다음과 같이 나타낼 수 있습니다.
푸이 공식
기본적으로이 접근 방식은 활성 사용자가 유사한 항목을 평가하는 방법을 포착하려고합니다.

가중 합계는 예측이 사전 정의 된 범위 내에 있는지 확인하기 위해 유사성 항의 합계로 조정됩니다.






3.2.2 Regression

This approach is similar to the weighted sum method but instead of directly using the ratings of similar items it uses an approximation of the ratings based on regression model.

In practice, the similarities computed using cosine or correlation measures may be misleading in the sense that two rating vectors may be distant (in Euclidean sense) yet may have very high similarity. 

In that case using the raw ratings of the "so-called" similar item may result in poor prediction.

The basic idea is to use the same formula as the weighted
sum technique, but instead of using the similar item N's "raw" ratings values RuN's, this model uses their approximated values RuN based on a linear regression model. 

If we denote the respective vectors of the target item i and the similar item N by Ri and RN the linear regression model can be expressed as 

RN 공식  

The regression model parameters 알파 and 베타 are determined by going over both of the rating vectors. 

입실론 is the error of the regression model.

3.2.2 회귀

이 접근 방식은 가중 합계 방법과 유사하지만 유사한 항목의 등급을 직접 사용하는 대신 회귀 모델을 기반으로 한 등급의 근사치를 사용합니다.

실제로 코사인 또는 상관 측정을 사용하여 계산 된 유사성은 두 등급 벡터가 멀리 떨어져있을 수 있지만 (유클리드 의미에서) 매우 높은 유사성을 가질 수 있다는 점에서 오해의 소지가 있습니다.

이 경우 "소위"유사한 항목의 원시 등급을 사용하면 예측이 좋지 않을 수 있습니다.

기본 아이디어는 가중치 부여와 동일한 공식을 사용하는 것입니다.
그러나이 모델은 유사한 항목 N의 "원시"등급 값 RuN을 사용하는 대신 선형 회귀 모델을 기반으로하는 근사값 RuN을 사용합니다.

Ri 및 RN으로 대상 항목 i 및 유사한 항목 N의 각 벡터를 표시하면 선형 회귀 모델은 다음과 같이 표현 될 수 있습니다.

RN 공식

회귀 모델 매개 변수 알파 및 베타는 두 등급 벡터를 모두 검토하여 결정됩니다.

입실론은 회귀 모형의 오차입니다.






3.3 Performance Implications

The largest E-Commerce sites operate at a scale that stresses the direct implementation of collaborative filtering.

In neighborhood-based CF systems, the neighborhood formation process, especially the user-user similarity computation step turns out to be the performance bottleneck, which in turn can make the whole process unsuitable for real-time recommendation generation. 

One way of ensuring high scalability is to use a model-based approach. 

Model-based systems have the potential to contribute to recommender systems to operate at a high scale. 

The main idea here to isolate the neighborhood generation and prediction generation steps.

In this paper, we present a model-based approach to precompute item-item similarity scores. 

The similarity computation scheme is still correlation-based but the computation is performed on the item space. 

In a typical E-Commerce scenario, we usually have a set of item that is static compared to the number of users that changes most often. 

The static nature of items leads us to the idea of precomputing the item similarities. 

One possible way of precomputing the item similarities is to compute all-to-all similarity and then performing a quick table look-up to retrieve the required similarity values. 

This method, although saves time, requires an O(n^2) space for n items.

The fact that we only need a small fraction of similar items to compute predictions leads us to an alternate model-based scheme. 

In this scheme, we retain only a small number of similar items. 

For each item j we compute the k most similar items, where k  n and record these item numbers and their similarities with j. 

We term k as the model size. 

3.3 성능 영향

가장 큰 전자 상거래 사이트는 협업 필터링의 직접적인 구현을 강조하는 규모로 운영됩니다.

이웃 기반 CF 시스템에서 이웃 형성 프로세스, 특히 사용자-사용자 유사성 계산 단계는 성능 병목 현상으로 밝혀져 전체 프로세스가 실시간 추천 생성에 적합하지 않을 수 있습니다.

높은 확장 성을 보장하는 한 가지 방법은 모델 기반 접근 방식을 사용하는 것입니다.

모델 기반 시스템은 추천 시스템이 대규모로 작동하는 데 기여할 가능성이 있습니다.

여기서 주요 아이디어는 이웃 생성 및 예측 생성 단계를 분리하는 것입니다.

이 백서에서는 항목-항목 유사성 점수를 미리 계산하기위한 모델 기반 접근 방식을 제시합니다.

유사성 계산 체계는 여전히 상관 관계 기반이지만 계산은 항목 공간에서 수행됩니다.

일반적인 전자 상거래 시나리오에서는 일반적으로 가장 자주 변경되는 사용자 수에 비해 정적 인 항목 집합이 있습니다.

항목의 정적 특성은 항목 유사성을 미리 계산하는 아이디어로 이어집니다.

항목 유사성을 미리 계산하는 한 가지 가능한 방법은 전체 유사성을 계산 한 다음 빠른 테이블 조회를 수행하여 필요한 유사성 값을 검색하는 것입니다.

이 방법은 시간을 절약하지만 n 개 항목에 대해 O (n ^ 2) 공간이 필요합니다.

예측을 계산하기 위해 유사한 항목의 작은 부분 만 필요하다는 사실은 우리를 대체 모델 기반 체계로이 끕니다.

이 계획에서는 소수의 유사한 항목 만 유지합니다.

각 항목 j에 대해 우리는 k n 가장 유사한 항목을 계산하고 이러한 항목 번호와 그 유사성을 j로 기록합니다.

k를 모델 크기라고합니다.







Based on this model building step, our prediction generation algorithm works as follows. 

For generating predictions for a user u on item i, our algorithm first retrieves the precomputed k most similar items corresponding to the target item i. 

Then it looks how many of those k items were purchased by the user u, based on this intersection then the prediction is computed using basic item-based collaborative filtering algorithm. 

We observe a quality-performance trade-off here: to ensure good quality we must have a large model size, which leads to the performance problems discussed above. 

In one extreme, we can have a model size of n, which will ensure the exact same quality as the original scheme but will have high space complexity. 

However, our model building step ensures that we retain the most similar items. 

While generating predictions, these items contribute the most to the prediction scores. 

Accordingly, we hypothesize that this model-based approach will provide reasonably good prediction quality with even a small model size and hence provide a good performance. 

We experimentally validate our hypothesis later in this paper. 

In particular, we experiment with the model size by varying the number of similar items to be stored.

Then we perform experiments to compute prediction and response-time to determine the impact of the model size on quality and performance of the whole system.

이 모델 구축 단계를 기반으로 예측 생성 알고리즘은 다음과 같이 작동합니다.

항목 i에 대한 사용자 u에 대한 예측을 생성하기 위해 알고리즘은 먼저 대상 항목 i에 해당하는 사전 계산 된 k 개의 가장 유사한 항목을 검색합니다.

그런 다음 사용자 u가 구매 한 k 개의 항목 중 몇 개를이 교차점을 기반으로하여 예측은 기본 항목 기반 협업 필터링 알고리즘을 사용하여 계산됩니다.

우리는 여기서 품질-성능 절충안을 관찰합니다. 좋은 품질을 보장하려면 모델 크기가 커야하며, 이로 인해 위에서 설명한 성능 문제가 발생합니다.

한 가지 극단적 인 경우, 모델 크기 n을 가질 수 있는데, 이는 원래 계획과 똑같은 품질을 보장하지만 공간 복잡성이 높습니다.

그러나 모델 구축 단계에서는 가장 유사한 항목을 유지합니다.

예측을 생성하는 동안 이러한 항목은 예측 점수에 가장 많이 기여합니다.

따라서 우리는이 모델 기반 접근 방식이 작은 모델 크기로도 합리적으로 좋은 예측 품질을 제공하여 좋은 성능을 제공 할 것이라고 가정합니다.

우리는이 백서 뒷부분에서 우리의 가설을 실험적으로 검증합니다.

특히, 저장할 유사한 항목의 수를 변경하여 모델 크기를 실험합니다.

그런 다음 예측 및 응답 시간을 계산하는 실험을 수행하여 모델 크기가 전체 시스템의 품질 및 성능에 미치는 영향을 확인합니다.






4. EXPERIMENTAL EVALUATION
4.1 Data set

We used experimental data from our research website to evaluate different variants of item-based recommendation algorithms.

Movie data. 

We used data from our MovieLens recommender system. 

MovieLens is a web-based research recommender system that debuted in Fall 1997. 

Each week hundreds of users visit MovieLens to rate and receive recommendations for movies. 

The site now has over 43000 users who have expressed opinions on 3500+ different movies. 

We randomly selected enough users to obtain 100; 000 ratings from the database (we only considered users that had rated 20 or more movies). 

We divided the database into a training set and a test set. 

For this purpose, we introduced a variable that determines what percentage of data is used as training and test sets; we call this variable x. 

A value of x = 0:8 would indicate 80% of the data was used as training set and 20% of the data was used as test set. 

The data set was converted into a user-item matrix A that had 943 rows (i.e., 943 users) and 1682 columns (i.e., 1682 movies that were rated by at least one of the users). 

For our experiments, we also take another factor into consideration, sparsity level of data sets. 

For the data matrix R This is defined as 1  nonzero entries total entries . 

The sparsity level of the Movie data set is, therefore, 1 - 100,000/(943*1682) , which is 0:9369.

Throughout the paper we term this data set as ML.

4. 실험적 평가
4.1 데이터 세트

연구 웹 사이트의 실험 데이터를 사용하여 항목 기반 추천 알고리즘의 다양한 변형을 평가했습니다.

영화 데이터.

MovieLens 추천 시스템의 데이터를 사용했습니다.

MovieLens는 1997 년 가을에 데뷔 한 웹 기반 연구 추천 시스템입니다.

매주 수백 명의 사용자가 MovieLens를 방문하여 영화를 평가하고 추천을받습니다.

현재이 사이트에는 3500 개 이상의 다양한 영화에 대한 의견을 표명 한 43000 명 이상의 사용자가 있습니다.

100 명을 얻기에 충분한 사용자를 무작위로 선택했습니다. 데이터베이스의 000 등급 (영화 20 개 이상의 등급을받은 사용자 만 고려).

우리는 데이터베이스를 훈련 세트와 테스트 세트로 나누었습니다.

이를 위해 학습 및 테스트 세트로 사용되는 데이터의 비율을 결정하는 변수를 도입했습니다. 이 변수를 x라고 부릅니다.

x = 0 : 8 값은 데이터의 80 %가 학습 세트로 사용되고 데이터의 20 %가 테스트 세트로 사용되었음을 나타냅니다.

데이터 세트는 943 개의 행 (즉, 943 명의 사용자)과 1,682 개의 열 (즉, 최소 한 명의 사용자가 평가 한 1682 개의 영화)이있는 사용자 항목 행렬 A로 변환되었습니다.

실험을 위해 데이터 세트의 희소성 수준이라는 또 다른 요소도 고려합니다.

데이터 행렬 R의 경우 이것은 0이 아닌 1 개의 총 항목으로 정의됩니다.

따라서 Movie 데이터 세트의 희소성 수준은 1-100,000 / (943 * 1682)이며 0 : 9369입니다.

백서 전체에서이 데이터 세트를 ML이라고합니다.






4.2 Evaluation Metrics
Recommender systems research has used several types of measures for evaluating the quality of a recommender system. 

They can be mainly categorized into two classes: 

* Statistical accuracy metrics evaluate the accuracy of a system by comparing the numerical recommendation scores against the actual user ratings for the user-item pairs in the test dataset. 

Mean Absolute Error (MAE) between ratings and predictions is a widely used metric. 

MAE is a measure of the deviation of recommendations from their true user-specified values. 

For each ratings-prediction pair < pi ; qi > this metric treats the absolute error between them, i.e., jpi qi j equally. 

The MAE is computed by first summing these absolute errors of the N corresponding ratings-prediction pairs and then computing the average. 

Formally, MAE공식 

The lower the MAE, the more accurately the recommendation engine predicts user ratings. 

Root Mean Squared Error (RMSE), and Correlation are also used as statistical accuracy metric.

* Decision support accuracy metrics evaluate how effective a prediction engine is at helping a user select highquality items from the set of all items. 

These metrics assume the prediction process as a binary operationeither items are predicted (good) or not (bad). 

With this observation, whether a item has a prediction score of 1:5 or 2:5 on a five-point scale is irrelevant if the user only chooses to consider pred most commonly used decision support accuracy metrics are reversal rate, weighted errors and ROC sensitivity [23].

We used MAE as our choice of evaluation metric to report prediction experiments because it is most commonly used and easiest to interpret directly. 

In our previous experiments [23] we have seen that MAE and ROC provide the same ordering of different experimental schemes in terms of prediction quality.

4.2 평가 지표
추천 시스템 연구는 추천 시스템의 품질을 평가하기 위해 여러 유형의 측정을 사용했습니다.

주로 두 가지 클래스로 분류 할 수 있습니다.

* 통계적 정확도 메트릭은 수치 추천 점수를 테스트 데이터 세트의 사용자 항목 쌍에 대한 실제 사용자 등급과 비교하여 시스템의 정확도를 평가합니다.

평점과 예측 사이의 평균 절대 오차 (MAE)는 널리 사용되는 측정 항목입니다.

MAE는 실제 사용자 지정 값에서 권장 사항의 편차를 측정 한 것입니다.

각 등급 예측 쌍에 대해 <pi; qi>이 측정 항목은 둘 사이의 절대 오차, 즉 jpi qi j를 동일하게 처리합니다.

MAE는 먼저 N 개의 해당 등급-예측 쌍의 절대 오차를 합한 다음 평균을 계산하여 계산됩니다.

공식적으로 매 공식

MAE가 낮을수록 추천 엔진이 사용자 평점을 더 정확하게 예측합니다.

RMSE (Root Mean Squared Error) 및 상관 관계도 통계 정확도 메트릭으로 사용됩니다.

* 의사 결정 지원 정확도 메트릭은 사용자가 모든 항목 집합에서 고품질 항목을 선택하는 데 예측 엔진이 얼마나 효과적인지 평가합니다.

이러한 측정 항목은 예측 프로세스를 항목이 예측 (양호)되거나 그렇지 않은 (나쁨) 이진 연산으로 가정합니다.

이 관찰을 통해 항목의 예측 점수가 5 점 척도에서 1 : 5인지 2 : 5인지 여부는 사용자가 가장 일반적으로 사용되는 의사 결정 지원 정확도 메트릭이 반전 률, 가중 오류 및 ROC를 고려하도록 선택하는 경우 관련이 없습니다. 감도 [23].

MAE는 가장 일반적으로 사용되고 직접 해석하기 쉽기 때문에 예측 실험을보고하기위한 평가 측정 항목으로 선택했습니다.

이전 실험 [23]에서 MAE와 ROC가 예측 품질 측면에서 서로 다른 실험 계획의 동일한 순서를 제공한다는 것을 확인했습니다.







4.2.1 Experimental Procedure

Experimental steps. 

We started our experiments by first dividing the data set into a training and a test portion. 

Before starting full experimental evaluation of different algorithms we determined the sensitivity of different parameters to different algorithms and from the sensitivity plots we fixed the optimum values of these parameters and used them for the rest of the experiments. 

To determine the parameter sensitivity, we work only with the training data and further subdivide it into a training and test portion and carried on our experiments on them. 

For conducted a 10-fold cross validation of our experiments by randomly choosing different training and test sets each time and taking the average of the MAE values.

Benchmark user-based system. 

To compare the performance of item-based prediction we also entered the training ratings set into a collaborative filtering recommendation engine that employs the Pearson nearest neighbor algorithm (user-user). 

For this purpose we implemented a exible prediction engine that implements user-based CF algorithms.

We tuned the algorithm to use the best published Pearson nearest neighbor algorithm and configured it to deliver the highest quality prediction without concern for performance (i.e., it considered every possible neighbor to form optimal neighborhoods).

Experimental platform. 

All our experiments were implemented using C and compiled using optimization  flag-06.

We ran all our experiments on a linux based PC with Intel Pentium III processor having a speed of 600 MHz and 2GB of RAM.

4.2.1 실험 절차

실험 단계.

먼저 데이터 세트를 훈련 부분과 테스트 부분으로 나누면서 실험을 시작했습니다.

서로 다른 알고리즘에 대한 전체 실험 평가를 시작하기 전에 서로 다른 알고리즘에 대한 서로 다른 매개 변수의 민감도를 결정했고 민감도 플롯에서 이러한 매개 변수의 최적 값을 고정하고 나머지 실험에 사용했습니다.

매개 변수 민감도를 결정하기 위해 우리는 훈련 데이터로만 작업하고이를 훈련 및 테스트 부분으로 더 세분화하고 이에 대한 실험을 수행했습니다.

매번 다른 훈련 및 테스트 세트를 무작위로 선택하고 MAE 값의 평균을 취하여 실험의 10 배 교차 검증을 수행했습니다.

사용자 기반 시스템을 벤치 마크합니다.

항목 기반 예측의 성능을 비교하기 위해 Pearson 최근 접 이웃 알고리즘 (사용자-사용자)을 사용하는 협업 필터링 권장 엔진에 설정된 훈련 등급도 입력했습니다.

이를 위해 사용자 기반 CF 알고리즘을 구현하는 확장 가능한 예측 엔진을 구현했습니다.

우리는 가장 잘 게시 된 Pearson 최근 접 이웃 알고리즘을 사용하도록 알고리즘을 조정하고 성능에 대한 걱정없이 최고 품질의 예측을 제공하도록 구성했습니다 (즉, 가능한 모든 이웃이 최적의 이웃을 형성하도록 고려함).

실험적 플랫폼.

모든 실험은 C를 사용하여 구현되었고 최적화 플래그 -06을 사용하여 컴파일되었습니다.

모든 실험은 속도가 600MHz이고 RAM이 2GB 인 Intel Pentium III 프로세서가있는 Linux 기반 PC에서 실행되었습니다.







4.3 Experimental Results
In this section we present our experimental results of applying item-based collaborative filtering techniques for generating predictions. 

Our results are mainly divided into two parts|quality results and performance results. 

In assessing the quality of recommendations, we first determined the sensitivity of some parameters before running the main experiment. 

These parameters include the neighborhood size, the value of the training/test ratio x, and effects of different similarity measures. 

For determining the sensitivity of various parameters, we focused only on the training data set and further divided it into a training and a test portion and used them to learn the parameters.

4.3.1 Effect of Similarity Algorithms
We implemented three different similarity algorithms basic cosine, adjusted cosine and correlation as described in Section 3.1 and tested them on our data sets. 

For each similarity algorithms, we implemented the algorithm to compute the neighborhood and used weighted sum algorithm to generate the prediction. 

We ran these experiments on our training data and used test set to compute Mean Absolute Error (MAE). 

Figure 4 shows the experimental results. 

It can be observed from the results that offsetting the user-average for cosine similarity computation has a clear advantage, as the MAE is significantly lower in this case. 

Hence, we select the adjusted cosine similarity for the rest of our experiments.

4.3.2 Sensitivity of Training/Test Ratio
To determine the sensitivity of density of the data set, we carried out an experiment where we varied the value of x from 0:2 to 0:9 in an increment of 0:1. 

For each of these training/test ratio values we ran our experiments using the two prediction generation techniques{basic weighted sum and regression based approach. 

Our results are shown in Figure 5. 

We observe that the quality of prediction increase as we increase x. 

The regression-based approach shows better results than the basic scheme for low values of x but as we increase x the quality tends to fall below the basic scheme. 

From the curves, we select x = 0:8 as an optimum value for our subsequent experiments.


4.3 실험 결과
이 섹션에서는 예측 생성을 위해 항목 기반 협업 필터링 기술을 적용한 실험 결과를 제시합니다.

우리의 결과는 주로 두 부분의 품질 결과와 성능 결과로 나뉩니다.

권장 사항의 품질을 평가할 때 먼저 주요 실험을 실행하기 전에 일부 매개 변수의 민감도를 결정했습니다.

이러한 매개 변수에는 이웃 크기, 훈련 / 검정 비율 x의 값 및 다양한 유사성 측정의 효과가 포함됩니다.

다양한 매개 변수의 민감도를 결정하기 위해 우리는 훈련 데이터 세트에만 집중하고이를 훈련과 테스트 부분으로 더 나누고 매개 변수를 학습하는 데 사용했습니다.

4.3.1 유사성 알고리즘의 효과
3.1 절에 설명 된대로 세 가지 유사성 알고리즘의 기본 코사인, 조정 된 코사인 및 상관 관계를 구현하고 데이터 세트에서 테스트했습니다.

각 유사성 알고리즘에 대해 이웃을 계산하는 알고리즘을 구현하고 가중 합계 알고리즘을 사용하여 예측을 생성했습니다.

훈련 데이터에서 이러한 실험을 실행하고 테스트 세트를 사용하여 절대 평균 오차 (MAE)를 계산했습니다.

그림 4는 실험 결과를 보여줍니다.

이 경우 MAE가 현저히 낮기 때문에 코사인 유사성 계산에 대한 사용자 평균을 상쇄하는 것이 분명한 이점이 있음을 결과에서 볼 수 있습니다.

따라서 나머지 실험에 대해 조정 된 코사인 유사성을 선택합니다.

4.3.2 훈련 / 테스트 비율의 민감도
데이터 세트의 밀도 민감도를 결정하기 위해 x 값을 0 : 1 씩 0 : 2에서 0 : 9로 변경하는 실험을 수행했습니다.

이러한 각 학습 / 테스트 비율 값에 대해 두 가지 예측 생성 기술 (기본 가중 합계 및 회귀 기반 접근 방식)을 사용하여 실험을 실행했습니다.

결과는 그림 5에 나와 있습니다.

x가 증가할수록 예측의 질이 증가하는 것을 관찰합니다.

회귀 기반 접근 방식은 x의 낮은 값에 대한 기본 계획보다 더 나은 결과를 보여 주지만 x를 증가 시키면 품질이 기본 계획 아래로 떨어지는 경향이 있습니다.

곡선에서 x = 0 : 8을 후속 실험에 대한 최적 값으로 선택합니다.








4.3.3 Experiments with neighborhood size
The size of the neighborhood has significant impact on the prediction quality [12]. 

To determine the sensitivity of this parameter, we performed an experiment where we varied the number of neighbors to be used and computed MAE.

Our results are shown in Figure 5. 

We can observe that the size of neighborhood does affect the quality of prediction. 

But the two methods show different types of sensitivity. 

The basic item-item algorithm improves as we increase the neighborhood size from 10 to 30, after that the rate of increase diminishes and the curve tends to be at. 

On the other hand, the regression-based algorithm shows decrease in prediction quality with increased number of neighbors.

Considering both trends we select 30 as our optimal choice of neighborhood size.

4.3.3 이웃 크기 실험
이웃의 크기는 예측 품질에 상당한 영향을 미칩니다 [12].

이 매개 변수의 민감도를 결정하기 위해 사용할 이웃 수를 변경하고 MAE를 계산하는 실험을 수행했습니다.

결과는 그림 5에 나와 있습니다.

이웃의 크기가 예측의 질에 영향을 미친다는 것을 알 수 있습니다.

그러나 두 가지 방법은 서로 다른 유형의 감도를 보여줍니다.

기본 항목-항목 알고리즘은 이웃 크기를 10에서 30으로 늘리면 증가율이 감소하고 곡선이되는 경향이 있습니다.

반면 회귀 기반 알고리즘은 이웃 수가 증가함에 따라 예측 품질이 저하되는 것을 보여줍니다.

두 가지 추세를 모두 고려하여 30을 최적의 이웃 크기로 선택합니다.





4.3.4 Quality Experiments
Once we obtain the optimal values of the parameters, we compare both of our item-based approaches with the benchmark user-based algorithm. 

We present the results in Figure 6. 

It can be observed from the charts that the basic item-item algorithm out performs the user based algorithm at all values of x (neighborhood size = 30) and all values of neighborhood size (x = 0:8). 

For example, at x = 0:5 user-user scheme has an MAE of 0:755 and item-item scheme shows an MAE of 0:749. 

Similarly at a neighborhood size of 60 user-user and item-item schemes show MAE of 0:732 and 0:726 respectively. 

The regression-based algorithm, however, shows interesting behavior. 

At low values of x and at low neighborhood size it out performs the other two algorithms, but as the density of the data set is increased or as we add more neighbors it performs worse, even compared to the user-based algorithm. 

We also compared our algorithms against the naive nonpersonalized algorithm described in [12].

We draw two conclusions from these results. 

First, itembased algorithms provide better quality than the user-based algorithms at all sparsity levels. 

Second, regression-based algorithms perform better with very sparse data set, but as we add more data the quality goes down. 

We believe this happens as the regression model suffers from data overfitting at high density levels.

4.3.4 품질 실험
매개 변수의 최적 값을 얻은 후에는 두 항목 기반 접근 방식을 벤치 마크 사용자 기반 알고리즘과 비교합니다.

결과는 그림 6에 나와 있습니다.

기본 항목-항목 알고리즘 출력은 x (이웃 크기 = 30)의 모든 값과 이웃 크기 (x = 0 : 8)의 모든 값에서 사용자 기반 알고리즘을 수행하는 것을 차트에서 확인할 수 있습니다.

예를 들어, at x = 0 : 5 사용자-사용자 체계는 MAE가 0 : 755이고 항목-항목 체계는 MAE가 0 : 749입니다.

유사하게 60 개의 사용자-사용자 및 항목-항목 체계의 이웃 크기에서 MAE는 각각 0 : 732 및 0 : 726입니다.

그러나 회귀 기반 알고리즘은 흥미로운 동작을 보여줍니다.

x 값이 낮고 이웃 크기가 낮 으면 다른 두 알고리즘을 수행하지만 데이터 세트의 밀도가 증가하거나 이웃을 더 추가하면 사용자 기반 알고리즘에 비해 성능이 저하됩니다.

또한 우리의 알고리즘을 [12]에 설명 된 순진한 비 개인화 알고리즘과 비교했습니다.

이 결과에서 두 가지 결론을 도출합니다.

첫째, 항목 기반 알고리즘은 모든 희소성 수준에서 사용자 기반 알고리즘보다 더 나은 품질을 제공합니다.

둘째, 회귀 기반 알고리즘은 매우 희소 한 데이터 세트에서 더 잘 수행되지만 더 많은 데이터를 추가하면 품질이 저하됩니다.

회귀 모델이 고밀도 수준에서 데이터 과적 합으로 어려움을 겪기 때문에 이런 일이 발생한다고 생각합니다.






4.3.5 Performance Results
After showing that the item-based algorithm provides better quality of prediction than the user-based algorithm, we focus on the scalability issues. 

As discussed earlier, itembased similarity is more static and allows us to precompute the item neighborhood. 

This precomputation of the model has certain performance benefits. 

To make the system even more scalable we looked into the sensitivity of the model size and then looked into the impact of model size on the response time and throughput.


4.3.5 성능 결과
항목 기반 알고리즘이 사용자 기반 알고리즘보다 더 나은 예측 품질을 제공함을 보여준 후 확장 성 문제에 중점을 둡니다.

앞서 논의했듯이 항목 기반 유사성은 더 정적이고 항목 이웃을 미리 계산할 수 있습니다.

이 모델의 사전 계산에는 특정 성능 이점이 있습니다.

시스템의 확장 성을 더욱 높이기 위해 모델 크기의 민감도를 조사한 다음 모델 크기가 응답 시간 및 처리량에 미치는 영향을 조사했습니다.





4.4 Sensitivity of the Model Size
To experimentally determine the impact of the model size on the quality of the prediction, we selectively varied the number of items to be used for similarity computation from 25 to 200 in an increment of 25. 

A model size of l means that we only considered l best similarity values for model building and later used k of them for the prediction generation process, where k < l. 

Using the training data set we precomputed the item similarity using different model sizes and then used only the weighted sum prediction generation technique to provide the predictions. 

We then used the test data set to compute MAE and plotted the values. 

To compare with the full model size (i.e., model size = no. of items) we also ran the same test considering all similarity values and picked best k for prediction generation. 

We repeated the entire process for three different x values (training/test ratios). 

Figure 7 shows the plots at different x values. 

It can be observed from the plots that the MAE values get better as we increase the model size and the improvements are drastic at the beginning, but gradually slow down as we increase the model size. 

The most important observation from these plots is the high accuracy that can be achieved using only a fraction of items. 

For example, at x = 0:3 the full item-item scheme provided an MAE of 0:7873, but using a model size of only 25, we were able to achieve an MAE value of 0:842. 

At x = 0:8 these numbers are even more appealing|for the full item-item we had an MAE of 0:726 but using a model size of only 25 we were able to obtain an MAE of 0:754, and using a model size of 50 the MAE was 0:738. 

In other words, at x = 0:8 we were within 96% and 98:3% of the full item-item scheme's accuracy using only 1.9% and 3% of the items, respectively!

This model size sensitivity has important performance implications. 

It appears from the plots that it is useful to precompute the item similarities using only a fraction of items and yet possible to obtain good prediction quality.

4.4 모델 크기의 민감도
모델 크기가 예측 품질에 미치는 영향을 실험적으로 확인하기 위해 유사성 계산에 사용할 항목 수를 25 개에서 200 개씩 25 개씩 선택적으로 변경했습니다.

모델 크기가 l이라는 것은 모델 구축을 위해 l 개의 최상의 유사성 값만 고려하고 나중에 예측 생성 프로세스에 k <l을 사용했음을 의미합니다.

훈련 데이터 세트를 사용하여 서로 다른 모델 크기를 사용하여 항목 유사성을 미리 계산 한 다음 가중 합계 예측 생성 기술 만 사용하여 예측을 제공했습니다.

그런 다음 테스트 데이터 세트를 사용하여 MAE를 계산하고 값을 플로팅했습니다.

전체 모델 크기 (즉, 모델 크기 = 항목 수)와 비교하기 위해 모든 유사성 값을 고려하여 동일한 테스트를 실행하고 예측 생성을 위해 최상의 k를 선택했습니다.

세 가지 x 값 (훈련 / 테스트 비율)에 대해 전체 프로세스를 반복했습니다.

그림 7은 서로 다른 x 값의 플롯을 보여줍니다.

플롯에서 볼 수 있듯이 모델 크기를 늘리면 MAE 값이 좋아지고 처음에는 개선이 과감하지만 모델 크기를 늘릴수록 점차 느려집니다.

이 플롯에서 가장 중요한 관찰은 항목의 일부만 사용하여 얻을 수있는 높은 정확도입니다.

예를 들어, x = 0 : 3에서 전체 항목-항목 체계는 0 : 7873의 MAE를 제공했지만 모델 크기가 25 인 경우 MAE 값 0 : 842를 얻을 수있었습니다.

x = 0 : 8에서이 숫자는 전체 항목 항목에 대해 훨씬 더 매력적입니다 | 우리는 0 : 726의 MAE를 가졌지 만 25의 모델 크기를 사용하여 0 : 754의 MAE를 얻을 수있었습니다. 50의 모델 크기 MAE는 0 : 738입니다.

즉, x = 0 : 8에서 우리는 아이템의 1.9 %와 3 %만을 사용하여 전체 아이템-아이템 체계의 정확도의 96 %와 98 : 3 % 이내였습니다!

이 모델 크기 민감도는 성능에 중요한 영향을 미칩니다.

항목의 일부만 사용하여 항목 유사성을 미리 계산하는 것이 유용하지만 좋은 예측 품질을 얻을 수 있다는 것이 플롯에서 나타납니다.





4.4.1 Impact of the model size on run-time and throughput
Given the quality of prediction is reasonably good with small model size, we focus on the run-time and throughput of the system. 

We recorded the time required to generate predictions for the entire test set and plotted them in a chart with varying model size. 

We plotted the run time at different x values. 

Figure 8 shows the plot. 

Note here that at x = 0:25 the whole system has to make prediction for 25; 000 test cases. 

From the plot we observe a substantial difference in the run-time between the small model size and the full item-item prediction case. 

For x = 0:25 the run-time is 2:002 seconds for a model size of 200 as opposed to 14:11 for the basic item-item case. 

This difference is even more prominent with x = 0:8 where a model size of 200 requires only 1:292 seconds and the basic item-item case requires 36:34 seconds.

These run-time numbers may be misleading as we computed them for different training/test ratios where the workload size, i.e., number of predictions to be generated is different (recall that at x = 0:3 our algorithm uses 30; 000 ratings as training data and uses the rest of 70; 000 ratings as test data to compare predictions generated by the system to the actual ratings). 

To make the numbers comparable we compute the throughput (predictions generated per second) for the model based and basic item-item schemes. 

Figure 8 charts these results. 

We see that for x = 0:3 and at a model size of 100 the system generates 70; 000 ratings in 1:487 seconds producing a throughput rate of 47; 361 where as the basic item-item scheme produced a throughput of 4961 only. 

At x = 0:8 these two numbers are 21; 505 and 550 respectively.

4.4.1 런타임 및 처리량에 대한 모델 크기의 영향
모델 크기가 작을수록 예측 품질이 상당히 좋으므로 시스템의 런타임과 처리량에 중점을 둡니다.

전체 테스트 세트에 대한 예측을 생성하는 데 필요한 시간을 기록하고 다양한 모델 크기를 사용하여 차트에 표시했습니다.

우리는 다른 x 값에서 런타임을 플로팅했습니다.

그림 8은 플롯을 보여줍니다.

여기서 x = 0:25에서 전체 시스템은 25에 대한 예측을해야합니다. 000 개의 테스트 케이스.

플롯에서 우리는 작은 모델 크기와 전체 항목-항목 예측 케이스 간의 런타임에서 상당한 차이를 관찰합니다.

x = 0:25의 경우 런타임은 기본 항목-항목 케이스의 경우 14:11과 달리 모델 크기 200의 경우 2 : 002 초입니다.

이 차이는 x = 0 : 8에서 더욱 두드러지며 200의 모델 크기는 1 : 292 초만 필요하고 기본 항목-항목 케이스는 36:34 초가 필요합니다.

이러한 런타임 수치는 워크로드 크기, 즉 생성 될 예측 수가 다른 여러 훈련 / 테스트 비율에 대해 계산했기 때문에 오해의 소지가있을 수 있습니다 (x = 0 : 3에서 우리 알고리즘은 30을 사용합니다. 시스템에서 생성 된 예측을 실제 평가와 비교하기위한 테스트 데이터로 나머지 70, 000 개의 평가를 사용합니다.

수치를 비교할 수 있도록 모델 기반 및 기본 항목-항목 체계에 대한 처리량 (초당 생성 된 예측)을 계산합니다.

그림 8은 이러한 결과를 보여줍니다.

x = 0 : 3이고 모델 크기가 100 일 때 시스템은 70을 생성합니다. 1 : 487 초에 000 등급으로 47의 처리 속도를 생성합니다. 361 여기서 기본 항목-항목 체계는 4961의 처리량 만 생성했습니다.

x = 0 : 8에서이 두 숫자는 21입니다. 각각 505와 550.







4.5 Discussion
From the experimental evaluation of the item-item collaborative filtering scheme we make some important observations. 

First, the item-item scheme provides better quality of predictions than the use-user (k-nearest neighbor) scheme.

The improvement in quality is consistent over different neighborhood size and training/test ratio. 

However, the improvement is not significantly large. 

The second observation is that the item neighborhood is fairly static, which can be potentially pre-computed, which results in very high online performance. 

Furthermore, due to the model-based approach, it is possible to retain only a small subset of items and produce reasonably good prediction quality. 

Our experimental results support that claim. 

Therefore, the itemitem scheme is capable in addressing the two most important challenges of recommender systems for E-Commerce{quality of prediction and high performance.


4.5 토론
항목-항목 협업 필터링 체계의 실험적 평가에서 몇 가지 중요한 관찰을 수행합니다.

첫째, 항목-항목 체계는 사용 사용자 (k- 최근 접 이웃) 체계보다 더 나은 예측 품질을 제공합니다.

품질 향상은 이웃 크기와 훈련 / 테스트 비율에 따라 일관됩니다.

그러나 개선은 크게 크지 않습니다.

두 번째 관찰은 항목 이웃이 상당히 정적이고 잠재적으로 사전 계산 될 수있어 온라인 성능이 매우 높다는 것입니다.

또한 모델 기반 접근 방식으로 인해 항목의 작은 하위 집합 만 유지하고 합리적으로 좋은 예측 품질을 생성 할 수 있습니다.

우리의 실험 결과는 그 주장을 뒷받침합니다.

따라서 itemitem 체계는 전자 상거래를위한 추천 시스템의 가장 중요한 두 가지 문제 (예측 품질 및 고성능)를 해결할 수 있습니다.







5. CONCLUSION
Recommender systems are a powerful new technology for extracting additional value for a business from its user databases. 

These systems help users find items they want to buy from a business. 

Recommender systems benefit users by enabling them to find items they like. 

Conversely, they help the business by generating more sales. 

Recommender systems are rapidly becoming a crucial tool in E-commerce on the Web. 

Recommender systems are being stressed by the huge volume of user data in existing corporate databases, and will be stressed even more by the increasing volume of user data available on the Web. 

New technologies are needed that can dramatically improve the scalability of recommender systems.

In this paper we presented and experimentally evaluated a new algorithm for CF-based recommender systems. 

Our results show that item-based techniques hold the promise of allowing CF-based algorithms to scale to large data sets and at the same time produce high-quality recommendations.

5. 결론
Recommender 시스템은 사용자 데이터베이스에서 비즈니스에 대한 추가 가치를 추출하기위한 강력한 신기술입니다.

이러한 시스템은 사용자가 비즈니스에서 구매하려는 항목을 찾는 데 도움이됩니다.

추천 시스템은 사용자가 좋아하는 항목을 찾을 수 있도록하여 사용자에게 혜택을줍니다.

반대로, 그들은 더 많은 판매를 생성하여 비즈니스를 돕습니다.

추천 시스템은 웹에서 전자 상거래에서 중요한 도구가되고 있습니다.

Recommender 시스템은 기존 기업 데이터베이스에있는 방대한 양의 사용자 데이터로 인해 스트레스를 받고 있으며 웹에서 사용 가능한 사용자 데이터의 양이 증가함에 따라 더 많은 스트레스를 받게 될 것입니다.

추천 시스템의 확장 성을 획기적으로 향상시킬 수있는 새로운 기술이 필요합니다.

이 논문에서 우리는 CF 기반 추천 시스템을위한 새로운 알고리즘을 제시하고 실험적으로 평가했습니다.

우리의 결과는 항목 기반 기술이 CF 기반 알고리즘을 대규모 데이터 세트로 확장하는 동시에 고품질 권장 사항을 생성 할 수 있다는 약속을 유지하고 있음을 보여줍니다.


