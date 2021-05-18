## 2012_IEEE_LARS [LARS: A Location-Aware Recommender System]

![main](./image/main.PNG)

---
### Abstract  
* LARS: 위치 인식 추천 시스템(Location-Aware Recommender System)  
* 위치 기반 rating(3):
  * 비 공간 항목에 대한 공간 등급(spatial ratings for non-spatial items)  
  * 공간 항목에 대한 비 공간 등급(nonspatial ratings for spatial items)   
  * 공간 항목에 대한 공간 등급(spatial ratings for spatial items)  
* 사용자 분할: 사용자 평가 위치를 활용  
* 이동 패널티: 항목 위치를 활용  

---

### 1. INTRODUCTION  
* 기존 추천 시스템: CF  
  * triple (user, rating, item) 트리플(사용자, 등급, 항목)   

* 제안되는 LARS: ratings의 공간적 측면
  * |tuple|구성|추가된 튜플|
    |:---:|---|---|
    |4|(user, **ulocation**, rating, item)|ulocation: 사용자 위치|
    |4|(user, rating, item, **ilocation**)|ilocation: 항목 위치|
    |5|(user, **ulocation**, rating, item, **ilocation**)||
  
#### A. Motivation: A Study of Location-Based Ratings
* datasets: MovieLens, Foursquare  

##### Preference locality 
* 선호 지역: 공간 영역 사용자가 다른 지역을 선호 함(인접 지역의 사용자가 선호하지 X)  

* ![Fig1](./image/Fig1.PNG)  
  * (a) 주별 선호 장르 다름  
    * 영화 선호도가 특정 공간 지역에 고유함    
  * (b) 인접 지역 방문자 장소 선호도   
    * 사용자와 공간적으로 가까운 위치 선호, 추천에 영향    

* *localization: 사용지 포함 공간 영역 내, 고유 선호도 -> 추천 영향*  

##### Travel locality  
* 여행 지역: 사용자 장소 방문 시, 제한된 거리 이동하려는 경향  
  * 추천 후보: 이동 거리가 더 가까운 공간 항목 우선순위   

#### B. Our Contribution: LARS - A Location-Aware Recommender Like traditional recommender systems, LARS suggests k items personalized for a querying user u. 

* 사용자 분할: 선호지역성 활용  
  * (user, ulocation, rating, item)  
  * 적응 형 피라미드 구조(adaptive pyramid structure)  
    * 사용자 위치 속성별 등급, 다른 계층 구조에서 다양한 크기의 공간 영역으로 분할  
    * R 지역에있는 쿼리 사용자의 경우 R에있는 등급만 활용하는 기존 CF 적용  
    * 피라미드의 모든 영역을 유지 여부: 확장성, 지역성 균형(상반되는 요소) => 동적 조정  
* 여행 패널티: 이동 거리 있을수록 패널티  
  * (user, rating, item, ilocation)  
  * 모든 공간 항목 이동 거리 계산X(시스템 리소스)  
    * 목록을 변경할 수 없음을 발견, 조기 종료 쿼리 처리 

* *사용자 분할 및 여행 패널티 기술 개별 or 모두 사용*  
* LARS는 CF 대비 2배 정확 추천 생성  

--- 

### 2. LARS OVERVIEW  
#### A. LARS Query Model  
```
U: 사용자(또는 애플리케이션) ID
K: 숫자 제한                         >       LARS      >      추천 항목 K개
L: 위치
```
사용자 (또는 애플리케이션)는 LARS에 사용자 ID U, 숫자 제한 K 및 위치 L을 제공합니다. 그런 다음 LARS는 K 개의 권장 항목을 사용자에게 반환합니다.
Users (or applications) provide LARS with a user id U, numeric limit K, and location L; LARS then returns K recommended items to the user. 


LARS는 스냅 샷 (즉, 일회성) 쿼리와 연속 쿼리를 모두 지원하므로 사용자가 LARS를 구독하고 위치가 변경되면 추천 업데이트를받습니다.
LARS supports both snapshot (i.e., one-time) queries and continuous queries, whereby a user subscribes to LARS and receives recommendation updates as her location changes. 


LARS가 권장 사항을 생성하는 데 사용하는 기술은 시스템에서 사용 가능한 위치 기반 등급 유형에 따라 다릅니다.
The technique LARS uses to produce recommendations depends on the type of location-based rating available in the system. 


위치 기반 등급의 각 유형에 대한 쿼리 처리 지원은 섹션 III에서 V까지 설명합니다.
Query processing support for each type of location-based rating is discussed in Sections III to V.


B. 항목 기반 협업 필터링
B. Item-Based Collaborative Filtering


LARS는 상용 시스템 (예 : Amazon [1])에서 인기와 폭 넓은 채택으로 인해 선택된 항목 기반 협업 필터링 (약칭 CF)을 주요 권장 기법으로 사용합니다.
LARS uses item-based collaborative filtering (abbr. CF) as its primary recommendation technique, chosen due to its popularity and widespread adoption in commercial systems (e.g., Amazon [1]). 


협업 필터링 (CF)은 n 명의 사용자 집합 U = {u1, ..., un} 및 m 개의 항목 집합 I = {i1, ..., im}을 가정합니다.
Collaborative filtering (CF) assumes a set of n users U = {u1, ..., un} and a set of m items I = {i1, ..., im}. 


각 사용자 uj는 일련의 항목 Iuj ⊆ I에 대한 의견을 표현합니다.
Each user uj expresses opinions about a set of items Iuj ⊆ I. 


의견은 숫자 등급 (예 : 1 ~ 5 개의 별표 넷플릭스 등급 [2]) 또는 단항 (예 : Facebook '체크인'[5]) 일 수 있습니다.
Opinions can be a numeric rating (e.g., the Netflix scale of one to five stars [2]), or unary (e.g., Facebook “check-ins” [5]). 


개념적으로 등급은 그림 2 (a)에 설명 된대로 사용자 및 항목을 차원으로하는 매트릭스로 표시됩니다.
Conceptually, ratings are represented as a matrix with users and items as dimensions, as depicted in Figure 2(a). 


쿼리 사용자 u가 주어지면 CF는 u가 가장 좋아할 것으로 예상되는 k 개의 권장 항목 Ir ⊂ I 세트를 생성합니다.
Given a querying user u, CF produces a set of k recommended items Ir ⊂ I that u is predicted to like the most.

1 단계 : 모델 구축.
Phase I: Model Building. 


이 단계는 동일한 사용자가 하나 이상의 공통 등급 (즉, 공동 등급 차원)을 갖는 각 객체 ip 및 iq 쌍에 대한 유사성 점수 sim (ip, iq)을 계산합니다.
This phase computes a similarity score sim(ip,iq) for each pair of objects ip and iq that have at least one common rating by the same user (i.e., co-rated dimensions). 


유사성 계산은 아래에서 다룹니다.
Similarity computation is covered below. 


이러한 점수를 사용하여 그림 2 (b)와 같이 유사성 점수 sim (ip, iq)에 의해 정렬 된 유사 항목 목록 L 인 각 항목 i ∈ I에 대해 저장하는 모델이 구축됩니다.
Using these scores, a model is built that stores for each item i ∈ I, a list L of similar items ordered by a similarity score sim(ip,iq), as depicted in Figure 2(b). 


이 모델을 구축하는 것은 O (R2U) 프로세스이며, 여기서 R과 U는 각각 등급 및 사용자 수입니다.
Building this model is an O(R2U) process, where R and U are the number of ratings and users, respectively. 


각 목록 L에 대해 유사성 점수가 가장 높은 n 개의 가장 유사한 항목 만 저장하여 모델을 자르는 것이 일반적입니다 [9].
It is common to truncate the model by storing, for each list L, only the n most similar items with the highest similarity scores [9]. 


n의 값은 모델 크기라고하며 일반적으로 | I |보다 훨씬 작습니다.
The value of n is referred to as the model size and is usually much less than |I|. 

2 단계 : 추천 생성.
Phase II: Recommendation Generation. 


쿼리하는 사용자 u가 주어지면 u [9]에 의해 평가되지 않은 각 항목 i에 대한 u의 예상 평가 P (u, i)를 계산하여 추천이 생성됩니다.
(1)
이 계산 전에 사용자 u가 평가 한 항목 만 포함하도록 각 유사성 목록 L을 줄입니다.
Given a querying user u, recommendations are produced by computing u’s predicted rating P(u,i) for each item i not rated by u [9]:
(1) 
Before this computation, we reduce each similarity list L to contain only items rated by user u. 



예측은 sim (i, l)에 의해 가중치가 부여 된 관련 항목 l ∈ L에 대한 사용자 u의 등급 인 ru, l의 합이고, l과 후보 항목 i의 유사도를 합한 다음 i와 i 사이의 유사성 점수 합계로 정규화합니다. 엘.
The prediction is the sum of ru,l, a user u’s rating for a related item l ∈ L weighted by sim(i,l), the similarity of l to candidate item i, then normalized by the sum of similarity scores between i and l. 


사용자는 P (u, i)에 의해 순위가 매겨진 top-k 항목을 추천으로받습니다.
The user receives as recommendations the top-k items ranked by P(u,i).

유사성 계산.
Computing Similarity. 



sim (ip, iq)을 계산하기 위해 평가 행렬의 사용자 등급 공간에서 각 항목을 벡터로 나타냅니다.
To compute sim(ip, iq), we represent each item as a vector in the user-rating space of the rating matrix. 


예를 들어, 그림 3은 그림 2 (a)의 행렬에서 항목 ip 및 iq에 대한 벡터를 보여줍니다.
For instance, Figure 3 depicts vectors for items ip and iq from the matrix in Figure 2(a). 


많은 유사성 함수가 제안되었습니다 (예 : Pearson Correlation, Cosine).
우리는 그 인기로 인해 LARS에서 코사인 유사성을 사용합니다.
(2)
이 점수는 벡터의 공동 등급 치수를 사용하여 계산됩니다. 예를 들어 그림 3에서 ip와 iq 간의 코사인 유사성은 원으로 표시된 공동 등급 치수를 사용하여 .7 계산됩니다.
Many similarity functions have been proposed (e.g., Pearson Correlation, Cosine); 
we use the Cosine similarity in LARS due to its popularity:
(2)
This score is calculated using the vectors’ co-rated dimensions, e.g., the Cosine similarity between ip and iq in Figure 3 is .7 calculated using the circled co-rated dimensions. 


코사인 거리는 숫자 등급 (예 : 척도 [1,5])에 유용합니다.
Cosine distance is useful for numeric ratings (e.g., on a scale [1,5]). 


단항 등급의 경우 다른 유사성 함수가 사용됩니다 (예 : 절대 합 [10]).
For unary ratings, other similarity functions are used (e.g., absolute sum [10]).


이 백서에서는 항목 기반 CF를 사용하기로 선택했지만 다른 추천 기술을 사용하는 데 어떤 요인도 배제되지 않습니다.
While we opt to use item-based CF in this paper, no factors disqualify us from employing other recommendation techniques. 


예를 들어, 항목 대신 사용자 간의 상관 관계를 사용하는 사용자 기반 CF [6]를 쉽게 사용할 수 있습니다.
For instance, we could easily employ user-based CF [6], that uses correlations between users (instead of items).



![Fig2](./image/Fig2.PNG)  
![Fig3](./image/Fig3.PNG)  
--- 

### 3. SPATIAL USER RATINGS FOR NON-SPATIAL ITEMS


III. 비 공간 항목에 대한 공간 사용자 등급
III. SPATIAL USER RATINGS FOR NON-SPATIAL ITEMS 


이 섹션에서는 LARS가 튜플 (user, ulocation, rating, item)로 표시되는 비 공간 항목에 대한 공간 등급을 사용하여 권장 사항을 생성하는 방법을 설명합니다.
This section describes how LARS produces recommendations using spatial ratings for non-spatial items represented by the tuple (user, ulocation, rating, item). 


아이디어는 선호 지역성, 즉 사용자 의견이 공간적으로 고유하다는 관찰을 활용하는 것입니다 (섹션 I-A의 분석을 기반으로 함).
The idea is to exploit preference locality, i.e., the observation that user opinions are spatially unique (based on analysis in Section I-A). 


비 공간 항목에 대한 공간 등급을 사용하여 권장 사항을 생성하기위한 세 가지 요구 사항을 식별합니다.
We identify three requirements for producing recommendations using spatial ratings for non-spatial items: 


(1) 지역성 : 사용자 위치가 쿼리하는 사용자 위치와 공간적으로 가까운 (즉, 공간적 이웃) 사용자 위치의 등급에 따라 권장 사항이 영향을 받아야합니다.
(1) Locality: recommendations should be influenced by those ratings with user locations spatially close to the querying user location (i.e., in a spatial neighborhood); 


(2) 확장 성 : 추천 절차 및 데이터 구조는 많은 사용자에게 확장되어야합니다.
(2) Scalability: the recommendation procedure and data structure should scale up to large number of users; 


(3) 영향 : 시스템 사용자는 권장 사항에 영향을 미치는 공간적 이웃 (예 : 도시 블록, 우편 번호 또는 카운티)의 크기를 제어 할 수 있어야합니다.
(3) Influence: system users should have the ability to control the size of the spatial neighborhood (e.g., city block, zip code, or county) that influences their recommendations.


LARS는 적응 형 피라미드 구조를 유지하는 사용자 분할 기술을 사용하여 요구 사항을 충족합니다. 여기서 적응 형 피라미드의 모양은 지역성, 확장 성 및 영향의 세 가지 목표에 따라 결정됩니다.
LARS achieves its requirements by employing a user partitioning technique that maintains an adaptive pyramid structure, where the shape of the adaptive pyramid is driven by the three goals of locality, scalability, and influence. 


아이디어는 등급 튜플 (사용자, ulocation, 등급, 항목)을 ulocation 속성에 따라 공간 영역으로 적응 적으로 분할하는 것입니다.
The idea is to adaptively partition the rating tuples (user, ulocation, rating, item) into spatial regions based on the ulocation attribute. 


그런 다음 LARS는 쿼리하는 사용자를 포함하는 공간 영역 내 등급의 나머지 세 가지 속성 (사용자, 등급, 항목)에 대해 기존의 협업 필터링 방법 (항목 기반 CF 사용)을 사용하여 권장 사항을 생성합니다.
Then, LARS produces recommendations using any existing collaborative filtering method (we use item-based CF) over the remaining three attributes (user, rating, item) of only the ratings within the spatial region containing the querying user. 


등급은 다양한 취향을 가진 사용자로부터 올 수 있으며, 우리의 방법은 특정 공간 영역으로 제한된 등급만을 기반으로 개인화 된 사용자 추천을 생성하도록 협업 필터링을 강제합니다.
We note that ratings can come from users with varying tastes, and that our method only forces collaborative filtering to produce personalized user recommendations based only on ratings restricted to a specific spatial region. 


이 섹션에서는 섹션 III-A의 피라미드 구조, 섹션 III-B의 쿼리 처리, 마지막으로 섹션 III-C의 데이터 구조 유지 관리에 대해 설명합니다. 
In this section, we describe the pyramid structure in Section III-A, query processing in Section III-B, and finally data structure maintenance in Section III-C.

A. 데이터 구조
A. Data Structure


LARS는 그림 4와 같이 부분 피라미드 구조 [11] (부분 쿼드 트리 [12]와 동일)를 사용합니다.
LARS employs a partial pyramid structure [11] (equivalent to a partial quad-tree [12]) as depicted in Figure 4. 


피라미드는 공간을 H 레벨 (즉, 피라미드 높이)로 분해합니다.
The pyramid decomposes the space into H levels (i.e., pyramid height). 


주어진 레벨 h에 대해 공간은 4h 동일 면적 그리드 셀로 분할됩니다.
For a given level h, the space is partitioned into 4h equal area grid cells. 


예를 들어 피라미드 루트 (수준 0)에서 하나의 격자 셀은 전체 지리적 영역을 나타내고 수준 1은 공간을 4 개의 등가 영역 셀로 분할하는 식입니다.
For example, at the pyramid root (level 0), one grid cell represents the entire geographic area, level 1 partitions space into four equi-area cells, and so forth. 


우리는 고유 한 식별자 cid로 각 셀을 나타냅니다.
We represent each cell with a unique identifier cid. 


각 셀에는 셀의 공간 영역에 포함 된 사용자 위치와 함께 공간 등급 만 사용하여 구축 된 항목 기반 협업 필터링 모델이 저장됩니다.
In each cell, we store an item-based collaborative filtering model built using only the spatial ratings with user locations contained in the cell’s spatial region. 


등급은 최대 H 개의 협업 필터링 모델에 기여할 수 있습니다. 포함 된 사용자 위치를 포함하는 가장 낮은 유지 관리 그리드 셀부터 루트 수준까지 각 피라미드 수준 당 하나씩.
A rating may contribute to up to H collaborative filtering models: one per each pyramid level starting from the lowest maintained grid cell containing the embedded user location up to the root level. 


피라미드의 루트 셀 (레벨 0)은 "전통적인"(즉, 비 공간적) 항목 기반 협업 필터링 모델을 나타냅니다.
Note that the root cell (level 0) of the pyramid represents a “traditional” (i.e., non-spatial) itembased collaborative filtering model. 


LARS는 지역 성과 확장 성의 절충안을 기반으로 주기적으로 셀을 병합하거나 분할하므로 피라미드의 레벨은 불완전 할 수 있습니다 (섹션 III-C에서 논의 됨).
Levels in the pyramid can be incomplete, as LARS will periodically merge or split cells based on trade-offs of locality and scalability (discussed in Section III-C). 


예를 들어 그림 4에서 수준 3의 오른쪽 위 모서리에있는 4 개의 셀은 유지되지 않습니다 (빈 흰색 사각형으로 표시됨).
For example, in Figure 4, the four cells in the upper right corner of level 3 are not maintained (depicted as blank white squares).


우리는 주어진 공간을 완전히 덮을 수있는 "공간 분할"구조이기 때문에 피라미드를 선택했습니다.
We chose to employ a pyramid as it is a “space-partitioning” structure that is guaranteed to completely cover a given space.


우리의 목적을 위해 "데이터 파티셔닝"구조 (예 : R- 트리)는 데이터 포인트를 인덱싱하고 주어진 공간을 완전히 커버하지 않을 수 있으므로 덜 이상적입니다.
For our purposes, “data-partitioning” structures (e.g., R-trees) are less ideal, as they index data points and are not guaranteed to completely cover a given space.

B. 쿼리 처리
B. Query Processing


사용자 위치 L과 제한 K가있는 추천 쿼리 (섹션 II-A에 설명 됨)가 주어지면 LARS는 두 가지 쿼리 처리 단계를 수행합니다.
Given a recommendation query (as described in Section II-A) with user location L and a limit K, LARS performs two query processing steps: 


(1) 사용자 위치 L은 L을 포함하는 적응 피라미드에서 가장 낮은 유지 관리 셀 C를 찾는 데 사용됩니다.
(1) The user location L is used to find the lowest maintained cell C in the adaptive pyramid that contains L. 


이것은 피라미드의 가장 낮은 수준에서 셀을 검색하기 위해 사용자 위치를 해싱함으로써 수행됩니다.
This is done by hashing the user location to retrieve the cell at the lowest level of the pyramid. 


이 셀이 유지되지 않으면 가장 가까운 유지 된 조상 셀을 반환합니다.
If this cell is not maintained, we return the nearest maintained ancestor cell. 


(2) C에 저장된 모델을 사용하여 항목 기반 협업 필터링 기술 (섹션 II-B에서 다룹니다)을 사용하여 Top-k 권장 항목을 생성합니다.
(2) The top-k recommended items are generated using the item-based collaborative filtering technique (covered in Section II-B) using the model stored at C. 


앞서 언급했듯이 C의 모델은 C 내의 사용자 위치와 관련된 공간 등급 만 사용하여 빌드됩니다.
As mentioned earlier, the model in C is built using only the spatial ratings associated with user locations within C.


기존 추천 쿼리 (즉, 스냅 샷 쿼리) 외에도 LARS는 연속 쿼리를 지원하며 다음과 같이 각 사용자의 영향 요구 사항을 고려할 수 있습니다.
In addition to traditional recommendation queries (i.e., snapshot queries), LARS also supports continuous queries and can account for the influence requirement for each user as follows.


연속 쿼리.
Continuous queries. 


LARS는 연속 쿼리가 실행되면 전체를 평가하고 초기 응답으로 사용자 U에게 권장 사항을 다시 보냅니다.
LARS evaluates a continuous query in full once it is issued, and sends recommendations back to a user U as an initial answer. 


LARS는 위치 업데이트를 사용하여 U의 움직임을 모니터링합니다.
LARS then monitors the movement of U using her location updates. 


U가 현재 그리드 셀의 경계를 넘지 않는 한, LARS는 초기 답변이 여전히 유효하기 때문에 아무것도하지 않습니다.
As long as U does not cross the boundary of her current grid cell, LARS does nothing as the initial answer is still valid. 


U가 셀 경계를 넘으면 LARS는 새 셀에 대한 권장 사항 쿼리를 재평가하고 마지막으로보고 된 답변에 증분 업데이트 [13] 만 보냅니다.
Once U crosses a cell boundary, LARS reevaluates the recommendation query for the new cell and only sends incremental updates [13] to the last reported answer. 


스냅 샷 쿼리와 마찬가지로 수준 h의 셀이 유지되지 않으면 쿼리는 피라미드에서 가장 가까운 유지 된 조상 셀로 일시적으로 상위 셀로 전송됩니다.
Like snapshot queries, if a cell at level h is not maintained, the query is temporarily transferred higher in the pyramid to the nearest maintained ancestor cell. 


더 높은 수준의 셀이 더 큰 공간 영역을 유지하므로 연속 쿼리가 공간 경계를 덜 자주 교차하므로 필요한 권장 사항 업데이트의 양이 줄어 듭니다.
Note that since higher-level cells maintain larger spatial regions, the continuous query will cross spatial boundaries less often, reducing the amount of required recommendation updates.


영향력 수준.
Influence level. 


LARS는 쿼리 사용자가 권장 사항에 영향을 미치는 데 사용되는 공간 이웃의 크기를 제어하는 선택적 영향 수준 (위치 L 및 제한 K 외에)을 지정할 수 있도록 허용하여 영향 요구 사항을 해결합니다.
LARS addresses the influence requirement by allowing querying users to specify an optional influence level (in addition to location L and limit K) that controls the size of the spatial neighborhood used to influence their recommendations. 


영향 수준 I은 피라미드 수준에 매핑되며 Google 또는 Bing지도 (예 : 도시 블록, 이웃, 전체 도시)에서 "확대 / 축소"수준과 매우 유사합니다.
An influence level I maps to a pyramid level and acts much like a “zoom” level in Google or Bing maps (e.g., city block, neighborhood, entire city). 


레벨 I은 LARS가 가장 낮은 유지 보수 그리드 셀 (기본값) 대신 레벨 I의 쿼리 사용자 위치를 포함하는 그리드 셀에서 시작하여 추천 쿼리를 처리하도록 지시합니다.
The level I instructs LARS to process the recommendation query starting from the grid cell containing the querying user location at level I, instead of the lowest maintained grid cell (the default). 


영향 수준이 0이면 LARS가 피라미드의 루트 셀을 사용하므로 기존 (비 공간) 협업 필터링 추천 시스템으로 작동합니다.
An influence level of zero forces LARS to use the root cell of the pyramid, and thus act as a traditional (non-spatial) collaborative filtering recommender system.



C. 데이터 구조 유지 관리
C. Data Structure Maintenance


이 섹션에서는 피라미드 데이터 구조를 만들고 유지하는 방법에 대해 설명합니다.
This section describes building and maintaining the pyramid data structure. 


처음에는 피라미드를 구축하기 위해 현재 시스템에있는 모든 위치 기반 등급을 사용하여 높이 H의 완전한 피라미드를 구축하여 모든 H 레벨의 모든 셀이 존재하고 협업 필터링 모델을 포함합니다.
Initially, to build the pyramid, all location-based ratings currently in the system are used to build a complete pyramid of height H, such that all cells in all H levels are present and contain a collaborative filtering model. 


초기 높이 H는 원하는 지역 수준에 따라 선택되며, 가장 낮은 피라미드 수준의 셀은 가장 지역화 된 영역을 나타냅니다.
The initial height H is chosen according to the level of locality desired, where the cells in the lowest pyramid level represent the most localized regions. 


이 초기 빌드 후, 우리는 가장 낮은 수준 h부터 시작하여 모든 셀을 스캔하고 허용되는 양의 양이 허용되는 것으로 결정되면 사분면 (즉, 공통 부모가있는 4 개의 셀)을 부모로 병합하는 병합 단계를 호출합니다. 지역 성은 손실되지 않습니다 (병합은 섹션 III-C1에서 논의 됨).
After this initial build, we invoke a merging step that scans all cells starting from the lowest level h and merges quadrants (i.e., four cells with a common parent) into their parent at level h − 1 if it is determined that a tolerated amount of locality will not be lost (merging is discussed in Section III-C1). 


원래 부분 피라미드 [11]는 정적 데이터에 대한 공간 쿼리와 관련이 있었지만 피라미드 유지 관리는 다루지 않았습니다.
We note that while the original partial pyramid [11] was concerned with spatial queries over static data, it did not address pyramid maintenance.


시간이 지남에 따라 새로운 사용자, 등급 및 항목이 시스템에 추가됩니다.
As time goes by, new users, ratings, and items will be added to the system. 


이 새로운 데이터는 피라미드 셀에서 유지되는 협업 필터링 모델의 크기를 증가시킬뿐만 아니라 각 셀에서 생성 된 권장 사항을 변경합니다.
This new data will both increase the size of the collaborative filtering models maintained in the pyramid cells, as well as alter recommendations produced from each cell.


이러한 변경 사항을 설명하기 위해 LARS는 셀 단위로 유지 관리를 수행합니다.
To account for these changes, LARS performs maintenance on a cell-by-cell basis. 


N % 새 등급을 받으면 셀 C에 대한 유지 관리가 트리거됩니다. 백분율은 C의 기존 등급 수에서 계산됩니다.
Maintenance is triggered for a cell C once it receives N% new ratings; the percentage is computed from the number of existing ratings in C. 


협업 필터링의 매력적인 품질은 모델이 성숙함에 따라 (즉, 모델을 구축하는 데 더 많은 데이터가 사용됨) 그로부터 생성 된 top-k 권장 사항을 크게 변경하기 위해 더 많은 업데이트가 필요하다는 것입니다 [14].
We do this because an appealing quality of collaborative filtering is that as a model matures (i.e., more data is used to build the model), more updates are needed to significantly change the top-k recommendations produced from it [14]. 


따라서 유지 관리가 덜 자주 필요합니다.
Thus, maintenance is needed less often. 

알고리즘 1은 LARS 유지 관리 알고리즘에 대한 의사 코드를 제공합니다.
Algorithm 1 provides the pseudocode for the LARS maintenance algorithm. 


이 알고리즘은 피라미드 셀 C와 레벨 h를 입력으로 취하며 모델 재 구축 및 병합 / 분할 유지 관리의 두 가지 주요 단계를 포함합니다.
The algorithm takes as input a pyramid cell C and level h, and includes two main steps: model rebuild and merge/split maintenance.


1 단계 : 모델 재 구축.
Step I: Model Rebuild. 


첫 번째 단계는 섹션 II-B (4 행)에 설명 된대로 셀 C에 대한 항목 기반 CF (협업 필터링) 모델을 다시 빌드하는 것입니다.
The first step is to rebuild the item-based collaborative filtering (CF) model for a cell C, as described in Section II-B (line 4). 


새로운 위치 기반 등급이 시스템에 입력됨에 따라 모델이 "진화"할 수 있도록 CF 모델을 재 구축해야합니다 (예 : 새 항목, 등급 또는 사용자에 대한 설명).
Rebuilding the CF model is necessary to allow the model to “evolve” as new locationbased ratings enter the system (e.g., accounting for new items, ratings, or users). 


CF 모델 구축 비용이 O (R2U) (섹션 II-B 당) 인 경우, 레벨 h에서 셀 C에 대한 모델 재 구축 비용은 (R / 4h) 2 (U / 4h) = R24hU입니다. 등급과 사용자가 균일하게 분배됩니다.
Given the cost of building the CF model is O(R2U) (per Section II-B), the cost of the model rebuild for a cell C at level h is (R/4h)2(U/4h)=R24hU, assuming ratings and users are uniformly distributed.


2 단계 : 병합 / 분할 유지 관리.
Step II: Merging/Split Maintenance. 


셀 C에 대한 CF 모델을 재 구축 한 후 LARS는 확장 성 및 지역성의 절충안을 기반으로 셀을 병합하거나 분할 할 수있는 병합 / 분할 유지 관리 단계를 호출합니다.
After rebuilding the CF model for cell C, LARS invokes a merge/split maintenance step that may decide to merge or split cells based on tradeoffs in scalability and locality. 


알고리즘은 먼저 C에 수준 h + 1 (6 행)로 유지되는 하위 사분면 q가 있는지, q의 4 개 셀 중 자신의 하위를 유지하지 않는 셀 (7 행)이 없는지 확인합니다.
The algorithm first checks if C has a child quadrant q maintained at level h + 1 (line 6), and that none of the four cells in q have maintained children of their own (line 7). 



두 경우 모두 유지되면 LARS는 q 사분면을 상위 셀 C에 병합 할 후보로 간주합니다 (8 행에서 CheckDoMerge 함수 호출).
If both cases hold, LARS considers quadrant q as a candidate to merge into its parent cell C (calling function CheckDoMerge on line 8). 

병합에 대한 자세한 내용은 섹션 III-C1에서 제공합니다.
We provide details of merging in Section III-C1. 


반면에 C에 수준 h + 1 (10 행)에서 유지되는 자식 사분면이없는 경우 LARS는 C를 수준 h + 1에서 4 개의 자식 셀로 분할하는 것을 고려합니다 (11 행에서 CheckDoSplit 함수 호출).
On the other hand, if C does not have a child quadrant maintained at level h + 1 (line 10), LARS considers splitting C into four child cells at level h+ 1 (calling function CheckDoSplit on line 11). 


분할 작업은 섹션 III-C2에서 다룹니다.
The split operation is covered in Section III-C2. 


병합 및 분할은 사분면에서 완전히 수행됩니다 (즉, 동일한 상위를 가진 4 개의 등가 셀).
Merging and splitting are performed completely in quadrants (i.e., four equi-area cells with the same parent). 


우리는 부분 피라미드를 간단하게 유지하기 위해이 결정을 내 렸습니다.
We made this decision for simplicity in maintaining the partial pyramid. 


그러나 우리는 또한 (섹션 III-D에서) 사분면보다 더 미세한 단위로 병합 및 분할하여이 제약 조건을 완화하는 방법에 대해 논의합니다.
However, we also discuss (in Section III-D) relaxing this constraint by merging and splitting at a finer granularity than a quadrant.

피라미드 유지 관리의 다음 기능에 주목합니다.
We note the following features of pyramid maintenance: 

(1) 유지 관리는 완전히 오프라인으로 수행 할 수 있습니다. 즉, LARS는 피라미드의 일부가 업데이트되는 동안 "이전"피라미드 셀을 사용하여 계속 권장 사항을 생성 할 수 있습니다.
(1) Maintenance can be performed completely offline, i.e., LARS can continue to produce recommendations using the ”old” pyramid cells while part of the pyramid is being updated; 


(2) 유지 보수는 전체 피라미드를 한 번에 재건하는 것이 아니라 한 번에 하나의 셀만 재건됩니다.
(2) maintenance does not entail rebuilding the whole pyramid at once, instead, only one cell is rebuilt at a time; 


(3) 유지 보수는 피라미드 셀에 N %의 새로운 등급이 추가 된 후에 만 수행됩니다. 즉, 유지 보수는 많은 작업에 대해 상환됩니다.
(3) main-tenance is performed only after N% new ratings are added to a pyramid cell, meaning maintenance will be amortized over many operations.


1) 셀 병합 : 병합은 레벨 h에서 셀의 전체 사분면을 버리고 레벨 h-1에서 공통 부모를 사용하는 것을 수반합니다.
1) Cell Merging: Merging entails discarding an entire quadrant of cells at level h with a common parent at level h−1. 


병합은 병합 된 셀의 항목 기반 CF (협업 필터링) 모델을 폐기하여 저장소를 줄이므로 LARS의 확장 성 (예 : 저장소 및 계산 오버 헤드)을 향상시킵니다.
Merging improves scalability (i.e., storage and computational overhead) of LARS, as it reduces storage by discarding the item-based collaborative filtering (CF) models of the merged cells. 


또한 병합은 두 가지 방식으로 계산 오버 헤드를 개선합니다.
Furthermore, merging improves computational overhead in two ways: 

(a) 주기적으로 재 구축되는 CF 모델이 적기 때문에 유지 보수 계산이 적습니다.
(a) less maintenance computation, since less CF models are periodically rebuilt, and 


(b) 병합 된 셀이 더 큰 공간 영역을 나타 내기 때문에 덜 연속적인 쿼리 처리 계산을 수행하므로 사용자는 셀 경계를 넘어가는 빈도가 낮아 추천 업데이트가 덜 발생합니다.
(b) less continuous query processing computation, as merged cells represent a larger spatial region, hence, users will cross cell boundaries less often triggering less recommendation updates. 


병합 된 셀은 더 넓은 공간 영역에서 커뮤니티 의견을 캡처하여 작은 셀보다 덜 고유 한 (즉, "로컬") 권장 사항을 생성하므로 지역성이 손상됩니다.
Merging hurts locality, since merged cells capture community opinions from a wider spatial region, causing less unique (i.e., “local”) recommendations than smaller cells.


사분면 q를 부모 셀 CP (즉, 알고리즘 1의 8 행에있는 CheckDoMerge 함수)에 병합할지 여부를 결정하기 위해 두 가지 백분율 값을 계산합니다.
To determine whether to merge a quadrant q into its parent cell CP (i.e., function CheckDoMerge on line 8 in Algorithm 1), we calculate two percentage values: 


(1) 지역성 손실, (잠재적으로) 병합에 의해 손실 된 지역성 양,
(1) locality loss, the amount of locality lost by (potentially) merging, and 

(2) 확장 성 이득, (잠재적으로) 병합을 통해 얻은 확장 성의 양.
(2) scalability gain, the amount of scalability gained by (potentially) merging. 



이러한 백분율 계산에 대한 자세한 내용은 다음에 설명합니다.
Details of calculating these percentages are covered next. 


병합을 결정할 때 확장 성 이득과 지역성 손실 간의 균형을 정의하는 [0,1] 범위의 실수 인 시스템 매개 변수 M을 정의합니다.
When deciding to merge, we define a system parameter M, a real number in the range [0,1] that defines a tradeoff between scalability gain and locality loss. 


다음과 같은 경우 LARS가 병합됩니다 (즉, q 사분면 삭제).
LARS merges (i.e., discards quadrant q) if: 


(1 − M) ∗ 확장 성 이득> M ∗ 지역성 손실
(삼)
M 값이 작을수록 확장 성 확보가 중요하며 시스템은 작은 확장 성 향상을 위해 많은 양의 지역성을 잃을 수 있습니다.
(1 − M) ∗ scalability gain > M ∗ locality loss 
(3)
A smaller M value implies gaining scalability is important and the system is willing to lose a large amount of locality for small gains in scalability. 


반대로 M 값이 클수록 확장 성은 문제가되지 않으며 병합하려면 손실 된 지역성이 작아야합니다.
Conversely, a larger M value implies scalability is not a concern, and the amount of locality lost must be small in order to merge. 


극단적으로 M = 0 (즉, 항상 병합)으로 설정하면 LARS가 기존 CF 추천 시스템으로 작동 함을 의미하고 M = 1로 설정하면 LARS가 병합되지 않습니다. 즉, LARS는 모든 셀을 전혀 유지하는 완전한 피라미드 구조를 사용합니다. 수준.
At the extremes, setting M=0 (i.e., always merge) implies LARS will function as a traditional CF recommender system, while setting M=1 causes LARS to never merge, i.e., LARS will employ a complete pyramid structure maintaining all cells at all levels.

지역성 손실 계산. 
Calculating Locality Loss. 

셀 사분면 q를 버릴 때 추천 고유성의 손실을 관찰하고 부모 셀 CP를 사용하여 대신 추천을 생성하여 지역성 손실을 계산합니다.
We calculate locality loss by observing the loss of recommendation uniqueness when discarding a cell quadrant q and using its parent cell CP to produce recommendations in its place. 


이 계산은 세 단계로 수행됩니다.
We perform this calculation in three steps. 


(1) 샘플.
(1) Sample. 


CP 내에서 적어도 하나의 등급을 가진 다양한 시스템 사용자 U의 샘플을 취합니다 (그리고 정의에 따라 더 지역화 된 셀 Cu ∈ q 중 하나).
We take a sample of diverse system users U that have at least one rating within CP (and by definition one of the more localized cells Cu ∈ q).


공간상의 이유로 사용자 샘플링에 대해서는 자세히 논의하지 않지만 직관은 각 사용자의 평가 이력을 비교하여 다양한 취향을 가진 사용자 집합을 선택하는 것입니다.
Due to space, we do not discuss user sampling in detail, however, the intuition is to select a set of users with diverse tastes by comparing each user’s rating history. 


계산에 사용자 벡터 (항목 벡터 대신)를 사용한다는 점을 제외하면 방정식 2와 동일한 방식으로 사용자 간의 코사인 거리를 사용하여 다양성을 측정합니다.
We measure diversity using the Cosine distance between users in the same manner as Equation 2, except we employ user vectors in the calculation (instead of item vectors). 

(2) 비교합니다.
(2) Compare. 


각 사용자 u ∈ U에 대해 병합 된 셀 CP (즉, 부모)에서 생성 된 top-k 추천 RP 목록과 사용자가 more에서받는 추천 Ru 목록을 비교하여 추천 고유성 손실 가능성을 측정합니다. 국소화 된 세포 Cu ∈ q.
For each user u ∈ U, we measure the potential loss of recommendation uniqueness by comparing the list of top-k recommendations RP produced from the merged cell CP (i.e., the parent) with the list of recommendations Ru that the user receives from the more localized cell Cu ∈ q. 


공식적으로, 고유성의 손실은 비율 | Ru-RP | k는 Ru에는 나타나지만 상위 권장 사항 RP에는 나타나지 않는 권장 항목 수를 나타내며 총 권장 개체 수로 정규화됩니다.
Formally, the loss of uniqueness can be computed as the ratio |Ru−RP | k, which indicates the number of recommended items that appear in Ru but not in the parent recommendation RP , normalized to the total number of recommended objects k. 


(3) 평균.
(3) Average. 


U의 모든 사용자에 대한 고유성의 평균 손실을 계산하여 지역성 손실이라고하는 단일 백분율 값을 생성합니다.
We calculate the average loss of uniqueness over all users in U to produce a single percentage value, termed locality loss. 


확장 성 이득 계산.
Calculating scalability gain. 


확장 성 이득은 스토리지 및 계산 절감으로 측정됩니다.
Scalability gain is measured in storage and computation savings. 









![Fig4](./image/Fig4.PNG)  
![Fig5](./image/Fig5.PNG)  
--- 
### 4. NON-SPATIAL USER RATINGS FOR SPATIAL ITEMS
### 5. SPATIAL USER RATINGS FOR SPATIAL ITEMS
### 6. EXPERIMENTS
![Fig6](./image/Fig6.PNG)  
![Fig7](./image/Fig7.PNG)  
![Fig8](./image/Fig8.PNG)  
![Fig9](./image/Fig9.PNG)  
![Fig10](./image/Fig10.PNG)  
--- 

--- 
