## 2012_IEEE_LARS [LARS: A Location-Aware Recommender System]

![main](./image/main.PNG)

---
### Abstract  
* LARS: 위치 인식 추천 시스템(Location-Aware Recommender System)  
* 위치 기반 rating(3):
  * **3.** 비 공간 항목에 대한 공간 등급(spatial ratings for non-spatial items)  
  * **4.** 공간 항목에 대한 비 공간 등급(non-spatial ratings for spatial items)   
  * **5.** 공간 항목에 대한 공간 등급(spatial ratings for spatial items)  
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

#### B. Item-Based Collaborative Filtering  
* ![Fig2](./image/Fig2.PNG)  

##### Phase I: Model Building. 
##### Phase II: Recommendation Generation. 

###### Computing Similarity. 
  * Cosine similarity

![Fig3](./image/Fig3.PNG)  

--- 

### 3. SPATIAL USER RATINGS FOR NON-SPATIAL ITEMS
* tuple(user, **ulocation**, rating, item)  

* 선호 지역성: 사용자 의견이 공간적으로 고유함  

* 요구사항(3):  
  * (1) 지역성: 공간적 이웃 등급에 따라 추천 항목 영향  
  * (2) 확장성: 추천 절차/데이터 구조 사용자 확장  
  * (3) 영향:   영향을 미치는 공간적 이웃 크기 제어 가능  

* 요구사항 충족:  
  * 사용자 분할 기술: 적응형 피라미드 구조(지역성, 확장성, 영향)  

* 적응형 피라미드 구조
  * ulocation: 공간 영역 적응적으로 분할   
  * (쿼리 사용자 공간 영역 내) user, rating, item: CF  

#### A. Data Structure

* ![Fig4](./image/Fig4.PNG)  
  * quad-tree  
  * H 레벨: 피라미드 높이로 분해  
    * <img src="https://latex.codecogs.com/gif.latex?4%5Eh"> 동일 면적 그리드 셀로 분할(등가 영역 셀 *cid*)  
  * 셀: (공간정보)사용자 위치 + 공간 등급 -> item 기반 CF 모델 저장  
  * level0(root cell): 전통적인(비 공간적) 항목 기반 CF 모델  
  * 빈 공간: 지역성/확장성 절충 > 셀 병합/분할 > 피라미드의 레벨 불완전  

#### B. Query Processing
* 쿼리 처리 단계(2):  
  * (1) L(사용자 위치): L 포함, 적응 피라미드 가장 낮은 maintained 셀 C 찾음, 해싱 이용    
    * 유지되지 않으면 가장 가까운 ancestor 셀 반환  
  * (2) C(셀): 저장된 모델 사용, 항목 기반 CF > Top-k 추천 항목 생성  
* 스냅샷 쿼리

##### Continuous queries. 
* 연속 쿼리 실행: 전체 평가 > 초기 응답으로 사용자 U에게 추천 재전송  
  * 위치 업데이트: U의 움직임 모니터링  
    * 셀의 경계를 넘지 않으면, 초기 답변 유효
    * 셀의 경계를 넘으면, 새 셀에 대한 추천 쿼리를 재평가 > 마지막 답변 ± 업데이트  

##### Influence level. 
* I 영향 수준: 추천 영향 공간 이웃 크기 제어, 피라미드 수준에 매핑(지도 확대/축소)  
  * 선택적 영향 수준(위치 L, 제한 K) 제외  

#### C. Data Structure Maintenance
* 피라미드 데이터 구조
  * 초기 구축: 모든 위치 기반 등급 사용 > 높이 H의 완전 피라미드 구축 > H 레벨 모든 셀 존재(CF 포함)  
  * 병합 단계: 가장 낮은 수준 h부터 모든 셀 스캔 > 사분면 부모로 병합(지역성 손실 X)  
  * 유지 관리: 새로운 트리플로 인한 업데이트 > 셀 단위 유지 관리(트리거)    

<br>

* ![A1](./image/A1.PNG)  
  * 입력: 셀 C, 레벨 h  
  * Step I: 모델 재구축 및 병합(Model Rebuild)  
    * 셀C에 대한 항목 기반 CF 재빌드  
  * Step II 분할 유지 관리(Merging/Split Maintenance)  
    * 확장성/지역성 절충 > 셀 병합/분할 

* 피라미드 maintenance features:  
  * (1) 완전히 오프라인 수행  
    * 피라미드의 일부 업데이트되는 동안 "이전" 피라미드 셀을 사용 가능  
  * (2) 전체 피라미드를 한 번에 재건 X, 한 번에 하나의 셀만 재건  
  * (3) 피라미드 셀에 N%의 새로운 등급이 추가 된 후에만 수행
  
##### 1) 셀 병합: 레벨 h에서 셀의 전체 사분면을 버리고, 레벨 h-1에서 공통 부모를 사용하는 것
* 병합 된 셀의 항목 기반 CF 모델 폐기 > 저장소 ↓ > 확장성 ↑  
* 더 넓은 공간 영역 캡처 > 덜 고유 한(local) 추천 생성 => 지역성 손상  
* 백분율 계산: 병합 여부 결정  
  * ![(3)](./image/(3).PNG)  
    * M: 확장성 이득, 지역성 손실 균형을 정의 실수 시스템 매개변수[0,1]  
    * LARS 병합(q 사분면 삭제)  
      * M=0, 기존 CF    
      * M=1, 병합 X  

###### Calculating Locality Loss.  
* 셀 사분면 q를 버릴 때, 고유성의 손실  
* 부모 셀 CP를 사용 > 추천 생성 > 지역성 손실 계산  
* 단계(3): Sample > Compare > Average => 단일 백분율 값을 생성  

###### Calculating scalability gain. 
* 확장성 이득: 스토리지 및 계산 절감  
  * 스토리지 이득(비율) = 병합된 각 자식 셀 크기 합산 / 부모 셀 크기 합  

###### Cost. 
* 모델 재구축 단계보다 적음  

###### Example. 
* ![Fig5](./image/Fig5.PNG)  
  * level h: 병합 후보 셀 C1~C4  
  * level h-1: 부모 CP로 병합  

  * 지역성 손실  
    * 샘플링 된 User U1~U4: 위치 자식 Cu /부모 Cp 셀 각각 표시  
        * U1 C1 추천 = {I1, I2, I5, I6}
        * U1 Cp 추천 = {I1, I2, I5, I7}
        * 병합: I6 손실, 지역성 손실 = 25%   
        * U1~U4 지역성 손실 평균 = 25%  

  * 확장성 이득   
    * C1+C2+C3+C4+Cp = 4GB (가정)  
    * C1+C2+C3+C4 = 2GB (가정)  
    * 확장성 이득 = 2/4 = 50%   

  * LARS  
    * M = 0.7 (가정)   
    * (0.3 * 50) < (0.7 * 25) => C1, C2, C3, C4 셀을 CP로 병합하지 않음  

##### 2) 분할: 레벨 h-1의 셀 아래에 피라미드 레벨 h에서 새로운 셀 사분면 생성
* 분할: 지역성 ↑ , 확장성 ↓, 연속 쿼리 처리 부정적인 영향  

* 분할 여부 결정(CheckDoSplit)    
  * ![(4)](./image/(4).PNG)   

###### Speculative splitting 투기적 분할   
* 4개의 셀의 CF 모델 처음부터 구축 > 지역성 이득 / 확장성 손실을 측정  
* LARS 분할 결정 > 모든 등급 사용, 새로 분할 된 셀에 대한 전체 모델 빌드  

###### Calculating locality gain. 
* *1) 셀 병합* 동일  

###### Calculating scalability loss. 
* 확장성 손실 메트릭 = 예상 크기 합계 / 기존 상위 셀 크기의 합계  

###### Cost. 
*  지역성 이득 + 확장성 손실  

###### Example. 
* *1) 셀 병합* 동일 상황   
* M = 0.7, (0.7 * 25) > (0.3 * 50) => 지역성 이득 >> 확장성 손실 => Cp 4개 셀로 분할  

#### D. Partial Merging and Splitting
##### 1) 부분 병합: 확장성 유지, 더 적은 지역성을 희생, 더 세분화 된 수준에서 부분 병합
* ![Fig5](./image/Fig5.PNG)  
  * C3, C4 그대로 두고, C1, C2 셀만 병합 
    * 병합 된 후보 셀 C12가 부모셀 역할(지역성 손실: CF 초기 빌드 => 추가 오버헤드)  
    * 저장 이득 계산 시 C12 제외  

##### 2) 부분 분할: 모든 기술은 동일 유지  
* 구별 가능한 경우(2):  
  * level h 부모가 level h + 1에서 4 개 미만의 셀로 분할  
    * 부분 자식 셀 인식 위한 예측 분할 필요  
  * level h 셀을 개별 셀로 분할, level h + 1 셀은 생성 X  
    * 원래 셀 사분면 줄인 이전 부분 병합 발생  

--- 

### 4. NON-SPATIAL USER RATINGS FOR SPATIAL ITEMS

이 섹션에서는 LARS가 튜플 (사용자, 등급, 항목, ilocation)로 표시되는 공간 항목에 대한 비 공간 등급을 사용하여 권장 사항을 생성하는 방법을 설명합니다.
This section describes how LARS produces recommendations using non-spatial ratings for spatial items represented by the tuple (user, rating, item, ilocation). 


아이디어는 여행 지역성을 활용하는 것입니다. 즉, 사용자가 여행 거리를 기반으로 공간 장소 선택을 제한한다는 관찰 (섹션 I-A의 분석을 기반으로 함)입니다.
The idea is to exploit travel locality, i.e., the observation that users limit their choice of spatial venues based on travel distance (based on analysis in Section I-A). 


전통적인 (비 공간적) 추천 기술은 부담스러운 이동 거리 (예 : 수백 마일 거리)로 추천을 생성 할 수 있습니다.
Traditional (non-spatial) recommendation techniques may produces recommendations with burdensome travel distances (e.g., hundreds of miles away). 


LARS는 질의하는 사용자로부터 이동 거리가 멀어 질수록 항목의 추천 순위에 페널티를주는 기술인 이동 패널티를 사용하여 합리적인 이동 거리 내에서 추천을 생성합니다.
LARS produces recommendations within reasonable travel distances by using travel penalty, a technique that penalizes the recommendation rank of items the further in travel distance they are from a querying user. 


여행 패널티는 각 항목까지의 여행 거리를 계산하여 값 비싼 계산 오버 헤드를 초래할 수 있습니다.
Travel penalty may incur expensive computational overhead by calculating travel distance to each item. 


따라서 LARS는 조기 종료가 가능한 효율적인 쿼리 처리 기술을 사용하여 모든 항목에 대한 이동 거리를 계산하지 않고 추천을 생성합니다.
Thus, LARS employs an efficient query processing technique capable of early termination to produce the recommendations without calculating the travel distance to all items. 


섹션 IV-A는 쿼리 처리 프레임 워크를 설명하고 섹션 IV-B는 이동 거리 계산을 설명합니다.
Section IV-A describes the query processing framework while Section IV-B describes travel distance computation.

A. 쿼리 처리
#### A. Query Processing

여행 페널티 기법을 사용하는 공간 항목에 대한 쿼리 처리는 계산 된 RecScore (u, i)를 기반으로 쿼리 사용자 u에 대한 각 공간 항목 i의 순위를 매김으로써 단일 시스템 전체 항목 기반 협업 필터링 모델을 사용하여 Top-k 권장 사항을 생성합니다. 같이:
![(5)](./image/(5).PNG)  
P (u, i)는 사용자 u에 대한 항목 i의 표준 항목 기반 CF 예측 등급입니다 (섹션 II-B 참조).
Query processing for spatial items using the travel penalty technique employs a single system-wide item-based collaborative filtering model to generate the top-k recommendations by ranking each spatial item i for a querying user u based on RecScore(u, i), computed as:
(5)
P(u, i) is the standard item-based CF predicted rating of item i for user u (see Section II-B). 


T ravelP enalty (u, i)는 평가 척도와 동일한 값 범위 (예 : [0, 5])로 정규화 된 u와 i 사이의 도로 네트워크 이동 거리입니다.
T ravelP enalty(u, i) is the road network travel distance between u and i normalized to the same value range as the rating scale (e.g., [0, 5]). 


권장 사항을 처리 할 때 모든 후보 항목에 대해 Equation 5를 계산하여 Top-k 권장 사항을 찾는 것을 피하는 것을 목표로합니다. 이는 이동 거리를 계산할 필요가있을 때 상당히 비쌀 수 있습니다.
When processing recommendations, we aim to avoid calculating Equation 5 for all candidate items to find the top-k recommendations, which can become quite expensive given the need to compute travel distances. 


이러한 계산을 피하기 위해 우리는 단조롭게 증가하는 이동 패널티 순서 (즉, 이동 거리)로 항목을 평가하여 top-k 쿼리 처리 [15], [16], [17]의 조기 종료 원칙을 사용할 수 있습니다.
To avoid such computation, we evaluate items in monotonically increasing order of travel penalty (i.e., travel distance), enabling us to use early termination principles from top-k query processing [15], [16], [17]. 


이제 쿼리 처리 알고리즘의 주요 아이디어를 제시하고 다음 섹션에서는 이동 거리의 증가 순서로 이동 패널티를 계산하는 방법에 대해 설명합니다.
We now present the main idea of our query processing algorithm and in the next section discuss how to compute travel penalties in an increasing order of travel distance. 


알고리즘 2는 쿼리 사용자 ID U, 위치 L 및 제한 K를 입력으로 사용하고 상위 k 개 권장 항목의 목록 R을 반환하는 쿼리 처리 알고리즘의 의사 코드를 제공합니다.
Algorithm 2 provides the pseudo code of our query processing algorithm that takes a querying user id U, a location L, and a limit K as input, and returns the list R of top-k recommended items. 
![A2](./image/A2.PNG)  

알고리즘은 knearest-neighbor 알고리즘을 실행하여 이동 패널티가 가장 낮은 k 항목으로 목록 R을 채우는 것으로 시작합니다. R은 방정식 5를 사용하여 계산 된 추천 점수로 정렬됩니다.
The algorithm starts by running a knearest-neighbor algorithm to populate the list R with k items with lowest travel penalty; R is sorted by the recommendation score computed using Equation 5. 


이 초기 부분은 최저 추천 점수 값 (LowestRecScore)을 R에서 k 번째 항목의 RecScore (3 ~ 8 행)로 설정하여 마무리됩니다.
This initial part is concluded by setting the lowest recommendation score value (LowestRecScore) as the RecScore of the k th item in R (Lines 3 to 8).


그런 다음 알고리즘은 페널티 점수 순서대로 항목을 하나씩 검색하기 시작합니다.
Then, the algorithm starts to retrieve items one by one in the order of their penalty score. 


이는 다음 섹션에서 설명하는 바와 같이 증분 k- 최근 접 이웃 알고리즘을 사용하여 수행 할 수 있습니다.
This can be done using an incremental k-nearest-neighbor algorithm, as will be described in the next section. 

각 항목 i에 대해 시스템에서 가능한 최대 등급 값인 MAX RATING에서 i의 이동 패널티를 빼서 가질 수있는 최대 추천 점수를 계산합니다 (예 : 5 (12 행)).
For each item i, we calculate the maximum possible recommendation score that i can have by subtracting the travel penalty of i from MAX RATING, the maximum possible rating value in the system, e.g., 5 (Line 12). 


이 최대 가능한 점수로 top-k 추천 항목 목록에 들어갈 수없는 경우 더 많은 항목에 대한 추천 점수 (및 이동 거리)를 계산하지 않고 R을 top-k 권장 사항으로 반환하여 즉시 알고리즘을 종료합니다 (13 행). ~ 15).
If i cannot make it into the list of top-k recommended items with this maximum possible score, we immediately terminate the algorithm by returning R as the top-k recommendations without computing the recommendation score (and travel distance) for more items (Lines 13 to 15). 


여기서의 근거는 우리가 페널티 순서가 높은 항목을 검색하고 나머지 항목이 가질 수있는 최대 점수를 계산하기 때문에 처리되지 않은 항목이 R에서 가장 낮은 권장 점수를 이길 가능성이 없다는 것입니다.
The rationale here is that since we are retrieving items in increasing order of their penalty and calculating the maximum score that any remaining item can have, then there is no chance that any unprocessed item can beat the lowest recommendation score in R. 


조기 종료 사례가 발생하지 않으면 식 5를 사용하여 각 항목 i에 대한 점수를 계속 계산하고 점수별로 정렬 된 R에 i를 삽입 (필요한 경우 k 번째 항목 제거) 그에 따라 가장 낮은 권장 값을 조정합니다 (줄 16-20).
If the early termination case does not arise, we continue to compute the score for each item i using Equation 5, insert i into R sorted by its score (removing the k th item if necessary), and adjust the lowest recommendation value accordingly (Lines 16 to 20).


여행 패널티는 유지 보수가 거의 필요 없습니다.
Travel penalty requires very little maintenance. 


필요한 유일한 유지 관리는 시스템에 들어오는 새로운 위치 기반 등급을 설명하기 위해 단일 시스템 전체 항목 기반 협업 필터링 모델을 가끔 다시 빌드하는 것입니다.
The only maintenance necessary is to occasionally rebuild the single system-wide item-based collaborative filtering model in order to account for new location-based ratings that enter the system. 


섹션 III-C에서 논의 된 추론에 따라 N %의 새로운 위치 기반 등급을받은 후 모델을 재 구축합니다.
Following the reasoning discussed in Section III-C, we rebuild the model after receiving N% new location-based ratings.


B. 증분 출장 페널티 계산
#### B. Incremental Travel Penalty Computation  

이 섹션에서는 LARS에서 구현 한 두 가지 방법에 대한 개요를 제공하여 여행 벌금에 따라 항목을 하나씩 점진적으로 검색합니다.
This section gives an overview of two methods we implemented in LARS to incrementally retrieve items one by one ordered by their travel penalty. 


두 가지 방법은 쿼리 처리 효율성과 페널티 정확도 사이의 균형을 보여줍니다.
The two methods exhibit a tradeoff between query processing efficiency and penalty accuracy: 


(1) 정확한 여행 벌칙을 제공하지만 계산 비용이 많이 드는 온라인 방법
(1) an online method that provides exact travel penalties but is expensive to compute, and 


(2) 덜 정확하지만 페널티 검색에 효율적인 오프라인 휴리스틱 방법.
(2) an offline heuristic method that is less exact but efficient in penalty retrieval. 


두 방법 모두 알고리즘 2의 11 행에서 서로 바꿔서 사용할 수 있습니다.
Both methods can be employed interchangeably in Line 11 of Algorithm 2.


1) 증분 KNN : 정확한 온라인 방법 : 항목 i에 대한 사용자 u에 대한 정확한 이동 패널티를 계산하기 위해 증분 k- 최근 접 이웃 (KNN) 기법 [18], [19], [20]을 사용합니다.
1) Incremental KNN: An Exact Online Method: To calculate an exact travel penalty for a user u to item i, we employ an incremental k-nearest-neighbor (KNN) technique [18], [19], [20]. 

사용자 위치 l이 주어지면 증분 KNN 알고리즘은 각 호출에서 이동 거리 d와 관련하여 u에 가장 가까운 다음 항목 i를 반환합니다.
Given a user location l, incremental KNN algorithms return, on each invocation, the next item i nearest to u with regard to travel distance d. 


우리의 경우 거리 d를 등급 척도로 정규화하여 방정식 5의 이동 패널티를 얻습니다.
In our case, we normalize distance d to the ratings scale to get the travel penalty in Equation 5. 


증분 KNN 기술은 유클리드 거리 [19]와 (도로) 네트워크 거리 [18], [20] 모두에 대해 존재합니다.
Incremental KNN techniques exist for both Euclidean distance [19] and (road) network distance [18], [20]. 


증분 KNN 기술을 사용하는 장점은 쿼리하는 사용자의 위치와 각 추천 후보 항목 사이의 정확한 이동 거리를 제공한다는 것입니다.
The advantage of using Incremental KNN techniques is that they provide an exact travel distances between a querying user’s location and each recommendation candidate item. 


단점은 거리가 쿼리 런타임에 온라인으로 계산되어야하므로 비용이 많이들 수 있다는 것입니다.
The disadvantage is that distances must be computed online at query runtime, which can be expensive. 

예를 들어, 유클리드 공간에서 증분 KNN을 사용하여 단일 항목을 검색하는 런타임 복잡성은 다음과 같습니다.
For instance, the runtime complexity of retrieving a single item using incremental KNN in Euclidean space is [19]: 


O (k + logN), 여기서 N 및 k는 각각 지금까지 검색된 총 항목 및 항목 수입니다.
O(k+logN), where N and k are the number of total items and items retrieved so far, respectively. 


2) 페널티 그리드 : 휴리스틱 오프라인 방법 : 여행 페널티를 점진적으로 검색하는 더 효율적이지만 덜 정확한 방법은 미리 계산 된 페널티 그리드를 사용하는 것입니다.
2) Penalty Grid: A Heuristic Offline Method: A more efficient, yet less accurate method to retrieve travel penalties incrementally is to use a pre-computed penalty grid. 


아이디어는 n × n 그리드를 사용하여 공간을 분할하는 것입니다.
The idea is to partition space using an n × n grid. 


각 그리드 셀 c는 크기가 같고 위치가 c로 정의 된 공간 영역 내에있는 모든 항목을 포함합니다.
Each grid cell c is of equal size and contains all items whose location falls within the spatial region defined by c. 


각 셀 c는 c 내의 어느 곳에서나 그리드의 다른 모든 n 2-1 대상 셀로 이동하기 위해 미리 계산 된 페널티 값을 저장하는 페널티 목록을 포함합니다. 이는 대상 그리드 셀 내의 모든 항목이 동일한 패널티 값을 공유 함을 의미합니다.
Each cell c contains a penalty list that stores the pre-computed penalty values for traveling from anywhere within c to all other n 2−1 destination cells in the grid; this means all items within a destination grid cell share the same penalty value. 


c에 대한 페널티 목록은 페널티 값별로 정렬되며 항상 페널티가 0 인 첫 번째 항목으로 c (자체)를 저장합니다.
The penalty list for c is sorted by penalty value and always stores c (itself) as the first item with a penalty of zero. 


항목을 점진적으로 검색하기 위해 쿼리하는 사용자를 포함하는 셀 내의 모든 항목은 패널티가 없으므로 순서에 관계없이 하나씩 반환됩니다.
To retrieve items incrementally, all items within the cell containing the querying user are returned one-by-one (in any order) since they have no penalty. 


이러한 항목이 모두 소진되면 패널티 목록의 다음 셀에 포함 된 항목이 반환되며 알고리즘 2가 조기에 종료되거나 모든 항목을 처리 할 때까지 계속됩니다.
After these items are exhausted, items contained in the next cell in the penalty list are returned, and so forth until Algorithm 2 terminates early or processes all items.

페널티 그리드를 채우려면 각 셀에서 그리드의 다른 모든 셀로 이동할 때 페널티 값을 계산해야합니다.
To populate the penalty grid, we must calculate the penalty value for traveling from each cell to every other cell in the grid. 


항목과 사용자가 도로망에 제한되어 있다고 가정하지만 결과없이 유클리드 공간을 사용할 수도 있습니다.
We assume items and users are constrained to a road network, however, we can also use Euclidean space without consequence. 


단일 소스 셀 c에서 대상 셀 d까지의 패널티를 계산하려면 먼저 c 내의 모든 항목에서 d 내의 모든 항목 대상까지 이동할 평균 거리를 찾습니다.
To calculate the penalty from a single source cell c to a destination cell d, we first find the average distance to travel from anywhere within c to all item destinations within d. 


이를 위해 (1) 둘 다 c 내의 도로 네트워크 세그먼트에 있고 (2) c의 중심에 가능한 한 가깝게있는 c 내의 앵커 포인트 p를 생성합니다.
To do this, we generate an anchor point p within c that both (1) lies on the road network segment within c and (2) lies as close as possible to the center of c. 


이러한 기준으로 p는 c에서 d로 이동하는 대략적인 평균 "시작점"역할을합니다.
With these criteria, p serves as an approximate average “starting point” for traveling from c to d. 


그런 다음 p에서 도로망의 d에 포함 된 모든 항목까지의 최단 경로 거리를 계산합니다 (최단 경로 알고리즘을 사용할 수 있음).
We then calculate the shortest path distance from p to all items contained in d on the road network (any shortest path algorithm can be used). 


마지막으로 계산 된 모든 최단 경로 거리를 c에서 d까지 평균합니다.
Finally, we average all calculated shortest path distances from c to d. 


마지막 단계로 c에서 d까지의 평균 거리를 정격 값 범위에 포함하도록 정규화합니다.
As a final step, we normalize the average distance from c to d to fall within the rating value range. 


등급 도메인은 일반적으로 작지만 (예 : 0 ~ 5) 거리는 마일 또는 킬로미터 단위로 측정되며 수식 5에 큰 영향을 미치는 큰 값을 가질 수 있으므로 정규화가 필요합니다.
Normalization is necessary as the rating domain is usually small (e.g., zero to five), while distance is measured in miles or kilometers and can have large values that heavily influence Equation 5. 

전체 패널티 그리드를 채우기 위해 각 셀에 대해이 전체 프로세스를 다른 모든 셀에 반복합니다.
We repeat this entire process for each cell to all other cells to populate the entire penalty grid.


새 항목이 시스템에 추가되면 셀 d에 해당 항목이 있으면 각 소스 셀 c에 대한 패널티 계산에 사용되는 평균 거리 값이 변경 될 수 있습니다.
When new items are added to the system, their presence in a cell d can alter the average distance value used in penalty calculation for each source cell c. 


따라서 N 개의 새 항목이 시스템에 입력 된 후 패널티 그리드에서 패널티 점수를 다시 계산합니다.
Thus, we recalculate penalty scores in the penalty grid after N new items enter the system.


우리는 공간 항목이 상대적으로 정적이라고 가정합니다. 예를 들어 레스토랑은 위치를 자주 변경하지 않습니다.
We assume spatial items are relatively static, e.g., restaurants do not change location often. 


따라서 기존 항목이 셀 위치를 변경하고 결과적으로 패널티 점수를 변경할 가능성은 거의 없습니다.
Thus, it is unlikely existing items will change cell locations and in turn alter penalty scores.

---

### 5. SPATIAL USER RATINGS FOR SPATIAL ITEMS
V. 공간 항목에 대한 공간 사용자 등급

이 섹션에서는 LARS가 튜플 (user, ulocation, rating, item, ilocation)로 표시되는 공간 항목에 대한 공간 등급을 사용하여 권장 사항을 생성하는 방법을 설명합니다.
This section describes how LARS produces recommendations using spatial ratings for spatial items represented by the tuple (user, ulocation, rating, item, ilocation). 


LARS의 두드러진 특징은 공간 항목에 대한 공간 사용자 평가를 사용하여 권장 사항을 생성하기 위해 거의 변경없이 사용자 분할 및 여행 패널티 기술을 함께 사용할 수 있다는 것입니다.
A salient feature of LARS is that both the user partitioning and travel penalty techniques can be used together with very little change to produce recommendations using spatial user ratings for spatial items. 


데이터 구조 및 유지 관리 기술은 섹션 III 및 IV에서 설명한 것과 동일합니다. 쿼리 처리 프레임 워크 만 약간의 수정이 필요합니다.
The data structures and maintenance techniques remain exactly the same as discussed in Sections III and IV; only the query processing framework requires a slight modification.


쿼리 처리는 알고리즘 2를 사용하여 권장 사항을 생성합니다.
Query processing uses Algorithm 2 to produce recommendations. 


그러나 유일한 차이점은 추천 점수 계산 (알고리즘 2의 16 행)에 사용 된 항목 기반 협업 필터링 예측 점수 P (u, i)가 부분 피라미드 셀에서 (국지화 된) 협업 필터링 모델을 사용하여 생성된다는 것입니다. 섹션 IV에서 사용 된 시스템 전체 협업 필터링 모델 대신 쿼리하는 사용자를 포함합니다.
However, the only difference is that the item-based collaborative filtering prediction score P(u, i) used in the recommendation score calculation (Line 16 in Algorithm 2) is generated using the (localized) collaborative filtering model from the partial pyramid cell that contains the querying user, instead of the system-wide collaborative filtering model as was used in Section IV.

---

### 6. EXPERIMENTS

이 섹션에서는 실제 시스템 구현을 기반으로하는 LARS의 실험적 평가를 제공합니다.
This section provides experimental evaluation of LARS based on an actual system implementation. 


LARS를 여러 변형 된 LARS와 함께 표준 항목 기반 협업 필터링 기술과 비교합니다.
We compare LARS with the standard item-based collaborative filtering technique along with several variations of LARS. 


실험은 세 가지 데이터 세트를 기반으로합니다.
Experiments are based on three data sets: 


(1) Foursquare : Foursquare 사용자 기록에서 파생 된 공간 항목에 대한 공간 사용자 등급으로 구성된 실제 데이터 세트입니다.
(1) Foursquare: a real data set consisting of spatial user ratings for spatial items derived from Foursquare user histories. 


(2) MovieLens : 인기있는 MovieLens 추천 시스템 [7]에서 가져온 비 공간 항목에 대한 공간 사용자 등급으로 구성된 실제 데이터 세트입니다.
(2) MovieLens: a real data set consisting of spatial user ratings for non-spatial items taken from the popular MovieLens recommender system [7]. 


Foursquare 및 MovieLens 데이터는 추천 품질을 테스트하는 데 사용됩니다.
The Foursquare and MovieLens data are used to test recommendation quality.


(3) 합성 : 미국 미네소타 주에있는 장소에 대한 공간 항목에 대한 공간 사용자 등급으로 구성된 종합적으로 생성 된 데이터 세트; 이 데이터를 사용하여 확장 성과 쿼리 효율성을 테스트합니다.
(3) Synthetic: a synthetically generated data set consisting spatial user ratings for spatial items for venues in the state of Minnesota, USA; we use this data to test scalability and query efficiency. 

모든 데이터 세트에 대한 자세한 내용은 부록 B에 있습니다.
Details of all data sets are found in Appendix B.


달리 언급하지 않는 한, M의 기본값은 0.3, k는 10, 피라미드 수준의 수는 8, 영향 수준은 가장 낮은 피라미드 수준입니다.
Unless mentioned otherwise, the default value of M is 0.3, k is 10, the number of pyramid levels is 8, and the influence level is the lowest pyramid level. 


이 섹션의 나머지 부분에서는 LARS 권장 사항 품질 (섹션 VI-A), 스토리지와 지역성 간의 균형 (섹션 VI-C), 확장 성 (섹션 VI-D) 및 쿼리 처리 효율성 (섹션 VI-E)을 평가합니다.
The rest of this section evaluates LARS recommendation quality (Section VI-A), trade-offs between storage and locality (Section VI-C), scalability (Section VI-D), and query processing efficiency (Section VI-E).


A. 다양한 피라미드 수준에 대한 권장 품질
#### A. Recommendation Quality for Varying Pyramid Levels


이러한 실험은 Fourquare 및 MovieLens 데이터를 모두 사용하여 표준 (비 공간) 항목 기반 협업 필터링 방법 (CF로 표시)에 대해 LARS의 추천 품질을 테스트합니다.
These experiments test the recommendation quality of LARS against the standard (non-spatial) item-based collaborative filtering method (denoted as CF) using both the Fourquare and MovieLens data. 


제안 된 기술의 효과를 테스트하기 위해 이동 패널티 만 활성화 된 LARS (abbr. LARS-T), 사용자 분할 만 활성화 한 LARS (abbr. LARS-U) 및 두 기술을 모두 활성화 한 LARS (abbr . LARS).
To test the effectiveness of our proposed techniques, we test the quality of LARS with only travel penalty enabled (abbr. LARS-T), LARS with only user partitioning enabled (abbr. LARS-U), and LARS with both techniques enabled (abbr. LARS). 


품질을 측정하기 위해 각 데이터 세트의 80 % 등급을 사용하여 각 추천 방법을 구축합니다.
To measure quality, we build each recommendation method using 80% of the ratings from each data set. 


보류 된 20 %의 각 등급은 사용자가 좋아하는 것으로 알려진 Foursquare 장소 또는 MovieLens 영화 (즉, 높은 등급)를 나타냅니다.
Each rating in the withheld 20% represents a Foursquare venue or MovieLens movie a user is known to like (i.e., rated highly). 


이 20 %의 각 등급 t에 대해 사용자 및 t와 관련된 ulocation을 제출하여 k 개의 권장 사항 R 세트를 요청합니다.
For each rating t in this 20%, we request a set of k recommendations R by submitting the user and ulocation associated with t. 

품질 측정 값은 R이 t와 관련된 항목을 포함하는 횟수입니다 (높을수록 좋습니다).
The quality measure is the count of how many times R contains the item associated with t (the higher the better). 


이 측정 항목의 근거는 각 보류 등급이 장소 (또는 사용자가 좋아하는 영화)에 대한 실제 방문을 나타내므로 사용자가 좋아하는 장소 (또는 영화)를 포함하는 많은 답변을 생성하는 기술이 고려된다는 것입니다. 더 높은 품질의.
The rationale for this metric is that since each withheld rating represents a real visit to a venue (or movie a user liked), the technique that produces a large number of answers that contain venues (or movies) a user is known to like is considered of higher quality.

![Fig6](./image/Fig6.PNG)  

그림 6 (a)는 Foursquare 데이터를 사용하여 다양한 지역 (즉, 다양한 수준의 적응 형 피라미드)에 대한 각 기술의 품질을 비교합니다.
Figure 6(a) compares the quality of each technique for varying locality (i.e., different levels of the adaptive pyramid) using the Foursquare data. 


CF와 LARS-T는 모두 적응 피라미드를 사용하지 않으므로 일정한 품질 값을 갖습니다.
Both CF and LARS-T do not use the adaptive pyramid, thus have constant quality values. 


CF와 LARS-T 간의 차이는 가능한 거리 내에서 항목을 추천하는 여행 패널티 기법을 사용하는 이점을 강조합니다.
The gap between CF and LARS-T highlights the benefit of using the travel penalty technique that recommends items within a feasible distance. 


한편, LARS 및 LARS-U의 품질은 더 많은 지역화 된 피라미드 셀이 추천을 생성하는 데 사용됨에 따라 증가하며, 이는 사용자 분할이 위치 기반 평가에 실제로 유익하고 필요하다는 것을 확인합니다.
Meanwhile, the quality of LARS and LARS-U increases as more localized pyramid cells are used to produce recommendation, which verifies that user partitioning is indeed beneficial and necessary for location-based ratings. 


궁극적으로 LARS는 추가 출장 벌금 사용으로 인해 우수한 성능을 제공합니다.
Ultimately, LARS has superior performance due to the additional use of travel penalty. 


출장 페널티는 중간 수준의 품질 향상을 가져 오지만, 나중에 섹션 VI-E에서 살펴볼 더 효율적인 쿼리 처리를 가능하게합니다.
While travel penalty produces moderate quality gain, it also enables more efficient query processing, which we observe later in Section VI-E).

그림 6 (b)는 MovieLens 데이터를 사용하여 다양한 지역에 대한 LARS-U 및 CF의 품질을 비교합니다 (영화는 공간이 아니므로 LARS 및 LARST는 적용되지 않음).
Figure 6(b) compares the quality of LARS-U and CF for varying locality using the MovieLens data (LARS and LARST do not apply since movies are not spatial). 


CF 품질은 일정하지만 LARS-U의 품질은 더 국지화 된 피라미드 셀에서 영화 추천을 생성 할 때 향상됩니다.
While CF quality is constant, the quality of LARS-U increases when it produces movie recommendations from more localized pyramid cells.


이 동작은 항목이 공간적이지 않은 경우에도 사용자 분할이 쿼리 사용자 위치에 지역화 된 품질 권장 사항을 제공하는 데 도움이되는지 확인합니다.
This behavior further verifies that user partitioning is beneficial in providing quality recommendations localized to a querying user location, even when items are not spatial. 


낮은 수준의 적응 형 피라미드에 대해 LARS-U 및 / 또는 LARS 모두에 대해 품질이 저하됩니다 (또는 MovieLens의 경우 수준이 꺼짐).
Quality decreases (or levels off for MovieLens) for both LARS-U and/or LARS for lower levels of the adaptive pyramid. 


이는 권장 사항 부족 때문입니다. 즉, 의미있는 권장 사항을 생성하기에 충분한 등급이 없기 때문입니다.
This is due to recommendation starvation, i.e., not having enough ratings to produce meaningful recommendations.


B. 다양한 k 값에 대한 권장 품질
#### B. Recommendation Quality for Varying Values of k

이러한 실험은 다양한 k 값 (즉, 권장 응답 크기)에 대해 LARS, LARS-U, LARS-T 및 CF의 권장 품질을 테스트합니다.
These experiments test recommendation quality of LARS, LARS-U, LARS-T, and CF for different values of k (i.e., recommendation answer sizes). 


Foursquare 및 MovieLens 데이터를 모두 사용하여 실험을 수행합니다.
We perform experiments using both the Foursquare and MovieLens data. 


우리의 품질 메트릭은 섹션 VI-A에서 이전에 제시된 것과 정확히 동일합니다.
Our quality metric is exactly the same as presented previously in Section VI-A.

![Fig7](./image/Fig7.PNG)  

그림 7 (a)는 Foursquare 데이터 세트를 사용하는 각 기술의 품질에 대한 권장 목록 크기 k의 효과를 보여줍니다.
Figure 7(a) depicts the effect of the recommendation list size k on the quality of each technique using the Foursquare data set. 


피라미드 높이 4를 사용하여 품질 수치를보고합니다 (즉, 그림 6 (a)의 섹션 VI-A에서 최고의 품질을 나타내는 수준).
We report quality numbers using the pyramid height of four (i.e., the level exhibiting the best quality from Section VI-A in Figure 6(a)). 


1에서 10까지의 모든 k 크기에 대해 LARS 및 LARS-U는 지속적으로 더 나은 품질을 보여줍니다.
For all sizes of k from one to ten, LARS and LARS-U consistently exhibit better quality. 


실제로 LARS는 모든 k에 대해 CF보다 지속적으로 두 배 더 정확합니다.
In fact, LARS is consistently twice as accurate as CF for all k. 


LARST는 더 작은 k 값에 대해 CF와 비슷한 품질을 나타내지 만 3 이상의 k 값에 대해 더 좋습니다.
LARST exhibits similar quality to CF for smaller k values, but does better for k values of three and larger.

그림 7 (b)는 MovieLens 데이터를 사용하여 LARS-U 및 CF의 품질에 대한 권장 목록 크기 k의 영향을 보여줍니다 (영화가 공간이 아니기 때문에이 실험에서는 LARS 및 LARS-T가 적용되지 않음).
Figure 7(b) depicts the effect of the recommendation list size k on the quality of LARS-U and CF using the MovieLens data (LARS and LARS-T do not apply in this experiment since movies are not spatial). 


이 실험은 피라미드 높이 7 (즉, 그림 6 (b)에서 최고의 품질을 나타내는 수준)을 사용하여 실행되었습니다.
This experiment was run using a pyramid hight of seven (i.e., the level exhibiting the best quality in Figure 6(b)). 


다시 말하지만, LARS-U는 1에서 10까지의 K 크기에 대해 CF보다 지속적으로 더 나은 품질을 보여줍니다.
Again, LARS-U consistently exhibits better quality than CF for sizes of K from one to ten. 


사실, CF의 품질은 k가 증가함에 따라 조금씩 증가합니다.
In fact, the quality of CF increases by just a fraction as k increases.


한편, LARS-U의 품질은 k가 1에서 10으로 증가함에 따라 7 배 증가합니다.
Meanwhile, the quality of LARS-U increases by a factor of seven as k increases from one to ten.


C. 스토리지 대. 소재지
#### C. Storage Vs. Locality

![Fig8](./image/Fig8.PNG)  

그림 8은 다양한 M이 LARS의 스토리지 및 지역성에 미치는 영향을 보여줍니다.
Figure 8 depicts the impact of varying M on both the storage and locality in LARS. 


LARS-M = 0 및 LARSM = 1을 상수로 플로팅하여 M의 극한 값을 나타냅니다. 즉, M = 0은 기존의 협업 필터링을 반영하고 M = 1은 LARS가 완전한 피라미드를 사용하도록합니다.
We plot LARS-M=0 and LARSM=1 as constants to delineate the extreme values of M, i.e., M=0 mirrors traditional collaborative filtering, while M=1 forces LARS to employ a complete pyramid. 


지역성에 대한 측정 항목은 완전한 피라미드 (즉, M = 1)와 비교할 때 지역성 손실 (섹션 III-C1에 정의 됨)입니다.
Our metric for locality is locality loss (defined in Section III-C1) when compared to a complete pyramid (i.e., M=1). 


LARS-M = 0은 가장 낮은 스토리지 오버 헤드를 필요로하지만 가장 높은 지역성 손실을 보이는 반면 LARS-M = 1은 지역성 손실이 없지만 가장 많은 저장소를 필요로합니다.
LARS-M=0 requires the lowest storage overhead, but exhibits the highest locality loss, while LARS-M=1 exhibits no locality loss but requires the most storage. 


LARS의 경우 M을 늘리면 LARS가 분할을 선호하기 때문에 스토리지 오버 헤드가 증가하여 각각 고유 한 협업 필터링 모델을 사용하여 더 많은 피라미드 셀을 유지 관리해야합니다.
For LARS, increasing M results in increased storage overhead since LARS favors splitting, requiring the maintenance of more pyramid cells each with its own collaborative filtering model. 


한편, M을 늘리면 LARS가 더 적게 병합되고 더 많은 지역화 된 셀을 유지하므로 지역성 손실이 더 작아집니다.
Meanwhile, increasing M results in smaller locality loss as LARS merges less and maintains more localized cells. 


지역성 손실의 가장 급격한 감소는 0에서 0.3 사이이므로 M = 0.3을 기본값으로 선택했습니다.
The most drastic drop in locality loss is between 0 and 0.3, which is why we chose M=0.3 as a default.


D. 확장 성
#### D. Scalability

![Fig9](./image/Fig9.PNG)  

그림 9는 등급 증가에 필요한 스토리지 및 총 유지 보수 오버 헤드를 보여줍니다.
Figure 9 depicts the storage and aggregate maintenance overhead required for an increasing number of ratings. 


LARS에 대한 극단적 인 경우를 나타 내기 위해 LARS-M = 0 및 LARS-M = 1을 다시 플로팅합니다.
We again plot LARS-M=0 and LARS-M=1 to indicate the extreme cases for LARS. 


그림 9 (a)는 등급 수를 10K에서 500K로 증가시키는 것이 스토리지 오버 헤드에 미치는 영향을 보여줍니다.
Figure 9(a) depicts the impact of increasing the number of ratings from 10K to 500K on storage overhead.


LARS-M = 0은 단일 협업 필터링 모델 만 유지하므로 가장 적은 양의 스토리지가 필요합니다.
LARS-M=0 requires the lowest amount of storage since it only maintains a single collaborative filtering model. 


LARSM = 1은 전체 피라미드의 모든 셀 (모든 수준)에 대해 협업 필터링 모델을 저장해야하므로 가장 많은 양의 스토리지가 필요합니다.
LARSM=1 requires the highest amount of storage since it requires storage of a collaborative filtering model for all cells (in all levels) of a complete pyramid. 


LARS의 스토리지 요구 사항은 스토리지를 절약하기 위해 셀을 병합하기 때문에 두 극단 사이에 있습니다.
The storage requirement of LARS is in between the two extremes since it merges cells to save storage. 


그림 9 (b)는 처음에 100K 등급으로 채워진 적응 형 피라미드를 유지하는 데 필요한 누적 계산 오버 헤드를 보여준 다음 200K 등급으로 업데이트되었습니다 (보고 된 50K 증가).
Figure 9(b) depicts the cumulative computational overhead necessary to maintain the adaptive pyramid initially populated with 100K ratings, then updated with 200K ratings (increments of 50K reported). 


추세는 병합으로 인해 LARS가 LARS-M = 1보다 더 나은 성능을 나타내는 스토리지 실험과 유사합니다.
The trend is similar to the storage experiment, where LARS exhibits better performance than LARS-M=1 due to merging. 


LARS-M = 0은 유지 관리 및 보관 오버 헤드 측면에서 최고의 성능을 보였지만 이전 실험에서는 품질 / 지역성에 허용 할 수없는 단점이 있음을 보여줍니다.
Though LARS-M=0 has the best performance in terms of maintenance and storage overhead, previous experiments show that it has unacceptable drawbacks in quality/locality.

E. 쿼리 처리 성능
#### E. Query Processing Performance

![Fig10](./image/Fig10.PNG)  

그림 10은 LARS, LARS-U (사용자 파티셔닝 만있는 LARS), LARS-T (여행 패널티 만있는 LARS), CF (기존의 협업 필터링) 및 LARS-M = 1 ( 완전한 피라미드).
Figure 10 depicts snapshot and continuous query processing performance of LARS, LARS-U (LARS with only user partitioning), LARS-T (LARS with only travel penalty), CF (traditional collaborative filtering), and LARS-M=1 (LARS with a complete pyramid).


스냅 샷 쿼리.
Snapshot queries. 


그림 10 (a)는 임의 위치에서 발생한 평균 500 개 이상의 쿼리에 대한 평균 스냅 샷 쿼리 성능에 대한 다양한 등급 수 (10K ~ 500K)의 효과를 보여줍니다.
Figure 10(a) gives the effect of various number of ratings (10K to 500K) on the average snapshot query performance averaged over 500 queries posed at random locations. 


LARS 및 LARS-M = 1은 일관되게 다른 모든 기술을 능가합니다.
LARS and LARS-M=1 consistently outperform all other techniques; 


LARS-M = 1은 항상 가장 작은 (즉, 가장 현지화 된) CF 모델에서 생성되는 권장 사항으로 인해 약간 더 좋습니다.
LARS-M=1 is slightly better due to recommendations always being produced from the smallest (i.e., most localized) CF models. 


LARS와 LARS-U (및 CF 및 LARS-T) 간의 성능 차이는 조기 종료와 함께 이동 페널티 기술을 사용하면 쿼리 응답 시간이 향상된다는 것을 보여줍니다.
The performance gap between LARS and LARS-U (and CF and LARS-T) shows that employing the travel penalty technique with early termination leads to better query response time. 


마찬가지로 LARS와 LARS-T 간의 성능 차이는 지역화 된 (즉, 더 작은) 협업 필터링 모델과 함께 사용자 분할 기술을 사용하는 것이 쿼리 처리에도 도움이된다는 것을 보여줍니다.
Similarly, the performance gap between LARS and LARS-T shows that employing user partitioning technique with its localized (i.e., smaller) collaborative filtering model also benefits query processing. 


연속 쿼리.
Continuous queries. 

그림 10 (b)는 500 개의 연속 쿼리의 집계 응답 시간을보고하여 LARS 변형의 연속 쿼리 처리 성능을 제공합니다.
Figure 10(b) provides the continuous query processing performance of the LARS variants by reporting the aggregate response time of 500 continuous queries. 


사용자 u가 초기 답변을 얻기 위해 연속 쿼리를 한 번 실행 한 다음 u가 이동함에 따라 답변이 지속적으로 업데이트됩니다.
A continuous query is issued once by a user u to get an initial answer, then the answer is continuously updated as u moves.


피라미드로 덮힌 공간 영역을 무작위로 걷는 것을 사용하여 u의 이동 거리를 1에서 30 마일까지 변경할 때 총 응답 시간을보고합니다.
We report the aggregate response time when varying the travel distance of u from 1 to 30 miles using a random walk over the spatial area covered by the pyramid. 


CF는 단일 셀만 존재하므로 업데이트가 필요하지 않으므로 모든 이동 거리에 대해 일정한 쿼리 응답 시간을 갖습니다.
CF has a constant query response time for all travel distances, as it requires no updates since only a single cell is present. 


그러나 CF는 사용자 위치 변경을 인식하지 못하므로 결과적으로 권장 품질이 떨어집니다 (섹션 VI-A의 실험에 따라).
However, since CF is unaware of user location change, the consequence is poor recommendation quality (per experiments from Section VI-A).


LARS-M = 1은 모든 수준의 모든 셀을 유지하고 사용자가 피라미드 셀 경계를 넘을 때마다 연속 쿼리를 업데이트하므로 성능이 더 나쁩니다.
LARS-M=1 exhibits the worse performance, as it maintains all cells on all levels and updates the continuous query whenever the user crosses pyramid cell boundaries. 


LARS-U는 병합으로 인해 LARS-M = 1보다 응답 시간이 짧습니다. 특정 영향 수준에 셀이 없으면 쿼리가 피라미드에서 다음으로 높은 상위 항목으로 전송됩니다.
LARS-U has a lower response time than LARS-M=1 due to merging: when a cell is not present on a given influence level, the query is transferred to its next highest ancestor in the pyramid. 


피라미드에서 더 높은 셀은 더 큰 공간 영역을 포함하므로 쿼리 업데이트가 덜 자주 발생합니다.
Since cells higher in the pyramid cover larger spatial regions, query updates occur less often. 

LARS-T는 LARS-U에 비해 약간 더 높은 쿼리 처리 오버 헤드를 나타냅니다. LARST는 조기 종료 알고리즘을 사용하지만 사용자가 페널티 그리드의 경계를 넘으면 권장 사항을 (재생성)하기 위해 대규모 (시스템 전체) 협업 필터링 모델을 사용합니다. .
LARS-T exhibits slightly higher query processing overhead compared to LARS-U: even though LARST employs the early termination algorithm, it uses a large (system-wide) collaborative filtering model to (re)generate recommendations once users cross boundaries in the penalty grid. 


LARS는 로컬 화 된 (즉, 더 작은) 협업 필터링 모델을 사용하는 조기 종료 알고리즘을 사용하여 결과를 생성하는 동시에 셀을 병합하여 업데이트 빈도를 줄이기 때문에 더 나은 집계 응답 시간을 보여줍니다.
LARS exhibits a better aggregate response time since it employs the early termination algorithm using a localized (i.e., smaller) collaborative filtering model to produce results while also merging cells to reduce update frequency.


VII. 관련된 일
### 7. RELATED WORK

위치 기반 서비스.
Location-based services. 

현재 위치 기반 서비스는 사용자에게 흥미로운 목적지를 제공하기 위해 두 가지 주요 방법을 사용합니다.
Current location-based services employ two main methods to provide interesting destinations to users. 


(1) KNN 기술 [19] 및 변형 (예 : 집계 KNN [21])은 단순히 사용자에게 가장 가까운 k 개의 객체를 검색하고 사용자 개인화 개념에서 완전히 제거됩니다.
(1) KNN techniques [19] and variants (e.g., aggregate KNN [21]) simply retrieve the k objects nearest to a user and are completely removed from any notion of user personalization. 


(2) 스카이 라인 [22] (및 공간 변형 [23]) 및 위치 기반 top-k 방법 [24]과 같은 선호 방법은 사용자가 명시 적 선호 제약을 표현하도록 요구합니다.
(2) Preference methods such as skylines [22] (and spatial variants [23]) and location-based top-k methods [24] require users to express explicit preference constraints. 


반대로 LARS는 사용자가 새롭고 흥미로운 항목을 찾을 수 있도록 위치 기반 등급을 사용하여 암시 적 선호도를 고려한 최초의 위치 기반 서비스입니다.
Conversely, LARS is the first location-based service to consider implicit preferences by using location-based ratings to help users discover new and interesting items.


최근 연구는 지역 밀착 형 순위 문제를 제안했다 [25].
Recent research has proposed the problem of hyper-local place ranking [25]. 


사용자 위치와 검색어 문자열 (예 : '프랑스 음식점')이 주어지면 지역 밀착 형 순위는 이전에 기록 된 방향 검색어 (예 : A 지점에서 B 지점까지의지도 방향 검색)의 영향을받은 상위 k 개의 관심 지점 목록을 제공합니다.
Given a user location and query string (e.g., “French restaurant”), hyper-local ranking provides a list of top-k points of interest influenced by previously logged directional queries (e.g., map direction searches from point A to point B). 


LARS와 정신적으로 비슷하지만 하이퍼 로컬 순위는 쿼리하는 사용자에 대한 답변을 개인화하지 않기 때문에 우리 작업과 근본적으로 다릅니다. 즉, 동일한 위치에서 동일한 검색어를 발행하는 두 명의 사용자가 정확히 동일한 순위의 답변 세트를 받게됩니다.
While similar in spirit to LARS, hyper-local ranking is fundamentally different from our work as it does not personalize answers to the querying user, i.e., two users issuing the same search term from the same location will receive exactly the same ranked answer set.


전통적인 추천자.
Traditional recommenders. 

다양한 기술이 트리플 (사용자, 등급, 항목)로 표시되는 비 공간 항목에 대한 비 공간 등급을 사용하여 추천을 생성 할 수 있습니다 (종합적인 조사는 [6] 참조).
A wide array of techniques are capable of producing recommendations using non-spatial ratings for non-spatial items represented as the triple (user, rating, item) (see [6] for a comprehensive survey). 


이를 "전통적인"추천 기술이라고합니다.
We refer to these as “traditional” recommendation techniques. 


이러한 접근 방식은 위치를 고려하는 데 가장 가까운 방법은 상황 속성을 통계적 추천 모델 (예 : 날씨, 목적지까지의 교통량)에 통합하는 것입니다 [26].
The closest these approaches come to considering location is by incorporating contextual attributes into statistical recommendation models (e.g., weather, traffic to a destination) [26]. 


그러나 LARS에서 수행 된 것처럼 명시적인 위치 기반 등급을 연구 한 기존 접근 방식은 없습니다.
However, no traditional approach has studied explicit location-based ratings as done in LARS. 


기존의 일부 상용 응용 프로그램은 사용자에게 흥미로운 항목을 제안 할 때 위치를 대략적으로 사용합니다.
Some existing commercial applications make cursory use of location when proposing interesting items to users. 


예를 들어 Netflix [2]는 사용자가 거주하는 도시의 인기 영화가 포함 된 '지역 즐겨 찾기'목록을 표시합니다.
For instance, Netflix [2] displays a “local favorites” list containing popular movies for a user’s given city. 


그러나 이러한 영화는 각 사용자에게 개인화되지 않습니다 (예 : 추천 기술 사용). 오히려이 목록은 특정 도시에 대한 총 임대 데이터를 사용하여 작성됩니다 [27].
However, these movies are not personalized to each user (e.g., using recommendation techniques); rather, this list is built using aggregate rental data for a particular city [27]. 


반면 LARS는 위치 기반 평가 및 쿼리 사용자 위치에 영향을받는 개인화 된 추천을 생성합니다.
LARS, on the other hand, produces personalized recommendations influenced by location-based ratings and a querying user location. 


위치 인식 추천자.
Location-aware recommenders. 

--- 
