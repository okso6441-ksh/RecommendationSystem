## Recommendation System

### theory  
* 주체: 사용자(User) / 상품(Item)  
* 특징: Push, 사용자 요구 전, 사용자 요구사항 정확히 알지 못하고 동작, 개인화     
* 동작과정: 사용자-상품 관계 분석> 연관관계 **점수화**> 추천    
  * 프로파일링(Profileing): 데이터 수집(Explict/**Implicit**)    
    * 데이터: 사용자, 상품, 트랜젝션(로그)  
* 유형: 랭킹(Top-K) v.s 예측(User-Item Matrix 결측치)  

* 알고리즘: 
  * Contents-based: 사용자가 좋아하던 상품과 비슷한 상품 추천    
  * Collaborative Filtering(협업필터링): 비슷한 사용자가 좋아하던 상품을 추천  
  * Context-based  
    * Context-aware, Location-based, Real-Time/Time-Sensitive  
  * Community-based  
  * Knowledge-based  
    * Case-based, Constraint-based   

* 평가: Offline/Online  
  * 랭킹: NDCG(Normalized Discounted Cumulative Gain)   
  * 예측: RMSE  
  * Precision@K(Top-K) >[확장]> MAP(Mean Average Precision)  

* 한계/어려움          
  * Scalability, Proactive, **Cold-Start** Problem, Privacy, Mobile Device/User Context, Long/Short-term user preference, Generic Model/Cross Domain, Stavation/Diversity
  * 필터버블(정보의 비대칭)    

  ---
  *참고 출처: [fastcampus] 딥러닝을 활용한 추천시스템 구현 올인원 패키지 Online*