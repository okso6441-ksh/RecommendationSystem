## 2015_IRSS [Image-based Recommendations on Styles and Substitutes]

![main](./image/main.PNG)

---

### Abstract

Humans inevitably develop a sense of the relationships between objects, some of which are based on their appearance. 
인간은 필연적으로 사물 간의 관계에 대한 감각을 개발하는데, 그중 일부는 외모에 기반을 둡니다.


Some pairs of objects might be seen as being alternatives to each other (such as two pairs of jeans), while others may be seen as being complementary (such as a pair of jeans and a matching shirt).
어떤 쌍의 물체는 서로의 대안으로 보일 수 있고 (예 : 청바지 두 쌍), 다른 물체는 보완적인 것으로 보일 수 있습니다 (예 : 청바지 한 쌍과 일치하는 셔츠).


This information guides many of the choices that people make, from buying clothes to their interactions with each other. 
이 정보는 사람들이 옷을 구입하는 것부터 서로 상호 작용하는 것까지 선택하는 많은 선택을 안내합니다.


We seek here to model this human sense of the relationships between objects based on their appearance. 
우리는 여기에서 외모를 기반으로 물체 간의 관계에 대한 인간의 감각을 모델링하고자합니다.


Our approach is not based on fine-grained modeling of user annotations but rather on capturing the largest dataset possible and developing a scalable method for uncovering human notions of the visual relationships within. 
우리의 접근 방식은 사용자 주석의 세분화 된 모델링을 기반으로하는 것이 아니라 가능한 가장 큰 데이터 세트를 캡처하고 내부의 시각적 관계에 대한 인간의 개념을 발견하기위한 확장 가능한 방법을 개발하는 데 기반합니다.


We cast this as a network inference problem defined on graphs of related images, and provide a large-scale dataset for the training and evaluation of the same. 
이를 관련 이미지의 그래프에 정의 된 네트워크 추론 문제로 캐스트하고이를 학습 및 평가할 수있는 대규모 데이터 세트를 제공합니다.


The system we develop is capable of recommending which clothes and accessories will go well together (and which will not), amongst a host of other applications.
우리가 개발 한 시스템은 다른 여러 애플리케이션 중에서 어떤 옷과 액세서리가 잘 어울리는 지 (그리고 그렇지 않을지) 추천 할 수 있습니다.

1. Introduction

We are interested here in uncovering relationships between the appearances of pairs of objects, and particularly in modeling the human notion of which objects complement each other and which might be seen as acceptable alternatives. 
우리는 여기에서 물체 쌍의 모양 사이의 관계를 밝히고 특히 어떤 물체가 서로 보완하고 수용 가능한 대안으로 보일 수 있는지에 대한 인간의 개념을 모델링하는 데 관심이 있습니다.


We thus seek to model what is a fundamentally human notion of the visual relationship between a pair of objects, rather than merely modeling the visual similarity between them. 
따라서 우리는 단순히 객체 간의 시각적 유사성을 모델링하는 것이 아니라 한 쌍의 객체 간의 시각적 관계에 대한 근본적인 인간 개념을 모델링하려고합니다.


There has been some interest of late in modeling the visual style of places [6, 27], and objects [39]. 
최근 장소 [6, 27] 및 객체 [39]의 시각적 스타일을 모델링하는 데 관심이있었습니다.


We, in contrast, are not seeking to model the individual appearances of objects, but rather how the appearance of one object might influence the desirable visual attributes of another.
대조적으로 우리는 개체의 개별적인 모습을 모델링하는 것이 아니라 한 개체의 모양이 다른 개체의 바람직한 시각적 속성에 어떻게 영향을 미칠 수 있는지를 모색하고 있습니다.


There are a range of situations in which the appearance of an object might have an impact on the desired appearance of another. 
개체의 모양이 원하는 다른 모양에 영향을 미칠 수있는 다양한 상황이 있습니다.


Questions such as ‘Which frame goes with this picture’, ‘Where is the lid to this’, and ‘Which shirt matches these shoes’ (see Figure 1) inherently involve a calculation of more than just visual similarity, but rather a model of the higher-level relationships between objects.  
'이 그림과 어울리는 프레임', '이 신발의 뚜껑은 어디에 있습니까?', '이 신발과 어울리는 셔츠'(그림 1 참조)와 같은 질문은 본질적으로 시각적 유사성 이상의 계산을 포함합니다. 개체 간의 상위 수준 관계.


The primary commercial application for such technology is in recommending items to a user based on other items they have already showed interest in. 
이러한 기술의 주요 상업적 응용은 사용자가 이미 관심을 보인 다른 항목을 기반으로 항목을 추천하는 것입니다.


Such systems are of considerable economic value, and are typically built by analysing meta-data, reviews, and previous purchasing patterns. 
이러한 시스템은 상당한 경제적 가치가 있으며 일반적으로 메타 데이터, 리뷰 및 이전 구매 패턴을 분석하여 구축됩니다.


By introducing into these systems the ability to examine the appearance of the objects in question we aim to overcome some of their limitations, including the ‘cold start’ problem [28, 41].
이러한 시스템에 문제가되는 물체의 모양을 검사 할 수있는 기능을 도입함으로써 우리는 '콜드 스타트 ​​(cold start)'문제를 포함한 몇 가지 한계를 극복하는 것을 목표로합니다 [28, 41].


The problem we pose inherently requires modeling human visual preferences. 
우리가 제기하는 문제는 본질적으로 인간의 시각적 선호도를 모델링해야합니다.


In most cases there is no intrinsic connection between a pair of objects, only a human notion that they are more suited to each other than are other potential partners. 
대부분의 경우 한 쌍의 객체 사이에는 본질적인 연결이 없으며 다른 잠재적 파트너보다 서로에게 더 적합하다는 인간의 생각 만 있습니다.


The most common approach to modeling such human notions exploits a set of hand-labeled images created for the task. 
이러한 인간 개념을 모델링하는 가장 일반적인 접근 방식은 작업을 위해 생성 된 수작업 라벨 이미지 세트를 활용합니다.


The labeling effort required means that most such datasets are typically relatively small, although there are a few notable exceptions. 
필요한 라벨링 노력은 몇 가지 주목할만한 예외가 있지만 대부분의 데이터 세트가 일반적으로 상대적으로 작다는 것을 의미합니다.


A small dataset means that complex procedures are required to extract as much information as possible without overfitting (see [2, 5, 22] for example). 
작은 데이터 세트는 과적 합없이 가능한 한 많은 정보를 추출하기 위해 복잡한 절차가 필요함을 의미합니다 (예를 들어 [2, 5, 22] 참조).

It also means that the results are unlikely to be transferable to related problems. 
또한 결과가 관련 문제로 이전 될 가능성이 낮음을 의미합니다.


Creating a labeled dataset is particularly onerous when modeling pairwise distances because the number of annotations required scales with the square of the number of elements.
레이블이 지정된 데이터 세트를 만드는 것은 특히 쌍 거리를 모델링 할 때 번거로운 작업입니다. 주석의 수는 요소 수의 제곱으로 조정되기 때문입니다.


We propose here instead that one might operate over a much larger dataset, even if it is only tangentially related to the ultimate goal. 
대신 우리는 궁극적 인 목표와 접선 적으로 만 관련되어 있더라도 훨씬 더 큰 데이터 세트에 대해 작동 할 수 있다고 제안합니다.


Thus, rather than devising a process (or budget) for manually annotating images, we instead seek a freely available source of a large amount of data which may be more loosely related to the information we seek. 
따라서 이미지에 수동으로 주석을 달기위한 프로세스 (또는 예산)를 고안하는 대신 우리가 찾는 정보와 더 느슨하게 관련 될 수있는 많은 양의 데이터를 자유롭게 사용할 수있는 소스를 찾습니다.


Large-scale databases have been collected from the web (without other annotation) previously [7, 34]. 
대규모 데이터베이스는 이전에 다른 주석없이 웹에서 수집되었습니다 [7, 34].


What distinguishes the approach we propose here, however, is the fact that it succeeds despite the indirectness of the connection between the dataset and the quantity we hope to model.
그러나 여기서 제안하는 접근 방식을 구별하는 것은 데이터 세트와 모델링하고자하는 수량 간의 연결이 간접적 임에도 불구하고 성공한다는 사실입니다.

1.1 A visual dataset of styles and substitutes
1.1 스타일 및 대체물의 시각적 데이터 세트

We have developed a dataset suitable for the purposes described above based on the Amazon web store. 
Amazon 웹 스토어를 기반으로 위에서 설명한 목적에 적합한 데이터 세트를 개발했습니다.


The dataset contains over 180 million relationships between a pool of almost 6 million objects. 
이 데이터 세트에는 거의 6 백만 개체의 풀간에 1 억 8 천만 개 이상의 관계가 포함되어 있습니다.


These relationships are a result of visiting Amazon and recording the product recommendations that it provides given our (apparent) interest in the subject of a particular web page. 
이러한 관계는 아마존을 방문하고 특정 웹 페이지의 주제에 대한 (명백한) 관심을 고려하여 제공하는 제품 권장 사항을 기록한 결과입니다.


The statistics of the dataset are shown in Table 1. 
데이터 세트의 통계는 표 1에 나와 있습니다.


An image and a category label are available for each object, as is the set of users who reviewed it. 
이미지와 카테고리 레이블은이를 검토 한 사용자 집합과 마찬가지로 각 개체에 사용할 수 있습니다.


We have made this dataset available for academic use, along with all code used in this paper to ensure that our results are reproducible and extensible.
우리는 결과가 재현 가능하고 확장 가능하도록이 백서에 사용 된 모든 코드와 함께이 데이터 세트를 학술 용으로 제공했습니다.


We label this the Styles and Substitutes dataset.
우리는 이것을 Styles and Substitutes 데이터 셋이라고 명명합니다.


The recorded relationships describe two specific notions of ‘compatibility’ that are of interest, namely those of substitute and complement goods. 
기록 된 관계는 관심있는 '호환성'의 두 가지 특정 개념, 즉 대체 및 보완 제품의 개념을 설명합니다.


Substitute goods are those that can be interchanged (such as one pair of pants for another), while complements are those that might be purchased together (such as a pair of pants and a matching shirt) [23]. 
대체 상품은 교환 할 수있는 상품 (예 : 바지 한 켤레를 다른 상품으로)이고 보완 상품은 함께 구매할 수있는 상품 (예 : 바지 한 켤레와 일치하는 셔츠)입니다 [23].

Specifically, there are 4 categories of relationship represented in the dataset: 
1) ‘users who viewed X also viewed Y’ (65M edges); 
2) ‘users who viewed X eventually bought Y’ (7.3M edges); 
3) ‘users who bought X also bought Y’ (104M edges); and 
4) ‘users bought X and Y simultaneously’ (3.4M edges). 
특히 데이터 세트에는 4 개의 관계 카테고리가 있습니다.
1) 'X를 본 사용자는 Y도 본 사용자'(6500 만 에지);
2) 'X를 본 사용자는 결국 Y를 구매'(730 만 에지);
3)‘X를 구매 한 사용자는 Y도 구매했습니다.’(104M 엣지) 과
4) '사용자가 X와 Y를 동시에 구매'(340 만 에지).

Critically, categories 1 and 2 indicate (up to some noise) that two products may be substitutable, while 3 and 4 indicate that two products may be complementary. 
비판적으로 카테고리 1과 2는 두 제품이 대체 가능할 수 있음을 (약간의 소음까지) 나타내며, 3과 4는 두 제품이 상호 보완적일 수 있음을 나타냅니다.


According to Amazon’s own tech report [19] the above relationships are collected simply by ranking products according to the cosine similarity of the sets of users who purchased/viewed them.
Amazon의 자체 기술 보고서 ​​[19]에 따르면 위의 관계는 제품을 구매 / 조회 한 사용자 집합의 코사인 유사성에 따라 제품의 순위를 매김으로써 간단히 수집됩니다.


Note that the dataset does not document users’ preferences for pairs of images, but rather Amazon’s estimate of the set of relationships between pairs objects. 
데이터 세트는 이미지 쌍에 대한 사용자의 선호도를 문서화하지 않고 쌍 객체 간의 관계 세트에 대한 Amazon의 추정치를 문서화합니다.


The human notion of the visual compatibility of these images is only one factor amongst many which give rise to these estimated relationships, and it is not a factor used by Amazon in creating them. 
이러한 이미지의 시각적 호환성에 대한 인간의 개념은 이러한 추정 된 관계를 유발하는 많은 요소 중 하나 일 뿐이며이를 생성 할 때 Amazon에서 사용하는 요소가 아닙니다.


We thus do not wish to summarize the Amazon data, but rather to use what it tells us about the images of related products to develop a sense of which objects a human might feel are visually compatible. 
따라서 아마존 데이터를 요약하는 것이 아니라 관련 제품의 이미지에 대해 알려주는 내용을 사용하여 인간이 시각적으로 호환되는 것으로 느낄 수있는 물체에 대한 감각을 개발합니다.


This is significant because many of the relationships between objects present in the data are not based on their appearance. 
이는 데이터에있는 개체 간의 많은 관계가 모양을 기반으로하지 않기 때문에 중요합니다.


People co-purchase hammers and nails due to their functions, for example, not their appearances. 
사람들은 외모가 아닌 기능으로 인해 망치와 못을 공동 구매합니다.


Our hope is that the non-visual decision factors will appear as uniformly distributed noise to a method which considers only appearance, and that the visual decision factors might reinforce each other to overcome the effect of this noise 
우리의 희망은 외모만을 고려하는 방법에 비 시각적 결정 요인이 균일하게 분포 된 노이즈로 나타나고, 시각적 결정 요인이이 노이즈의 영향을 극복하기 위해 서로를 강화할 수 있기를 바랍니다.

1.2 Related work

The closest systems to what we propose above are contentbased recommender systems [18] which attempt to model each user’s preference toward particular types of goods. 
위에서 제안한 것과 가장 가까운 시스템은 특정 유형의 상품에 대한 각 사용자의 선호도를 모델링하려는 콘텐츠 기반 추천 시스템 [18]입니다.


This is typically achieved by analyzing metadata from the user’s previous activities. 
이는 일반적으로 사용자의 이전 활동에서 메타 데이터를 분석하여 달성됩니다.


This is as compared to collaborative recommendation approaches which match the user to profiles generated based on the purchases/behavior of other users (see [1, 16] for surveys). 
이것은 다른 사용자의 구매 / 행동을 기반으로 생성 된 프로필에 사용자를 일치시키는 협업 추천 접근 방식과 비교됩니다 (설문 조사는 [1, 16] 참조).


Combinations of the two [3, 24] have been shown to help address the sparsity of the review data available, and the cold-start problem (where new products don’t have reviews and are thus invisible to the recommender system) [28, 41]. 
두 가지 [3, 24]의 조합은 사용 가능한 리뷰 데이터의 희소성과 콜드 스타트 ​​문제 (신제품에 리뷰가 없으므로 추천 시스템에 표시되지 않음)를 해결하는 데 도움이되는 것으로 나타났습니다 [28, 41].


The approach we propose here could also help address these problems.
여기서 제안하는 접근 방식은 이러한 문제를 해결하는데도 도움이 될 수 있습니다.


There are a range of services such as Jinni2 which promise content-based recommendations for TV shows and similar media, but the features they expoit are based on reviews and metadata (such as cast, director etc.), and their ontology is handcrafted. 
TV 프로그램 및 유사 미디어에 대한 콘텐츠 기반 추천을 약속하는 Jinni2와 같은 다양한 서비스가 있지만 이들이 노출하는 기능은 리뷰 및 메타 데이터 (예 : 캐스트, 감독 등)를 기반으로하며 온톨로지를 직접 제작합니다.


The Netflix prize was a well publicized competition to build a better personalized video recommender system, but there again no actual image analysis is taking place [17]. 
넷플릭스상은 더 나은 개인화 된 비디오 추천 시스템을 구축하기위한 잘 알려진 경쟁 이었지만 실제 이미지 분석은 다시 일어나지 않았습니다 [17].


Hu et al. [9] describe a system for identifying a user’s style, and then making clothing recommendations, but this is achieved through analysis of ‘likes’ rather than visual features.
Hu et al. [9]는 사용자의 스타일을 식별 한 다음 의상을 추천하는 시스템을 설명하지만 이는 시각적 특징이 아닌 '좋아요'분석을 통해 이루어집니다.


Content-based image retrieval gives rise to the problem of bridging the ‘semantic-gap’ [32], which requires returning results which have similar semantic content to a search image, even when the pixels bear no relationship to each other. 
콘텐츠 기반 이미지 검색은 픽셀이 서로 관계가없는 경우에도 유사한 의미 콘텐츠를 가진 결과를 검색 이미지에 반환해야하는 '의미 적 차이'[32]를 연결하는 문제를 야기합니다.


It thus bears some similarity to the visual recommendation problem, as both require modeling a human preference which is not satisfied by mere visual similarity. 
따라서 둘 다 단순한 시각적 유사성으로 만족되지 않는 인간 선호도를 모델링해야하기 때문에 시각적 추천 문제와 약간의 유사성을 가지고 있습니다.


There are a variety of approaches to this problem, many of which seek a set of results which are visually similar to the query and then separately find images depicting objects of the same class as those in the query image; see [2, 15, 22, 38], for example. 
이 문제에 대한 다양한 접근 방식이 있으며, 그 중 다수는 쿼리와 시각적으로 유사한 결과 집합을 찾은 다음 쿼리 이미지에있는 것과 동일한 클래스의 개체를 묘사하는 이미지를 개별적으로 찾습니다. 예를 들어 [2, 15, 22, 38] 참조.


Within the Information Retrieval community there has been considerable interest of late in incorporating user data into image retrieval systems [37], for example through browsing [36] and click-through behavior [26], or by making use of social tags [29]. 
정보 검색 커뮤니티 내에서 예를 들어 브라우징 [36] 및 클릭-스루 행동 [26] 또는 소셜 태그 사용 [29]을 통해 사용자 데이터를 이미지 검색 시스템 [37]에 통합하는 데 최근 상당한 관심이있었습니다. .


Also worth mentioning with respect to image retrieval is [12], which also considered using images crawled from Amazon, albeit for a different task (similar-image search) than the one considered here.
또한 이미지 검색과 관련하여 언급 할 가치가있는 것은 [12]인데, 여기에서 고려한 작업과는 다른 작업 (유사 이미지 검색)을 위해 Amazon에서 크롤링 한 이미지를 사용하는 것도 고려했습니다.

There have been a variety of approaches to modeling human notions of similarity between different types of images [30], forms of music [31], or even tweets [33], amongst other data types. 
다른 데이터 유형 중에서도 서로 다른 유형의 이미지 [30], 음악 형식 [31] 또는 트윗 [33] 사이의 유사성에 대한 인간 개념을 모델링하는 다양한 접근 방식이 있습니다.


Beyond measuring similarity, there has also been work on measuring more general notions of compatibility. 
유사성을 측정하는 것 외에도 호환성에 대한보다 일반적인 개념을 측정하는 작업도있었습니다.


Murillo et al. [25], for instance, analyze photos of groups of people collected from social media to identify which groups might be more likely to socialize with each other, thus implying a distance measure between images. 
Murillo et al. 예를 들어, 소셜 미디어에서 수집 된 사람들 그룹의 사진을 분석하여 어떤 그룹이 서로 어울릴 가능성이 더 높은지 식별하여 이미지 간의 거리 측정을 의미합니다.

This is achieved by estimating which of a manually-specified set of ‘urban tribes’ each group belongs to, possibly because only 340 images were available.
이것은 아마도 340 개의 이미지 만 사용할 수 있었기 때문에 각 그룹이 수동으로 지정한 '도시 부족'집합 중 어느 것이 속하는지 추정함으로써 달성됩니다.


Yamaguchi et al. [40] capture a notion of visual style when parsing clothing, but do so by retrieving visually similar items from a database. 
Yamaguchi et al. 의류를 분석 할 때 시각적 스타일의 개념을 포착하지만 데이터베이스에서 시각적으로 유사한 항목을 검색하여 수행합니다.


This idea was extended by Kiapour et al. [14] to identify discriminating characteristics between different styles (hipster vs. goth for example). 
이 아이디어는 Kiapour et al. [14] 서로 다른 스타일 (예 : hipster vs. goth) 간의 구별되는 특성을 식별합니다.


Di et al. [5] also identify aspects of style using a bag-of-words approach and manual annotations.
Di et al. [5] 또한 bag-of-words 접근 방식과 수동 주석을 사용하여 스타일의 측면을 식별합니다.


A few other works that consider visual features specifically for the task of clothing recommendation include [10, 13, 20]. 
의류 추천 작업을 위해 특별히 시각적 특징을 고려한 몇 가지 다른 작품으로는 [10, 13, 20]이 있습니다.


In [10] and [13] the authors build methods to parse complete outfits from single images, in [10] by building a carefully labeled dataset of street images annotated by ‘fashionistas’, and in [13] by building algorithms to automatically detect and segment items from clothing images. 
[10]과 [13]에서 저자는 단일 이미지에서 완전한 의상을 분석하는 방법을 구축하고, [10]에서는 'fashionistas'가 주석이 달린 거리 이미지의 신중하게 레이블이 지정된 데이터 세트를 구축하고, [13]에서는 자동 감지 알고리즘을 구축하여 의류 이미지에서 항목을 분류합니다.


In [13] the authors propose an approach to learn relationships between clothing items and events (e.g. birthday parties, funerals) in order to recommend eventappropriate items. 
[13]에서 저자는 이벤트에 적합한 항목을 추천하기 위해 의류 항목과 이벤트 (예 : 생일 파티, 장례식) 간의 관계를 학습하는 방법을 제안합니다.


Although related to our approach, these methods are designed for the specific task of clothing recommendation, requiring hand-crafted methods and carefully annotated data; in contrast our goal is to build a general-purpose method to understand relationships between objects from large volumes of unlabeled data. 
우리의 접근 방식과 관련이 있지만 이러한 방법은 의류 추천의 특정 작업을 위해 설계되었으며 수작업 방법과 신중하게 주석이 달린 데이터가 필요합니다. 반대로 우리의 목표는 레이블이 지정되지 않은 대량의 데이터에서 개체 간의 관계를 이해하는 범용 방법을 구축하는 것입니다.


Although our setting is perhaps most natural for categories like clothing images, we obtain surprisingly accurate performance when predicting relationships in a variety of categories, from recommending outfits to predicting which books will be co-purchased based on their cover art.
의류 이미지와 같은 카테고리에서는 설정이 가장 자연 스럽지만 의상 추천부터 커버 아트를 기반으로 공동 구매할 책 예측까지 다양한 카테고리의 관계를 예측할 때 놀라 울 정도로 정확한 성능을 얻습니다.

In summary, our approach is distinct from the above in that we aim to generalize the idea of a visual distance measure beyond measuring only similarity. 
요약하면, 우리의 접근 방식은 유사성 만 측정하는 것 이상의 시각적 거리 측정 개념을 일반화하는 것을 목표로한다는 점에서 위와 다릅니다.


Doing so demands a very large amount of training data, and our reluctance for manual annotation necessitates a more opportunistic data collection strategy. 
그렇게하려면 매우 많은 양의 교육 데이터가 필요하며 수동 주석 처리를 꺼리는 경우보다 기회주의적인 데이터 수집 전략이 필요합니다.


The scale of the data, and the fact that we don’t have control over its acquisition, demands a suitably scalable and robust modeling approach. 
데이터의 규모와 수집을 제어 할 수 없다는 사실로 인해 적절하게 확장 가능하고 강력한 모델링 접근 방식이 필요합니다.


The novelty in what we propose is thus in the quantity we choose to model, the data we gather to do so, and the method for extracting one from the other.
따라서 우리가 제안하는 참신함은 우리가 모델링하기로 선택한 양, 그렇게하기 위해 수집 한 데이터 및 하나에서 다른 하나를 추출하는 방법에 있습니다.


1.3 A visual and relational recommender system
1.3 시각적 및 관계형 추천 시스템


We label the process we develop for exploiting this data a visual and relational recommender system as we aim to model human visual preferences, and the system might be used to recommend one object on the basis of a user’s apparent interest in another. 
우리는 인간의 시각적 선호도를 모델링하는 것을 목표로이 데이터를 활용하기 위해 개발 한 프로세스를 시각적 및 관계형 추천 시스템이라고 표시하며,이 시스템은 사용자가 다른 것에 대한 명백한 관심을 기반으로 한 개체를 추천하는 데 사용될 수 있습니다.


The system shares these characteristics with more common forms of recommender system, but does so on the basis of the appearance of the object, rather than metadata, reviews, or similar.
시스템은 이러한 특성을 더 일반적인 형태의 추천 시스템과 공유하지만 메타 데이터, 리뷰 또는 이와 유사한 것이 아닌 객체의 모양을 기반으로합니다.

2. The Model

Our notation is defined in Table 2.
표기법은 표 2에 정의되어 있습니다.


We seek a method for representing the preferences of users for the visual appearance of one object given that of another. 
우리는 하나의 객체가 다른 객체의 시각적 인 모습에 대해 사용자의 선호도를 표현하는 방법을 찾고 있습니다.


A number of suitable models might be devised for this purpose, but very few of them will scale to the volume of data available. 
이러한 목적을 위해 여러 가지 적합한 모델이 고안 될 수 있지만 사용 가능한 데이터 양에 맞게 확장되는 모델은 거의 없습니다.


For every object in the dataset we calculate an F-dimensional feature vector x ∈ R F using a convolutional neural network as described in Section 2.3. 
데이터 세트의 모든 객체에 대해 2.3 절에 설명 된대로 컨벌루션 신경망을 사용하여 F 차원 특징 벡터 x ∈ R F를 계산합니다.


The dataset contains a set R of relationships where rij ∈ R relates objects i and j. 
데이터 세트에는 rij ∈ R이 객체 i 및 j와 관련된 관계의 세트 R이 포함됩니다.


Each relationship is of one of the four classes listed above. 
각 관계는 위에 나열된 네 가지 클래스 중 하나입니다.


Our goal is to learn a parameterized distance transform d(xi ,xj ) such that feature vectors {xi ,xj} for objects that are related (rij ∈ R) are assigned a lower distance than those that are not (rij ∈ R/ ). 
우리의 목표는 관련된 객체 (rij ∈ R)에 대한 특성 벡터 {xi, xj}가 그렇지 않은 객체 (rij ∈ R /)보다 낮은 거리에 할당되도록 매개 변수화 된 거리 변환 d (xi, xj)를 학습하는 것입니다. .


Specifically, we seek d(·,·) such that P(rij ∈ R) grows monotonically with −d(xi ,xj ).
특히, 우리는 P (rij ∈ R)가 −d (xi, xj)와 함께 단조롭게 성장하도록 d (·, ·)를 찾습니다.


Distances and probabilities: We use a shifted sigmoid function to relate distance to probability thus
거리와 확률 : 거리와 확률을 연관시키기 위해 이동 시그 모이 드 함수를 사용합니다.

(1)

This is depicted in Figure 2. 
이것은 그림 2에 묘사되어 있습니다.


This decision allows us to cast the problem as logistic regression, which we do for reasons of scalability. 
이 결정을 통해 문제를 로지스틱 회귀로 캐스트 할 수 있습니다.


Intuitively, if two items i and j have distance d(xi,xj ) = c, then they have probability 0.5 of being related; the probability increases above 0.5 for d(xi,xj ) < c, and decreases as d(xi,xj ) > c. 
직관적으로 두 항목 i와 j가 거리 d (xi, xj) = c를 가지면 관련 될 확률이 0.5입니다. 확률은 d (xi, xj) <c에 대해 0.5 이상으로 증가하고 d (xi, xj)> c로 감소합니다.


Note that we do not specify c in advance, but rather c is chosen to maximize prediction accuracy. 
c를 미리 지정하지 않고 예측 정확도를 최대화하기 위해 c를 선택했습니다.


We now describe a set of potential distance functions. 
이제 잠재적 인 거리 함수 세트를 설명합니다.


Weighted nearest neighbor: Given that different feature dimensions are likely to be more important to different relationships, the simplest method we consider is to learn which feature dimensions are relevant for a particular relationship. 
가중 최근 접 이웃 : 다른 특성 차원이 다른 관계에 더 중요 할 가능성이 있으므로 고려하는 가장 간단한 방법은 특정 관계와 관련된 특성 차원을 학습하는 것입니다.


We thus fit a distance function of the form
따라서 우리는 다음 형식의 거리 함수에 적합합니다.

(2)

Mahalanobis transform: (eq. 2) is limited to modeling the visual similarity between objects, albeit with varying emphasis per feature dimension. 
Mahalanobis 변환 : (eq. 2) 기능 차원에 따라 강조가 다양하지만 객체 간의 시각적 유사성을 모델링하는 것으로 제한됩니다.


It is not expressive enough to model subtler notions, such as which pairs of pants and shoes belong to the same ‘style’, despite having different appearances. 
외모가 다르지만 어떤 바지와 신발이 같은 '스타일'에 속하는지 등 미묘한 개념을 모델링하는 것은 표현력이 충분하지 않습니다.


For this we need to learn how different feature dimensions relate to each other, i.e., how the features of a pair of pants might be transformed to help identify a compatible pair of shoes. 
이를 위해 우리는 서로 다른 특성 차원이 어떻게 관련되는지, 즉 호환 가능한 신발을 식별하는 데 도움이되도록 바지 한 쌍의 특성을 변환하는 방법을 배워야합니다.


To identify such a transformation, we relate image features via a Mahalanobis distance, which essentially generalizes (eq. 2) so that weights are defined at the level of pairs of features. 
이러한 변환을 식별하기 위해 우리는 Mahalanobis 거리를 통해 이미지 특징을 연결합니다. 이는 본질적으로 일반화 (eq. 2)하여 가중치가 특징 쌍 수준에서 정의되도록합니다.


Specifically we fit
특히 우리는 적합합니다
(3)
A full rank p.s.d. matrix M has too many parameters to fit tractably given the size of the dataset. 
풀 랭크 p.s.d. 행렬 M에는 데이터 세트의 크기를 고려할 때 다루기 힘든 매개 변수가 너무 많습니다.


For example, using features with dimension F = 212, learning a transform as in (eq. 3) requires us to fit approximately 8 million parameters; not only would this be prone to overfitting, it is simply not practical for existing solvers. 
예를 들어, 차원 F = 212 인 특성을 사용하여 (등식 3)에서와 같이 변환을 학습하려면 약 8 백만 개의 매개 변수를 맞추어야합니다. 이는 과적 합되기 쉬울뿐만 아니라 기존 솔버에게는 실용적이지 않습니다.


To address these issues, and given the fact that M parameterises a Mahanalobis distance, we approximate M such that M ' YYT where Y is a matrix of dimension F × K. We therefore define 
(4)
Note that all distances (as well as their derivatives) can be computed in O(FK), which is significant for the scalability of the method. 
이러한 문제를 해결하기 위해 M이 Mahanalobis 거리를 매개 변수화한다는 사실을 고려하여 M 'YYT (여기서 Y는 차원 F × K의 행렬)가되도록 M을 근사합니다.
(4)
모든 거리 (및 그 파생물)는 O (FK)로 계산할 수 있으며, 이는 방법의 확장성에 중요합니다.


Similar ideas appear in [4, 35], which also consider the problem of metric learning via low-rank embeddings, albeit using a different objective than the one we consider here. 
유사한 아이디어가 [4, 35]에 나와 있는데, 여기에서는 여기에서 고려하는 것과는 다른 목표를 사용하지만 낮은 순위 임베딩을 통한 메트릭 학습 문제도 고려합니다.


2.1 Style space
In addition to being computationally useful, the low-rank transform in (eq. 4) has a convenient interpretation. 
2.1 스타일 공간
계산적으로 유용 할뿐만 아니라 (식 4)의 낮은 순위 변환은 편리한 해석을 제공합니다.


Specifically, if we consider the K-dimensional vector si = xiY, then (eq. 4) can be rewritten as 
특히 K 차원 벡터 si = xiY를 고려하면 (eq. 4)를 다음과 같이 다시 작성할 수 있습니다.
(5)
In other words, (eq. 4) yields a low-dimensional embedding of the features xi and xj . 
즉, (eq. 4)는 특성 xi 및 xj의 저 차원 임베딩을 생성합니다.


We refer to this low-dimensional representation as the product’s embedding into ‘style-space’, in the hope that we might identify Y such that related objects fall close to each other despite being visually dissimilar. 
우리는이 저차 원적 표현을 제품이 '스타일 공간'에 임베딩 된 것으로 지칭하며, 시각적으로 유사하지 않지만 관련 객체가 서로 가까이 떨어지도록 Y를 식별 할 수 있기를 바랍니다.


The notion of ‘style’ is learned automatically by training the model on pairs of objects which Amazon considers to be related. 
'스타일'이라는 개념은 Amazon이 관련이 있다고 간주하는 객체 쌍에 대해 모델을 학습함으로써 자동으로 학습됩니다.


2.2 Personalizing styles to individual users
2.2 개별 사용자에게 스타일 개인화

So far we have developed a model to learn a global notion of which products go together, by learning a notion of ‘style’ such that related products should have similar styles. As an addition to this model we can personalize this notion by learning for each individual user which dimensions of style they consider to be important.
지금까지 우리는 관련 제품이 비슷한 스타일을 가져야하는 '스타일'의 개념을 학습하여 어떤 제품이 결합되는지에 대한 글로벌 개념을 학습하는 모델을 개발했습니다. 이 모델에 추가하여 우리는 각 개별 사용자가 중요하다고 생각하는 스타일의 차원을 학습하여이 개념을 개인화 할 수 있습니다.


To do so, we shall learn personalized distance functions dY,u(xi , xj ) that measure the distance between the items i and j according to the user u. 
이를 위해 사용자 u에 따라 항목 i와 j 사이의 거리를 측정하는 개인화 된 거리 함수 dY, u (xi, xj)를 학습합니다.


We choose the distance function 
(6)
where D(u) is a K ×K diagonal (positive semidefinite) matrix.
거리 기능을 선택합니다
(6)
여기서 D (u)는 K × K 대각선 (양의 반정의) 행렬입니다.


In this way the entry D(u) kk indicates the extent to which the user u ‘cares about’ the k th style dimension.
이러한 방식으로 D (u) kk 항목은 사용자 u가 k 번째 스타일 차원에 대해 '관심'하는 정도를 나타냅니다.


In practice we fit a U × K matrix X such that D(u) kk = Xuk. 
실제로 우리는 D (u) kk = Xuk가되도록 U × K 행렬 X를 맞 춥니 다.


Much like the simplification in (eq. 5), the distance dY,u(xi, xj ) can be conveniently written as 
(7)
In other words, Xu is a personalized weighting of the projected style-space dimensions.
(식 5)의 단순화와 매우 유사하게 거리 dY, u (xi, xj)는 다음과 같이 편리하게 쓸 수 있습니다.
(7)
즉, Xu는 투영 된 스타일 공간 차원의 개인화 된 가중치입니다.


The construction in (eq. 6 and 7) only makes sense if there are users associated with each edge in our dataset, which is not true of the four graph types we have presented so far. 
(eq. 6 및 7)의 구성은 데이터 세트의 각 에지와 관련된 사용자가있는 경우에만 의미가 있으며 지금까지 제시 한 네 가지 그래프 유형에는 해당되지 않습니다.


Thus to study the issue of user personalization we make use of our rating and review data (see Table 1). 
따라서 사용자 개인화 문제를 연구하기 위해 평가 및 리뷰 데이터를 사용합니다 (표 1 참조).


From this we sample a dataset of triples (i,j,u) of products i and j that were both purchased by user u (i.e., u reviewed them both). 
이로부터 우리는 사용자 u가 구매 한 제품 i와 j의 트리플 (i, j, u) 데이터 세트를 샘플링합니다 (즉, u가 둘 다 검토 함).


We describe this further when we outline our experimental protocol in Section 4.1.
섹션 4.1에서 실험 프로토콜을 설명 할 때이를 더 자세히 설명합니다.

2.3 Features

Features are calculated from the original images using the Caffe deep learning framework [11]. 
특징은 Caffe 딥 러닝 프레임 워크를 사용하여 원본 이미지에서 계산됩니다 [11].


In particular, we used a Caffe reference model3 with 5 convolutional layers followed by 3 fully-connected layers, which has been pre-trained on 1.2 million ImageNet (ILSVRC2010) images. 
특히, 우리는 120 만 ImageNet (ILSVRC2010) 이미지에 대해 사전 학습 된 5 개의 컨볼 루션 레이어와 3 개의 완전 연결 레이어가있는 Caffe 참조 모델 3을 사용했습니다.


We use the output of FC7, the second fully-connected layer, which results in a feature vector of length F = 4096. 
두 번째 완전 연결 계층 인 FC7의 출력을 사용하여 길이 F = 4096의 특성 벡터를 생성합니다.


3. Training
Since we have defined a probability associated with the presence (or absence) of each relationship, we can proceed by maximizing the likelihood of an observed relationship set R. 
3. 훈련
각 관계의 존재 (또는 부재)와 관련된 확률을 정의 했으므로 관찰 된 관계 집합 R의 가능성을 최대화하여 진행할 수 있습니다.


In order to do so we randomly select a negative set Q = {rij |rij ∈/ R} such that |Q| = |R| and optimize the log likelihood 
(8)
Learning then proceeds by optimizing l(Y,c|R, Q) over both Y and c which we achieve by gradient ascent. 
그렇게하기 위해 무작위로 음의 집합 Q = {rij | rij ∈ / R}을 선택하여 | Q | = | R | 로그 가능성 최적화
(8)
그런 다음 기울기 상승으로 달성 한 Y와 c에 대해 l (Y, c | R, Q)를 최적화하여 학습을 진행합니다.


We use (hybrid) L-BFGS, a quasi-Newton method for non-linear optimization of problems with many variables [21]. 
우리는 많은 변수가있는 문제의 비선형 최적화를 위해 준 뉴턴 방법 인 (하이브리드) L-BFGS를 사용합니다 [21].


Likelihood (eq. 8) and derivative computations can be na¨ıvely parallelized over all pairs rij ∈ R ∪ Q. 
가능성 (등식 8) 및 미분 계산은 모든 쌍 rij ∈ R ∪ Q에 대해 순진하게 병렬화 될 수 있습니다.


Training on our largest dataset (Amazon books) with a rank K = 100 transform required around one day on a 12 core machine. 
가장 큰 데이터 세트 (Amazon 책)에 대한 교육 (랭크 K = 100 변환)은 12 코어 머신에서 하루 정도 필요합니다.

4. Experiments
We compare our model against the following baselines: 
다음 기준과 모델을 비교합니다.


We compare against Weighted Nearest Neighbor (WNN) classification, as is described in Section 1.3. 
섹션 1.3에 설명 된대로 WNN (Weighted Nearest Neighbor) 분류와 비교합니다.


We also compare against a method we label Category Tree (CT); CT is based on using Amazon’s detailed category tree directly (which we have collected for Clothing data, and use for later experiments), which allows us to assess how effective an image-based classification approach could be, if it were perfect. 
우리는 또한 카테고리 트리 (CT)라는 레이블을 붙인 방법과 비교합니다. CT는 Amazon의 세부 카테고리 트리를 직접 사용 (의류 데이터를 위해 수집하고 이후 실험에 사용)을 기반으로하므로 이미지 기반 분류 접근 방식이 완벽하다면 얼마나 효과적인지 평가할 수 있습니다.


We then compute a matrix of coocurrences between categories from the training data, and label two products (a,b) as ‘related’ if the category of b belongs to one of the top 50% of most commonly linked categories for products of category a. 
그런 다음 훈련 데이터에서 범주 간 동시성 행렬을 계산하고 b 범주가 범주 a의 제품에 대해 가장 일반적으로 연결된 범주의 상위 50 % 중 하나에 속하면 두 제품 (a, b)을 '관련됨'으로 표시합니다. .


Nearest neighbor results (calculated by optimizing a threshold on the `2 distance using the training data) were not significantly better than random, and have been suppressed for brevity. 
최근 접 이웃 결과 (훈련 데이터를 사용하여`2 거리에 대한 임계 값을 최적화하여 계산 됨)는 무작위보다 현저히 나아지지 않았으며 간결성을 위해 억제되었습니다.


Comparison against non-visual baselines As a non-visual comparison, we trained topic models on the reviews of each product (i.e., each document di is the set of reviews of the product i) and fit weighted nearest neighbor classifiers of the form 
(9)
where θi and θj are topic vectors derived from the reviews of the products i and j. 
비 시각적 기준에 대한 비교 비 시각적 비교로서, 우리는 각 제품의 리뷰에 대한 주제 모델을 훈련시키고 (즉, 각 문서 di는 제품 i의 리뷰 집합입니다) 양식의 가중 최근 접 이웃 분류기에 적합합니다.
(9)
여기서 θi 및 θj는 제품 i 및 j의 리뷰에서 파생 된 주제 벡터입니다.


In other words, we simply adapted our WNN baseline to make use of topic vectors rather than image features.
다시 말해, 우리는 단순히 이미지 특징보다는 주제 벡터를 사용하도록 WNN 기준선을 조정했습니다.


We used a 100-dimensional topic model trained using Vowpal Wabbit [8]. 
우리는 Vowpal Wabbit [8]을 사용하여 훈련 된 100 차원 주제 모델을 사용했습니다.


However, this baseline proved not to be competitive against the alternatives described above (e.g. only 60% accuracy on our largest dataset, ‘Books’). 
그러나이 기준은 위에서 설명한 대안에 비해 경쟁력이없는 것으로 입증되었습니다 (예 : 가장 큰 데이터 세트 인 'Books'에서 정확도가 60 %에 불과 함).


One explanation may simply be that is is difficult to effectively train topic models at the 1M+ document scale; another explanation is simply that the vast majority of products have few reviews. 
한 가지 설명은 단순히 1M 이상의 문서 규모에서 주제 모델을 효과적으로 교육하기 어렵다는 것입니다. 또 다른 설명은 대부분의 제품에 대한 리뷰가 거의 없다는 것입니다.


Not surprisingly, the number of reviews per product follows a power-law, e.g. for Men’s Clothing: 
당연히 제품 당 리뷰 수는 멱 법칙을 따릅니다. 남성 의류 :

그림-4

This issue is in fact exacerbated in our setting, as to predict a relationship between products we require both to have reliable feature representations, which will be true only if both products have several reviews. 
이 문제는 실제로 제품 간의 관계를 예측하기 위해 두 제품 모두 신뢰할 수있는 기능 표현을 필요로하며, 이는 두 제품 모두 여러 리뷰가있는 경우에만 해당됩니다.


Although we believe that predicting such relationships using text is a promising direction of future research (and one we are exploring), we simply wish to highlight the fact that there appears to be no ‘silver bullet’ to predict such relationships using text, primarily due to the ‘cold start’ issue that arises due to the long tail of obscure products with little text associated with them. 
텍스트를 사용하여 이러한 관계를 예측하는 것이 미래 연구의 유망한 방향이라고 생각하지만 (그리고 우리가 탐구중인), 텍스트를 사용하여 이러한 관계를 예측하는 데 '실버 총알'이 없다는 사실을 강조하고 싶습니다. 관련 텍스트가 거의없는 모호한 제품의 긴 꼬리로 인해 발생하는 '콜드 스타트'문제에 대한 것입니다.


Indeed, this is a strong argument in favor of building predictors based on visual features, since images are available even for brand new products which are yet to receive even a single review.
실제로 이것은 아직 단 한 번의 리뷰도받지 못한 새로운 제품에도 이미지를 사용할 수 있기 때문에 시각적 기능을 기반으로 예측 변수를 구축하는 데 유리한 강력한 주장입니다.

4.1 Experimental protocol

We split the dataset into its top-level categories (Books, Movies, Music, etc.) and further split the Clothing category into second-level categories (Men’s, Women’s, Boys, Girls, etc.). 
데이터 세트를 최상위 카테고리 (도서, 영화, 음악 등)로 나누고 의류 카테고리를 두 번째 수준 카테고리 (남성, 여성, 소년, 소녀 등)로 더 분할했습니다.


We focus on results from a few representative subcategories. 
우리는 몇 가지 대표적인 하위 범주의 결과에 중점을 둡니다.


Complete code for all experiments and all baselines is available online.
모든 실험 및 모든 기준에 대한 완전한 코드는 온라인으로 제공됩니다.


For each category, we consider the subset of relationships from R that connect products within that category. 
각 범주에 대해 해당 범주 내의 제품을 연결하는 R의 관계 하위 집합을 고려합니다.


After generating random samples of non-relationships, we separate R and Q into training, validation, and test sets (80/10/10%, up to a maximum of two million training relationships). 
비 관계의 무작위 샘플을 생성 한 후 R과 Q를 훈련, 검증 및 테스트 세트로 분리합니다 (80 / 10 / 10 %, 최대 2 백만 개의 훈련 관계).


Although we do not fit hyperparameters (and therefore do not make use of the validation set), we maintain this split in case it proves useful to those wishing to benchmark their algorithms on this data. 
하이퍼 파라미터에 적합하지 않지만 (따라서 검증 세트를 사용하지 않음),이 데이터에 대한 알고리즘을 벤치마킹하려는 사람들에게 유용하다고 입증되는 경우이 분할을 유지합니다.


While we did experiment with simple `2 regularizers, we found ourselves blessed with a sufficient overabundance of data that overfitting never presented an issue (i.e., the validation error was rarely significantly higher than the training error). 
우리는 간단한`2 정규화기로 실험을했지만 과적 합이 문제를 일으키지 않았던 충분한 데이터가 충분한 축복을 받았다는 것을 알았습니다 (즉, 검증 오류가 훈련 오류보다 거의 높지 않음).


To be completely clear, our protocol consists of the following: 
명확히하기 위해 우리의 프로토콜은 다음과 같이 구성됩니다.

1. Each category and graph type forms a single experiment (e.g. predict ‘bought together’ relationships for Women’s clothing).

2. Our goal is to distinguish relationships from non-relationships (i.e., link prediction). 
Relationships are identified when our predictor (eq. 1) outputs P(rij ∈ R) > 0.5.

3. We consider all positive relationships and a random sample of non-relationships (i.e., ‘distractors’) of equal size. 
Thus the performance of a random classifier is 50% for all experiments.

4. All results are reported on the test set. 
Results on a selection of top-level categories are shown in Table 4, with further results for clothing data shown in Table 

5. Recall when interpreting these results that the learned model has reference to the object images only. 
It is thus estimating the existence of a specified form of relationship purely on the basis of appearance.
1. 각 카테고리 및 그래프 유형은 하나의 실험을 형성합니다 (예 : 여성 의류에 대한 '함께 구매'관계 예측).

2. 우리의 목표는 관계와 비 관계 (즉, 링크 예측)를 구별하는 것입니다.
예측 변수 (식 1)가 P (rij ∈ R)> 0.5를 출력하면 관계가 식별됩니다.

3. 우리는 모든 긍정적 인 관계와 같은 크기의 비 관계 (즉, '산만하는 사람')의 무작위 표본을 고려합니다.
따라서 임의 분류기의 성능은 모든 실험에서 50 %입니다.

4. 모든 결과는 테스트 세트에보고됩니다.
선택한 최상위 범주에 대한 결과는 표 4에 나와 있으며 의류 데이터에 대한 추가 결과는 표에 나와 있습니다.

5. 이러한 결과를 해석 할 때 학습 된 모델이 대상 이미지 만 참조한다는 사실을 상기하십시오.
따라서 순전히 외모에 기초하여 특정 형태의 관계의 존재를 추정합니다.

In every case the proposed method outperforms both the category-based method and weighted nearest neighbor, and the increase from K = 10 to K = 100 uniformly improves performance. 
모든 경우에 제안 된 방법은 범주 기반 방법과 가중 최근 접 이웃 모두보다 성능이 우수하며 K = 10에서 K = 100으로 증가하면 성능이 균일하게 향상됩니다.


Interestingly, the performance on compliments vs. substitutes is approximately the same. 
흥미롭게도 칭찬과 대용품의 성과는 거의 동일합니다.


The extent to which the K = 100 results improve upon the WNN results may be seen as an indication of the degree to which visual similarity between images fails to capture a more complex human visual notion of which objects might be seen as being substitutes or compliments for each other. 
WNN 결과에서 K = 100 결과가 향상되는 정도는 이미지 간의 시각적 유사성이 어떤 대상이 대체물 또는 칭찬으로 보일 수 있는지에 대한 더 복잡한 인간의 시각적 개념을 포착하지 못하는 정도를 나타내는 것으로 볼 수 있습니다. 서로.


This distinction is smallest for ‘Books’ and greatest for ‘Clothing Shoes and Jewelery’ as might be expected. 
이 구분은 예상대로 '책'에서 가장 작고 '의류 신발 및 보석'에서 가장 큽니다.


We have no ground truth relating the true human visual preference for pairs of objects, of course, and thus evaluate above against our dataset. 
물론 개체 쌍에 대한 실제 인간의 시각적 선호도와 관련된 근거가 없으므로 데이터 세트에 대해 위에서 평가합니다.


This has the disadvantage that the dataset contains all of the Amazon recommendations, rather than just those based on decisions made by humans on the basis of object appearance. 
이것은 데이터 세트에 객체 모양을 기반으로 인간이 내린 결정에 기반한 것이 아니라 모든 Amazon 권장 사항이 포함되어 있다는 단점이 있습니다.


This means that in addition to documenting the performance of the proposed method, the results may also be taken to indicate the extent to which visual factors impact upon the decisions of Amazon customers. 
즉, 제안 된 방법의 성능을 문서화하는 것 외에도 결과를 통해 Amazon 고객의 결정에 시각적 요인이 영향을 미치는 정도를 나타낼 수도 있습니다.


The comparison across categories is particularly interesting. 
카테고리 간 비교는 특히 흥미 롭습니다.


It is to be expected that appearance would be a significant factor in Clothing decisions, but it was not expected that the purchase of Books. 
외모가 의복 결정에 중요한 요소가 될 것으로 예상되지만 책 구입은 예상되지 않았습니다.


One possible interpretation of this effect might be that customers have preferences for particular genres of books and that individual genres have characteristic styles of covers. 
이 효과에 대한 한 가지 가능한 해석은 고객이 특정 장르의 책에 대한 선호도를 가지고 있고 개별 장르가 독특한 스타일의 표지를 가지고 있다는 것입니다.

4.2 Personalized recommendations

Finally we evaluate the ability of our model to personalize copurchasing recommendations to individual users, that is we examine the effect of the user personalization term in (eqs. 6 and 7). 
마지막으로 개별 사용자에게 공동 구매 추천을 개인화하는 모델의 능력을 평가합니다. 즉, 사용자 개인화 기간의 효과를 조사합니다 (식 6 및 7).


Here we do not use the graphs from Tables 4 and 5, since those are ‘population level’ graphs which are not annotated in terms of the individual users who co-purchased and cobrowsed each pair of products. 
여기서는 표 4와 5의 그래프를 사용하지 않습니다. 이는 각 제품 쌍을 공동 구매하고 공동 검색 한 개별 사용자 측면에서 주석이 추가되지 않은 '인구 수준'그래프이기 때문입니다.


Instead for this task we build a dataset of co-purchases from products that users have reviewed. 
대신이 작업을 위해 사용자가 검토 한 제품의 공동 구매 데이터 세트를 작성합니다.


That is, we build a dataset of tuples of the form (i,j,u) for pairs of products i and j that were purchased by user u. 
즉, 사용자 u가 구매 한 제품 i 및 j 쌍에 대해 (i, j, u) 형식의 튜플 데이터 세트를 빌드합니다.


We train on users with at least 20 purchases, and randomly sample 50 co-purchases and 50 non-co-purchases from each user in order to build a balanced dataset. 
20 개 이상의 구매가있는 사용자에 대해 교육하고 균형 잡힌 데이터 세트를 구축하기 위해 각 사용자로부터 50 개의 공동 구매와 50 개의 비 공동 구매를 무작위로 샘플링합니다.


Results are shown in Table 3; here we see that the addition of a user personalization term yields a small but significant improvement when predicting co-purchases (similar results on other categories withheld for brevity).
결과는 표 3에 제시되어있다; 여기에서 사용자 개인화 용어를 추가하면 공동 구매를 예측할 때 작지만 상당한 개선이 이루어집니다 (간결성을 위해 다른 카테고리에 대한 유사한 결과는 보류 됨).

5. Visualizing Style Space

Recall that each image is projected into ‘style-space’ by the transformation si = xiY, and note that the fact that it is based on pairwise distances alone means that the embedding is invariant under isomorphism. 
각 이미지는 si = xiY 변환에 의해 '스타일 공간'으로 투영된다는 점을 상기하고, 이것이 쌍 단위 거리만을 기반으로한다는 사실은 임베딩이 동형 하에서 불변임을 의미합니다.


That is, applying rotations, translations, or reflections to si and sj will preserve their distance in (eq. 5). 
즉, si와 sj에 회전, 평행 이동 또는 반사를 적용하면 거리가 (eq. 5)에서 유지됩니다.


In light of these factors we perform k-means clustering on the K dimensional embedded coordinates of the data in order to visualize the effect of the embedding. 
이러한 요소를 고려하여 임베딩의 효과를 시각화하기 위해 데이터의 K 차원 임베디드 좌표에 k- 평균 클러스터링을 수행합니다.


Figure 3 shows images whose projections are close to the centers of a set of selected representative clusters for Men’s and Women’s clothing (using a model trained on the ‘also viewed’ graph with K = 100). 
그림 3은 남성 및 여성 의류에 대해 선택된 대표 클러스터 세트의 중심에 가까운 투영 이미지를 보여줍니다 (K = 100 인 '또한 본'그래프에서 학습 된 모델 사용).


Naturally items cluster around colors and shapes (e.g. shoes, t-shirts, tank tops, watches, jewelery), but more subtle characterizations exist as well. 
당연히 항목은 색상과 모양 (예 : 신발, 티셔츠, 탱크 탑, 시계, 보석류) 주변에 모여 있지만 더 미묘한 특성도 존재합니다.


For instance, leather boots are separated from ugg (that is sheep skin) boots, despite the fact that the visual differences are subtle. 
예를 들어, 가죽 부츠는 시각적 차이가 미묘하다는 사실에도 불구하고 어그 (즉 양가죽) 부츠와 구분됩니다.


This is presumably because these items are preferred by different sets of Amazon users. 
이는 아마도 이러한 항목이 다른 Amazon 사용자 집합에 의해 선호되기 때문일 것입니다.


Watches cluster into different color profiles, face shapes, and digital versus analogue. 
시계는 다양한 색상 프로필, 얼굴 모양, 디지털 대 아날로그로 클러스터됩니다.


Other clusters cross multiple categories, for instance we find clusters of highlycolorful items, items containing love hearts, and items containing animals. 
다른 클러스터는 여러 범주에 걸쳐 있습니다. 예를 들어 매우 다채로운 항목, 사랑의 마음을 포함하는 항목 및 동물이 포함 된 항목의 클러스터를 찾습니다.


Figure 4 shows a set of images which project to locations that span a cluster.
그림 4는 클러스터에 걸쳐있는 위치에 투영되는 이미지 세트를 보여줍니다.


Although performance is admittedly not outstanding for a category such as books, it is somewhat surprising that an accuracy of even 70% can be achieved when predicting book co-purchases. 
책과 같은 카테고리에서 성능이 뛰어나지는 않지만 책 공동 구매를 예측할 때 70 %의 정확도를 얻을 수 있다는 것은 다소 놀랍습니다.


Figure 5 visualizes a few examples of stylespace clusters derived from Books data. 
그림 5는 Books 데이터에서 파생 된 스타일 공간 클러스터의 몇 가지 예를 시각화합니다.


Here it seems that there is at least some meaningful information in the cover of a book to predict which products might be purchased together— children’s books, self-help books, romance novels, and comics (for example) all seem to have characteristic visual features which are identified by our model.
여기에서는 함께 구매할 수있는 제품을 예측하기 위해 책 표지에 의미있는 정보가 적어도 몇 가지있는 것 같습니다. 아동 도서, 자조 책, 로맨스 소설, 만화 (예 :) 모두 특징적인 시각적 특징이있는 것 같습니다. 우리 모델로 식별됩니다.


In Figure 6 we show how our model can be used to navigate between related items—here we randomly select two items that are unlikely to be co-browsed, and find a low cost path between them as measured by our learned distance measure. 
그림 6에서는 모델을 사용하여 관련 항목 사이를 탐색하는 방법을 보여줍니다. 여기서는 공동 탐색 할 가능성이없는 두 항목을 무작위로 선택하고 학습 된 거리 측정으로 측정 한 항목 사이의 저렴한 경로를 찾습니다.


Subjectively, the model identifies visually smooth transitions between the source and the target items.
주관적으로 모델은 소스와 대상 항목 사이의 시각적으로 부드러운 전환을 식별합니다.

Figure 7 provides a visualization of the embedding of Boys clothing achieved by setting K = 2 (on co-browsing data).
그림 7은 K = 2 (공동 브라우징 데이터에서)로 설정하여 달성 한 Boys 의류 임베딩의 시각화를 제공합니다.


Sporting shoes drift smoothly toward slippers and sandals, and underwear drifts gradually toward shirts and coats.
운동화는 슬리퍼와 샌들쪽으로 부드럽게 드리프트하고 속옷은 셔츠와 코트쪽으로 서서히 드리프트합니다.

6. Generating Recommendations

We here demonstrate that the proposed model can be used to generate recommendations that might be useful to a user of a web store. 
여기에서는 제안 된 모델을 사용하여 웹 스토어 사용자에게 유용 할 수있는 권장 사항을 생성 할 수 있음을 보여줍니다.


Given a query item (e.g. a product a user is currently browsing, or has just purchased), our goal is to recommend a selection of other items that might complement it. 
쿼리 항목 (예 : 사용자가 현재 탐색 중이거나 방금 구매 한 제품)이 주어지면이를 보완 할 수있는 다른 항목을 추천하는 것이 목표입니다.


For example, if a user is browsing pants, we might want to recommend a shirt, shoes, or accessories that belong to the same style. 
예를 들어 사용자가 바지를 탐색하는 경우 동일한 스타일에 속하는 셔츠, 신발 또는 액세서리를 추천 할 수 있습니다.


Here, Amazon’s rich and detailed category hierarchy can help us. 
여기에서 Amazon의 풍부하고 상세한 카테고리 계층 구조가 도움이 될 수 있습니다.


For categories such as women’s or men’s clothing, we might define an ‘outfit’ as a combination of pants, a top, shoes, and an accessory (we do this for the sake of demonstration, though far more complex combinations are possible—our category tree for clothing alone has hundreds of nodes). 
여성용 또는 남성용 의류와 같은 카테고리의 경우 '의상'을 바지,상의, 신발 및 액세서리의 조합으로 정의 할 수 있습니다 (더 복잡한 조합이 가능하지만 데모를 위해이 작업을 수행합니다. 옷 나무에만 수백 개의 노드가 있습니다).


Then, given a query item our goal is simply to select items from each of these categories that are most likely to be connected based on their visual style. 
그런 다음 쿼리 항목이 주어지면 우리의 목표는 단순히 시각적 스타일을 기반으로 연결될 가능성이 가장 높은 각 범주에서 항목을 선택하는 것입니다.


Specifically, given a query item xq, for each category C (represented as a set of item indices), we generate recommendations according to
특히 쿼리 항목 xq가 주어지면 각 범주 C (항목 색인 집합으로 표시됨)에 대해 다음에 따라 권장 사항을 생성합니다.


(10)
i.e., the minimum distance according to our measure (eq. 4) amongst objects belonging to the desired category. 
(10)
즉, 원하는 범주에 속하는 물체 사이의 측정 (식 4)에 따른 최소 거리입니다.


Examples of such recommendations are shown in Figures 1 and 8, with randomly chosen queries from women’s and men’s clothing. 
이러한 권장 사항의 예는 여성 및 남성 의류에서 무작위로 선택된 쿼리와 함께 그림 1과 8에 나와 있습니다.


Generally speaking the model produces apparently reasonable recommendations, with clothes in each category usually being of a consistent style. 
일반적으로 모델은 일반적으로 일관된 스타일의 각 카테고리의 옷과 함께 합리적인 권장 사항을 생성합니다.

7. Outfits in The Wild

An alternate application of the model is to make assessments about outfits (or otherwise combinations of items) that we observe ‘in the wild’. 
모델의 또 다른 적용은 우리가 '야생에서'관찰하는 의상 (또는 기타 항목의 조합)에 대한 평가를하는 것입니다.


That is, to the extent that the tastes and preferences of Amazon customers reflect the zeitgeist of society at large, this can be seen as a measurement of whether a candidate outfit is well coordinated visually.
즉, 아마존 고객의 취향과 선호도가 전반적으로 사회의 시대 정신을 반영하는 한 후보 의상이 시각적으로 잘 조화되어 있는지를 측정 한 것으로 볼 수있다.


To assess this possibility, we have built two small datasets of real outfits, one consisting of twenty-five outfits worn by the hosts of Top Gear (Jeremy Clarkson, Richard Hammond, and James May), and another consisting of seventeen ‘before’ and ‘after’ pairs of outfits from participants on the television show What Not to Wear (US seasons 9 and 10). 
이 가능성을 평가하기 위해 우리는 Top Gear의 호스트 (Jeremy Clarkson, Richard Hammond, James May)가 착용 한 25 개의 의상으로 구성된 실제 의상의 두 개의 작은 데이터 세트를 구축했으며, 다른 하나는 17 개의 'before'및 TV 쇼 What Not to Wear (미국 시즌 9 및 10) 참가자의 '애프터'의상.


For each outfit, we cropped each clothing item from the image, and then used Google’s reverse image search to identify images of similar items (examples are shown in Figure 9).
각 의상에 대해 이미지에서 각 의류 항목을 자른 다음 Google의 역 이미지 검색을 사용하여 유사한 항목의 이미지를 식별했습니다 (예는 그림 9에 표시됨).


Next we rank outfits according to the average log-likelihood of their pairs of components being related using a model trained on Men’s/Women’s co-purchases (we take the average so that there is no bias toward outfits with more or fewer components).
다음으로 남성 / 여성 공동 구매에 대해 학습 된 모델을 사용하여 관련 구성 요소 쌍의 평균 로그 가능성에 따라 의상 순위를 매 깁니다 (구성 요소가 많거나 적은 의상에 대한 편견이 없도록 평균을 취합니다).


All outfits have at least two items. 
모든 의상에는 최소 2 개의 아이템이 있습니다.


Figure 9 shows the most and least coordinated outfits on Top Gear; here we find considerable separation between the level of coordination for each presenter; Richard Hammond is typically the least coordinated, James May the most, while Jeremy Clarkson wears a combination of highly coordinated and highly uncoordinated outfits.
그림 9는 Top Gear에서 가장 많이 그리고 가장 적게 조정 된 의상을 보여줍니다. 여기서 우리는 각 발표자에 대한 조정 수준 사이에 상당한 분리를 발견했습니다. Richard Hammond는 일반적으로 가장 덜 코디네이터, James May가 가장 많은 반면 Jeremy Clarkson은 높은 코디네이터와 코디되지 않은 의상의 조합을 입습니다.


A slightly more quantitative evaluation comes from the television show What Not to Wear: here participants receive an ‘outfit makeover’, hopefully meaning that their made-over outfit is more coordinated than the original. 
TV 쇼 What Not to Wear에서 약간 더 정량적 인 평가가 나옵니다. 여기에서 참가자들은 '의상 화장'을 받게되는데, 이는 그들이 만든 옷이 원래 옷보다 더 잘 조화되어 있다는 것을 의미합니다.


Examples of participants before and after their makeover, along with the change in log likelihood are shown in Figure 10. 
로그 가능성의 변화와 함께 화장 전후 참가자의 예가 그림 10에 나와 있습니다.


Indeed we find that made-over outfits have a higher log likelihood in 12 of the 17 cases we observed (p ' 7%; log-likelihoods are normalized to correct any potential bias due to the number of components in the outfit). 
실제로 우리가 관찰 한 17 개 케이스 중 12 개에서 만든 의상이 로그 가능성이 더 높다는 것을 발견했습니다 (p '7 %; 의상의 구성 요소 수로 인한 잠재적 편향을 수정하기 위해 로그 가능성이 정규화 됨).


This is an important result, as it provides external (albeit small) validation of the learned model which is independent of our dataset.
이것은 우리의 데이터 세트와 독립적 인 학습 된 모델의 외부 (작지만) 검증을 제공하기 때문에 중요한 결과입니다.

8. Conclusion

We have shown that it is possible to model the human notion of what is visually related by investigation of a suitably large dataset, even where that information is somewhat tangentially contained therein. 
우리는 적절하게 큰 데이터 세트를 조사함으로써 시각적으로 관련된 것에 대한 인간의 개념을 모델링 할 수 있음을 보여주었습니다. 심지어 그 정보가 그 안에 다소 접선 적으로 포함되어있는 경우에도 마찬가지입니다.


We have also demonstrated that the proposed method is capable of modeling a variety of visual relationships beyond simple visual similarity. 
또한 제안 된 방법은 단순한 시각적 유사성을 넘어 다양한 시각적 관계를 모델링 할 수 있음을 입증했습니다.


Perhaps what distinguishes our method most is thus its ability to model what makes items complementary. 
아마도 우리의 방법을 가장 구별하는 것은 항목을 보완하는 것을 모델링하는 능력 일 것입니다.


To our knowledge this is the first attempt to model human preference for the appearance of one object given that of another in terms of more than just the visual similarity between the two. 
우리가 아는 한 이것은 둘 사이의 시각적 유사성 이상의 측면에서 다른 객체의 모양에 대한 인간의 선호도를 모델링하려는 첫 번째 시도입니다.


It is almost certainly the first time that it has been attempted directly and at this scale.
이 규모로 직접 시도한 것은 거의 확실합니다.


We also proposed visual and relational recommender systems as a potential problem of interest to the information retrieval community, and provided a large dataset for their training and evaluation. 
또한 정보 검색 커뮤니티가 관심을 가질만한 잠재적 인 문제로 시각 및 관계형 추천 시스템을 제안하고 교육 및 평가를위한 대규모 데이터 세트를 제공했습니다.


In the process we managed to figure out what not to wear, how to judge a book by its cover, and to show that James May is more fashionable than Richard Hammond. 
그 과정에서 우리는 무엇을 입지 말아야하는지, 표지로 책을 판단하는 방법을 파악하고, James May가 Richard Hammond보다 더 유행한다는 것을 보여주었습니다.


Acknowledgements. 

This research was supported by the Data 2 Decisions Cooperative Research Centre, and the Australian Research Council Discovery Projects funding scheme DP140102270.

감사합니다.

이 연구는 Data 2 Decisions Cooperative Research Centre 및 Australian Research Council Discovery Projects 기금 계획 DP140102270의 지원을 받았습니다.

---