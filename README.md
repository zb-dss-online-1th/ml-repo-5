# 피파온라인4의 승부예측 모델 구축 프로젝트
---
## 주제 및 목표
* 피파온라인4의 승리 예측 모형을 구축
* 피파온라인4의 공식경기 랭킹 TOP 1000명의 매치 기록을 각 유저당 30경기씩 수집하여, 이를 바탕으로 승리 예측 모형 구축
---
## 기획의도 및 배경
게임과 실제 스포츠 모두 일반인과 프로 선수간 엄청난 차이가 있는것은 사실이다.

프로선수와 아마추어 유저가 대회에서 맞붙지 않는 이상 서로간의 실력차이를 가늠하기란 어려운것이다.

그래서 과연 프로와 아마추어의 실력차이를 눈으로 체감 할 수 있는 지표가 있으면 좋겠다는 궁금증으로 부터 출발하였다.

그리고 앞으로 있을 중국 광저우 아시안게임에서 e-sport 종목으로 피파온라인4가 공식 채택되었고,

세계에서 가장 큰 국제대회인 월드컵이 개최될 예정이다.

축구에 관해 사람들의 이목이 집중될 이벤트들이 많이 있기에 온라인 게임으로나마 많은 사람들이 간접적으로 축구를 즐길것이고, 

자연스레 승부예측에 대해서도 관심을 가질것으로 예상해 피파온라인4의 데이터들을 가지고 승리를 예측해볼수있는 모델을 구상하기 되었다.

![image](https://user-images.githubusercontent.com/98334298/168064328-35da83d8-184a-4856-acbf-2f23f9170b2f.png)
![image](https://user-images.githubusercontent.com/98334298/168064625-8e0bde2e-90e0-4e2b-93ba-db445374b3a3.png)

---
## 선행자료 조사

|분류|주제|데이터|데이터 상세분류|사용 분석 기법|
|------|------|------|----|------|
|Reference1|2000–2018 EPL 기록의 데이터를 바탕으로 승리 예측|현실 데이터|N|SVM / Logistic Regression / K-Nearest Neighbor / Naive Bayes / Decision tree / ANN / Keras — ANN|
|Reference2|시간대에 따른 본인의 공식경기 매칭 수와 승률 분석|게임 데이터|Y|ANOVA / 로지스틱 회귀분석|
|5조|피파 온라인4의 공식경기 랭킹 TOP 1000 명의 데이터로 승리 예측|게임 데이터|Y|로지스틱 회귀분석 / 랜덤 포레스트 / SVM / LGBM|

* Reference1은 Kaggle에 있는 2000–2018 EPL 기록의 데이터를 바탕으로 승리 예측 모형에 관한 것이다. 이 데이터는 현실 세계의 축구 선수들이 직접 경기한 데이터를 바탕으로 구축되었으며, 슛 횟수, 패스횟수 등 여러가지 데이터가 있다. 하지만 게임 데이터에는 패스 횟수만 있는것이 아니라 숏패스, 롱패스, 스루패스등 각 데이터가 상세하게 분류되어 있다.
* Reference2는 hojjimin님 본인의 공식경기 API 데이터를 바탕으로 시간대에 따른 매칭수와 승률을 분석에 관한 것이다. 피파 온라인  API를 이용하여 게임 데이터를 수집했다는 점에서 동일하나, 수집한 데이터가 본인의 데이터라는 점, 그리고 랭킹 hojjimin님은 10,000안에 들지 못한 플레이어라는 점에서 본 조에서 수집한 데이터와의 질이 다르다.
---
## 데이터 수집 및 특성
* 넥슨 API를 이용한 데이터 수집
* 테이블 정의서(EDA 결과에 따라 변동 사항 있을 수 있음)

|name1|데이터1|유형1|설명1|name2|데이터2|유형2|설명2|
|------|----|---|---|------|----|---|---|
|MatchDetailDTO|seasonId|Integer|시즌 ID|ShootDTO|shootTotal|Integer|총 슛 수|
||matchResult|String|매치 결과 (“승”,“패”)||effectiveShootTotal|Integer|총 유효슛 수|
||matchEndType|Integer|매치종료 타입 (0: 정상종료)||shootOutScore|Integer|승부차기 슛 수|
||systemPause|Integer|게임 일시정지 수||goalTotal|Integer|총 골 수 (실제 골 수)goalInPenalty+goalOutPenalty+goalPenaltyKick|
||foul|Integer|파울 수||goalTotalDisplay|Integer|게임 종료 후 유저에게 노출되는 골 수|
||injury|Integer|부상 수||ownGoal|Integer|자책 골 수|
||redCards|Integer|받은 레드카드 수||shootHeading|Integer|헤딩 슛 수|
||yellowCards|Integer|받은 옐로카드 수||goalHeading|Integer|헤딩 골 수|
||dribble|Integer|드리블 거리(야드)||shootFreekick|Integer|프리킥 슛 수|
||cornerKick|Integer|코너킥 수||goalFreekick|Integer|프리킥 골 수|
||possession|Integer|점유율||shootInPenalty|Integer|인패널티 슛 수|
||OffsideCount|Integer|오프사이드 수||goalInPenalty|Integer|인패널티 골 수|
||averageRating|Double|경기 평점||shootOutPenalty|Integer|아웃패널티 슛 수|
||controller|String|사용한 컨트롤러 타입 (keyboard / pad / etc 중 1)||goalOutPenalty|Integer|아웃패널티 골 수|
|PassDTO|passTry|Integer|패스 시도 수||shootPenaltyKick|Integer|패널티킥 슛 수|
||passSuccess|Integer|패스 성공 수||goalPenaltyKick|Integer|패널티킥 골 수|
||shortPassTry|Integer|숏 패스 시도 수||blockTry|Integer|블락 시도 수|
||shortPassSuccess|Integer|숏 패스 성공 수|DefenceDTO|blockSuccess|Integer|블락 성공 수|
||longPassTry|Integer|롱 패스 시도 수||tackleTry|Integer|태클 시도 수|
||longPassSuccess|Integer|롱 패스 성공 수||tackleSuccess|Integer|태클 성공 수|
||bouncingLobPassTry|Integer|바운싱 롭 패스 시도 수|||||
||bouncingLobPassSuccess|Integer|바운싱 롭 패스 성공 수|||||
||drivenGroundPassTry|Integer|드리븐 땅볼 패스 시도 수|||||
||drivenGroundPassSuccess|Integer|드리븐 땅볼 패스 성공 수|||||
||throughPassTry|Integer|스루 패스 시도 수|||||
||throughPassSuccess|Integer|스루 패스 성공 수|||||
||lobbedThroughPassTry|Integer|로빙 스루 패스 시도 수|||||
||lobbedThroughPassSuccess|Integer|로빙 스루 패스 성공 수|||||
---
## 모델
- 로지스틱 회귀 분석
- 랜덤 포레스트
- SVM
- LightGBM
---
## 모델 평가
* 프로선수 김정민 (前성남 FC(소속 프로게이머), 前 T1, 現 SADDLER 소속)의 최근 공식경기 데이터
* 전 시즌 공식경기 랭커 TOP 1 유저 닉네임'강테아부지'의 최근 공식경기 데이터
* 두 유저의 데이터를 가지고 모델의 성능을 테스트했다.
---
## 사용할 프레임워크 및 라이브러리
1. json
2. requests
3. pandas 
4. numpy
5. seaborn 
6. matplotlib.pyplot 
7. bs4
  * BeautifulSoup
8. urllib.request
  * urlopen
9. sklearn.ensemble
  * RandomForestRegressor
10. sklearn.model_selection
  * GridSearchCV
  * RandomizedSearchCV
  * train_test_split
11. sklearn.metrics
  * r2_score
  * mean_absolute_error
  * mean_squared_error
  * mean_squared_log_error
  * accuracy_score
  * precision_score
  * recall_score
  * f1_score
  * roc_auc_score
  * confusion_matrix
12. sklearn.preprocessing
  * RobustScaler
  * StandardScaler
  * MinMaxScaler
13. sklearn.decomposition
  * PCA
14. sklearn.linear_model
  * LogisticRegression
15. sklearn.svm 
  * SVR
16. lightgbm
  * LGBMRegressor

## 참고문헌(자료)
- Refernce1 [https://medium.com/@bjmoon.korea/ai-x-%EB%94%A5%EB%9F%AC%EB%8B%9D-fianl-assignment-84e66d7e451d](https://medium.com/@bjmoon.korea/ai-x-%EB%94%A5%EB%9F%AC%EB%8B%9D-fianl-assignment-84e66d7e451d)
- Refernce2 [https://github.com/HojinHwang/FIFA-Online4](https://github.com/HojinHwang/FIFA-Online4)
- [https://hojjimin-statistic.tistory.com/10](https://hojjimin-statistic.tistory.com/10)
- [http://www.gameinsight.co.kr/news/articleView.html?idxno=16078](http://www.gameinsight.co.kr/news/articleView.html?idxno=16078)
