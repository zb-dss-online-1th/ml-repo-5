import json
import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.request import urlopen

api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2NvdW50X2lkIjoiMTQwOTM3NzUwOSIsImF1dGhfaWQiOiIyIiwidG9rZW5fdHlwZSI6IkFjY2Vzc1Rva2VuIiwic2VydmljZV9pZCI6IjQzMDAxMTQ4MSIsIlgtQXBwLVJhdGUtTGltaXQiOiI1MDA6MTAiLCJuYmYiOjE2NTEwNDg0NTEsImV4cCI6MTY2NjYwMDQ1MSwiaWF0IjoxNjUxMDQ4NDUxfQ.H2Rcu3IHdrajcam3_WE8leINtSGRx-oyRBh1WPA7e0g"
headers = {'Authorization': api_key}

# 각 시즌별로 top 1000의 플레이어의 공식 경기의 30경기 크롤링
# TOP 1000 이름 목록 가져오기
names = []
for k in range(1, 51):
    url = f'https://fifaonline4.nexon.com/datacenter/rank?n4seasonno=49&n4pageno={k}'
    soup = BeautifulSoup(urlopen(url), 'html.parser')
    outer = soup.find_all(attrs={'class': 'name profile_pointer'})
    
    for i in range(20):
        names.append(outer[i].text)

# TOP 1000 고유ID 가져오기 / 가끔 없는 애들도 있음
uniqID = []
for name in names:
    try:
        url = "https://api.nexon.co.kr/fifaonline4/v1.0/users?nickname="
        nickName = name
        full = url + nickName
        
        resGet = requests.get(full, headers = headers)
        resJson = resGet.json()
        uniqID.append(resJson["accessId"])
    
    except:
        continue

# TOP 1000 인당 공식 경기ID 30개씩 가져오기
matchID = []
for iD in uniqID:
    accessid = iD
    match = 50
    offset = 0 
    limit = 30
    full = f"https://api.nexon.co.kr/fifaonline4/v1.0/users/{accessid}/matches?matchtype={match}&offset={offset}&limit={limit}"
    
    resGet = requests.get(full, headers = headers)
    resJson = resGet.json()
    matchID.append(resJson)

# 각 경기 데이터 가져오기
matchData = []
for md in matchID:
    for mdi in md:
        url = f"https://api.nexon.co.kr/fifaonline4/v1.0/matches/{mdi}"
        match_url = requests.get(url, headers = headers)
        match_json = match_url.json()
        match_detail = pd.DataFrame(match_json)
        for i in range(2):
            if len(match_detail) == 2:
                b0 = {'matchId': match_detail['matchId'][0]}
                d0 = {"nickname": match_detail['matchInfo'][i]['nickname']}
                d1 = match_detail['matchInfo'][i]['matchDetail']
                d2 = match_detail['matchInfo'][i]['shoot']
                d3 = match_detail['matchInfo'][i]['pass']
                d4 = match_detail['matchInfo'][i]['defence']
            
                dic = dict(b0, **d0, **d1, **d2, **d3, **d4)
                matchData.append(dic)

raw_df = pd.DataFrame(matchData)
raw_df.to_csv('zero_ml_raw.csv', encoding='cp949')

#%% 프로선수 SaddlerJungmin, 랭커 캉테아부지 데이터 가져오기

api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2NvdW50X2lkIjoiMTQwOTM3NzUwOSIsImF1dGhfaWQiOiIyIiwidG9rZW5fdHlwZSI6IkFjY2Vzc1Rva2VuIiwic2VydmljZV9pZCI6IjQzMDAxMTQ4MSIsIlgtQXBwLVJhdGUtTGltaXQiOiI1MDA6MTAiLCJuYmYiOjE2NTEwNDg0NTEsImV4cCI6MTY2NjYwMDQ1MSwiaWF0IjoxNjUxMDQ4NDUxfQ.H2Rcu3IHdrajcam3_WE8leINtSGRx-oyRBh1WPA7e0g"
headers = {'Authorization': api_key}

def get_player(name, offset, limit):
    uniqID = []
    url = "https://api.nexon.co.kr/fifaonline4/v1.0/users?nickname="
    nickName = name
    full = url + nickName
    
    resGet = requests.get(full, headers = headers)
    resJson = resGet.json()
    uniqID.append(resJson["accessId"])
    
    matchID = []
    for iD in uniqID:
        accessid = iD
        match = 50
        offset = offset
        limit = limit
        full = f"https://api.nexon.co.kr/fifaonline4/v1.0/users/{accessid}/matches?matchtype={match}&offset={offset}&limit={limit}"
        
        resGet = requests.get(full, headers = headers)
        resJson = resGet.json()
        matchID.append(resJson)
    
    global matchData
    matchData = []
    for md in matchID:
        for mdi in md:
            url = f"https://api.nexon.co.kr/fifaonline4/v1.0/matches/{mdi}"
            match_url = requests.get(url, headers = headers)
            match_json = match_url.json()
            match_detail = pd.DataFrame(match_json)
            for i in range(2):
                if len(match_detail) == 2:
                    b0 = {'matchId': match_detail['matchId'][0]}
                    d0 = {"nickname": match_detail['matchInfo'][i]['nickname']}
                    d1 = match_detail['matchInfo'][i]['matchDetail']
                    d2 = match_detail['matchInfo'][i]['shoot']
                    d3 = match_detail['matchInfo'][i]['pass']
                    d4 = match_detail['matchInfo'][i]['defence']
                
                    dic = dict(b0, **d0, **d1, **d2, **d3, **d4)
                    matchData.append(dic)
                else:
                    print(matchID.index(md))

get_player('SaddlerJungmin', 500, 100)
raw_fin_pro_tmp1 = matchData
get_player('SaddlerJungmin', 600, 100)
raw_fin_pro_tmp2 = matchData
raw_fin_pro_df = pd.DataFrame(raw_fin_pro_tmp1 + raw_fin_pro_tmp2)
raw_fin_pro_df.to_csv('zero_ml_pro_raw.csv', encoding='cp949')

get_player('캉테아부지')
raw_fin_ama_df = pd.DataFrame(matchData)
raw_fin_ama_df.to_csv('zero_ml_ama_raw.csv', encoding='cp949')

#%% 데이터 전처리
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error

raw_df = pd.read_csv('zero_ml_raw.csv', encoding='cp949', index_col=0)
raw_df.columns

print(len(raw_df[raw_df['matchResult'] == '오류']))
print(len(raw_df[raw_df['matchResult'] == '무']))
print(len(raw_df[raw_df['matchEndType'] != 0]))

raw_df = raw_df.drop(index=raw_df[raw_df['matchEndType'] != 0].index)
raw_df = raw_df.drop(index=raw_df[raw_df['matchResult'] == '무'].index)
raw_df.reset_index(drop=True, inplace=True)

raw_df.duplicated().sum()
raw_df = raw_df.drop_duplicates()

raw_df['matchResult'].replace('승', 1, inplace=True)
raw_df['matchResult'].replace('패', 0, inplace=True)

raw_df.info()
raw_df.columns
pre_df = raw_df.drop(['matchId', 'nickname', 'seasonId', 'matchEndType', 'controller',
                      'systemPause', 'goalTotalDisplay', 'matchResult', 'goalInPenalty',
                      'goalOutPenalty', 'goalPenaltyKick', 'averageRating'], axis=1)

plt.figure(figsize = (30, 27))
sns.boxplot(data = pre_df, orient='h')
plt.show()

# goalTotal	Integer	총 골 수 (실제 골 수) goalInPenalty+goalOutPenalty+goalPenaltyKick
pre_df.columns
X = pre_df.drop(['goalTotal'], axis=1)
y = pre_df['goalTotal']

#%% 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix

def print_eval(y_train, y_pred):
    print('R^2:', r2_score(y_train, y_pred))
    print('MAE:', mean_absolute_error(y_train, y_pred))
    print('MSE:', mean_squared_error(y_train, y_pred))
    print('RMSE:', np.sqrt(mean_squared_error(y_train, y_pred)))

def get_clf_eval(y_test, pred):
    acc = accuracy_score(y_test, pred)
    pre = precision_score(y_test, pred)
    re = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    auc = roc_auc_score(y_test, pred)
    return acc, pre, re, f1, auc

def print_clf_eval(y_test, pred):
    confusion = confusion_matrix(y_test, pred)
    acc, pre, re, f1, auc = get_clf_eval(y_test, pred)
    
    print('==> Confusion matrix')
    print(confusion)
    print('====================')
    print('Accuracy: {0:.4f}, Precision: {1:.4f}'.format(acc, pre))
    print('Recall: {0:.4f}, F1: {1:.4f}, AUC: {2:.4f}'.format(re, f1, auc))
    
#%% split data
from sklearn.model_selection import train_test_split

X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.2, random_state=13)
X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.25, random_state=13)
print("X_train Data shape: {}".format(X_train.shape))
print("X_val Data shape: {}".format(X_val.shape))
print("X_test Data shape: {}".format(X_test.shape))

#%% scaling
from sklearn.preprocessing import RobustScaler

rs = RobustScaler()
X_train_rs = rs.fit_transform(X_train)
X_val_rs = rs.transform(X_val)
X_test_rs = rs.transform(X_test)

X_train_rs = pd.DataFrame(X_train_rs)
X_train_rs.columns = X.columns

X_val_rs = pd.DataFrame(X_val_rs)
X_val_rs.columns = X.columns

X_test_rs = pd.DataFrame(X_test_rs)
X_test_rs.columns = X.columns

plt.figure(figsize = (30, 27))
sns.boxplot(data=X_train_rs, orient='h')
plt.show()

#%% pca
from sklearn.decomposition import PCA

pca = PCA(n_components = 15).fit(X_train_rs)
X_train_rs_pca = pca.transform(X_train_rs)
print('eigen_value :', pca.explained_variance_)
print('cum_explained variance ratio :', sum(pca.explained_variance_ratio_))

plt.figure(figsize = (10, 8))
plt.plot(range(1, pca.n_components+1), np.cumsum(pca.explained_variance_ratio_), 'o-')
plt.grid()
plt.title('Cumulative Explained Variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance');

plt.figure(figsize = (10, 8))
plt.plot(range(1, pca.n_components+1), pca.explained_variance_ratio_, 'o-')
plt.grid()
plt.title('Scree Plot')
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance Ratio');

# 최종 
rs_pca = PCA(n_components = 8).fit(X_train_rs)
X_train_rs_pca = rs_pca.transform(X_train_rs)
X_val_rs_pca = rs_pca.transform(X_val_rs)
X_test_rs_pca = rs_pca.transform(X_test_rs)
print(f"Original shape: {X_train_rs.shape}")
print(f"Reduced shape: {X_train_rs_pca.shape}")
print('eigen_value :', rs_pca.explained_variance_)
print('cum_explained variance ratio :', sum(rs_pca.explained_variance_ratio_))

X_train_rs_pca = pd.DataFrame(X_train_rs_pca)
X_train_rs_pca = pd.DataFrame(X_train_rs_pca)

#%% feature_importances_

rf_clf_rs = RandomForestRegressor(random_state=13, n_jobs=-1)
rf_clf_rs.fit(X_train_rs, y_train)

best_cols_values = rf_clf_rs.feature_importances_
best_cols = pd.Series(best_cols_values, index=X_train_rs.columns)
topn_cols = best_cols.sort_values(ascending=False)[:8]

plt.figure(figsize=(8, 8))
plt.title('Feature Importances')
sns.barplot(x=topn_cols, y=topn_cols.index)
plt.show()

X_train_rs_fi = X_train_rs[topn_cols.index]
X_val_rs_fi = X_val_rs[topn_cols.index]
X_test_rs_fi = X_test_rs[topn_cols.index]

rf_clf = RandomForestRegressor(random_state=13, n_jobs=-1)
rf_clf.fit(X_train, y_train)

best_cols_values = rf_clf.feature_importances_
best_cols = pd.Series(best_cols_values, index=X_train_rs.columns)
topn_cols = best_cols.sort_values(ascending=False)[:8]

plt.figure(figsize=(8, 8))
plt.title('Feature Importances')
sns.barplot(x=topn_cols, y=topn_cols.index)
plt.show()

X_train_fi = X_train[topn_cols.index]
X_val_fi = X_val[topn_cols.index]
X_test_fi = X_test[topn_cols.index]

#%% 
rf_clf = RandomForestRegressor(random_state=13, n_jobs=-1)
rf_clf.fit(X_train, y_train)

best_cols_values = rf_clf.feature_importances_
best_cols = pd.Series(best_cols_values, index=X_train.columns)
top20_cols = best_cols.sort_values(ascending=False)[:20]

plt.figure(figsize=(8, 8))
plt.title('Feature Importances')
sns.barplot(x=top20_cols, y=top20_cols.index)
plt.show()

sns.pairplot(data=X_train[top20_cols.index])
plt.show()

#%% random forest로 전반적인 테스트
# mse기준

rf = RandomForestRegressor(random_state=13, n_jobs=-1).fit(X_train, y_train)
rf_pred_train = rf.predict(X_train)
rf_pred_val = rf.predict(X_val)
print_eval(y_train, rf_pred_train) # 0.12257088739388115
print_eval(y_val, rf_pred_val) # 0.8757745915425276
print(rf.score(X_train, y_train), # 0.9344017957978894 
rf.score(X_test, y_test)) # 0.5332829562210031

rf = RandomForestRegressor(random_state=13, n_jobs=-1).fit(X_train_rs, y_train)
rf_rs_pred_train = rf.predict(X_train_rs)
rf_rs_pred_val = rf.predict(X_val_rs)
print_eval(y_train, rf_rs_pred_train) # 0.12260488146724331
print_eval(y_val, rf_rs_pred_val) # 0.8757876982220085
print(rf.score(X_train_rs, y_train), # 0.93438360265094
rf.score(X_val_rs, y_val)) # 0.5332759714206216

rf = RandomForestRegressor(random_state=13, n_jobs=-1).fit(X_train_rs_pca, y_train)
rf_pca_pred_train = rf.predict(X_train_rs_pca)
rf_pca_pred_val = rf.predict(X_val_rs_pca)
print_eval(y_train, rf_pca_pred_train) # 0.19094902691013935
print_eval(y_val, rf_pca_pred_val) # 1.3652304300816915
print(rf.score(X_train_rs_pca, y_train), # 0.8978001202426175
rf.score(X_val_rs_pca, y_val)) # 0.271592358591077

rf = RandomForestRegressor(random_state=13, n_jobs=-1).fit(X_train_fi, y_train)
rf_fi_pred_train = rf.predict(X_train_fi)
rf_fi_pred_val = rf.predict(X_val_fi)
print_eval(y_train, rf_fi_pred_train) # 0.13983110350258957
print_eval(y_val, rf_fi_pred_val) # 0.9954383600465856
print(rf.score(X_train_fi, y_train), # 0.9251643724182808
rf.score(X_val_fi, y_val)) # 0.4695118433992672

rf = RandomForestRegressor(random_state=13, n_jobs=-1).fit(X_train_rs_fi, y_train)
rf_rs_fi_pred_train = rf.predict(X_train_rs_fi)
rf_rs_fi_pred_val = rf.predict(X_val_rs_fi)
print_eval(y_train, rf_rs_fi_pred_train) # 0.13972188851513695
print_eval(y_val, rf_rs_fi_pred_val) # 0.9949562520856694
print(rf.score(X_train_rs_fi, y_train), # 0.9252228227338588
rf.score(X_val_rs_fi, y_val)) # 0.4697687679600777 

''' 결론: 뭐가 문젠지 자꾸 과적합이 발생 '''

#%% 다시 데이터 전처리
pre_df = raw_df.drop(['matchId', 'nickname', 'seasonId', 'matchEndType', 'controller',
                      'systemPause', 'goalTotalDisplay', 'matchResult', 'goalInPenalty',
                      'goalOutPenalty', 'goalPenaltyKick', 'averageRating'], axis=1)
# 이상치 제거
for i in pre_df.columns:
    q1 = pre_df[i].quantile(0.25)
    q3 = pre_df[i].quantile(0.75)
    iqr = q3-q1
    condition = pre_df[i]>q3+1.5*iqr
    a = pre_df[condition].index
    pre_df.drop(a, inplace=True)

plt.figure(figsize = (30, 27))
sns.boxplot(data = pre_df, orient='h')
plt.show()

pre_df.drop(['dribble', 'throughPassSuccess', 'passSuccess'], axis=1, inplace=True)

pre_df.columns

X = pre_df.drop(['goalTotal'], axis=1)
y = pre_df['goalTotal']

X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.2, random_state=13)
X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.25, random_state=13)
print("X_train Data shape: {}".format(X_train.shape))
print("X_val Data shape: {}".format(X_val.shape))
print("X_test Data shape: {}".format(X_test.shape))

# 
rf_clf = RandomForestRegressor(random_state=13, n_jobs=-1).fit(X_train, y_train)
best_cols_values = rf_clf.feature_importances_
best_cols = pd.Series(best_cols_values, index=X_train.columns)
topn_cols = best_cols.sort_values(ascending=False)[:8]

plt.figure(figsize=(8, 8))
plt.title('Feature Importances')
sns.barplot(x=topn_cols, y=topn_cols.index)
plt.show()

X_train_fi = X_train[topn_cols.index]
X_val_fi = X_val[topn_cols.index]
X_test_fi = X_test[topn_cols.index]

# 이상치 제거했으니 standard scaler
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train_fi)
X_val_sc = sc.transform(X_val_fi)
X_test_sc = sc.transform(X_test_fi)

X_train_sc = pd.DataFrame(X_train_sc)
X_train_sc.columns = X_train_fi.columns

X_val_sc = pd.DataFrame(X_val_sc)
X_val_sc.columns = X_train_fi.columns

X_test_sc = pd.DataFrame(X_test_sc)
X_test_sc.columns = X_train_fi.columns

plt.figure()
sns.boxplot(data=X_train_sc, orient='h')
plt.show()

for i in X_train_fi.columns:
    f, ax = plt.subplots(figsize = (10, 6))
    sns.distplot(X_train_fi[i])

rf = RandomForestRegressor(max_depth= 50, random_state=13, n_jobs=-1).fit(X_train_sc, y_train)
rf_fi_pred_train = rf.predict(X_train_sc)
rf_fi_pred_val = rf.predict(X_val_sc)
print_eval(y_train, rf_fi_pred_train) # 0.13983110350258957
print_eval(y_val, rf_fi_pred_val) # 0.9954383600465856
print(rf.score(X_train_sc, y_train), # 0.9251643724182808
rf.score(X_val_sc, y_val)) # 0.4695118433992672

''' 뭔가 문젠지 과적합이 해결이 안됨 minmax로 다시'''

#%% 다시 
pre_df = raw_df.drop(['matchId', 'nickname', 'seasonId', 'matchEndType', 'controller',
                      'systemPause', 'goalTotalDisplay', 'matchResult', 'goalInPenalty',
                      'goalOutPenalty', 'goalPenaltyKick', 'averageRating'], axis=1)

pre_df.drop(['dribble', 'throughPassSuccess', 'tackleSuccess', 'passSuccess', 'drivenGroundPassSuccess',
             'shortPassSuccess', 'lobbedThroughPassSuccess', 'bouncingLobPassSuccess'], axis=1, inplace=True)

pre_df.columns

X = pre_df.drop(['goalTotal'], axis=1)
y = pre_df['goalTotal']

X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.2, random_state=13)
X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.25, random_state=13)
print("X_train Data shape: {}".format(X_train.shape))
print("X_val Data shape: {}".format(X_val.shape))
print("X_test Data shape: {}".format(X_test.shape))

# 
rf_clf = RandomForestRegressor(random_state=13, n_jobs=-1).fit(X_train, y_train)
best_cols_values = rf_clf.feature_importances_
best_cols = pd.Series(best_cols_values, index=X_train.columns)
topn_cols = best_cols.sort_values(ascending=False)[:8]

plt.figure(figsize=(8, 8))
plt.title('Feature Importances')
sns.barplot(x=topn_cols, y=topn_cols.index)
plt.show()

X_train_fi = X_train[topn_cols.index]
X_val_fi = X_val[topn_cols.index]
X_test_fi = X_test[topn_cols.index]

from sklearn.preprocessing import MinMaxScaler

mm = MinMaxScaler()
X_train_mm = mm.fit_transform(X_train_fi)
X_val_mm = mm.transform(X_val_fi)
X_test_mm = mm.transform(X_test_fi)

X_train_mm = pd.DataFrame(X_train_mm)
X_train_mm.columns = X_train_fi.columns

X_val_mm = pd.DataFrame(X_val_mm)
X_val_mm.columns = X_train_fi.columns

X_test_mm = pd.DataFrame(X_test_mm)
X_test_mm.columns = X_train_fi.columns


plt.figure(figsize = (30, 27))
sns.boxplot(data=X_train_mm, orient='h')
plt.show()

rf = RandomForestRegressor(n_estimators= 500, max_depth= 50, max_features=1, 
                            random_state=13, n_jobs=-1).fit(X_train_mm, y_train)
rf_fi_pred_train = rf.predict(X_train_mm)
rf_fi_pred_val = rf.predict(X_val_mm)
print_eval(y_train, rf_fi_pred_train) # 0.9270138798383085
print_eval(y_val, rf_fi_pred_val) # 0.4133478244145603

''' 결론: 그래도 해결 안됨 하...... 사실 과적합이 아닐 지도?? 그냥 이대로 진행 '''

#%% 혹시 모르니 변수 바꿔가며 ㄱㄱ
for i in range(1 , len(X_train_mm.columns)+1):
    print('컬럼 개수', X_train_mm.iloc[:,:i].columns)
    X_train_ex1 = X_train_mm.iloc[:,:i]
    X_val_ex1 = X_val_mm.iloc[:,:i]
    
    clf = RandomForestRegressor(random_state=13, n_jobs=-1)
    h_para = {'n_estimators': range(100, 1500, 15),
              'max_depth': range(1, 150, 15),
              'max_features': range(1, len(X_train_mm.iloc[:,:i].columns)+1)}
    rf_ex = RandomizedSearchCV(clf, h_para, scoring='neg_mean_squared_error', cv=5, 
                                  n_jobs = -1, verbose=1, random_state=13)
    rf_ex.fit(X_train_ex1, y_train)
    print(rf_ex.best_params_, f'{rf_ex.best_score_:.4f}')

    rf_ex_best = rf_ex.best_estimator_
    rf_ex_best.fit(X_train_ex1, y_train)
    
    rf_ex_pred_train = rf_ex_best.predict(X_train_ex1)
    rf_ex_pred_val = rf_ex_best.predict(X_val_ex1)
    print_eval(y_train, rf_ex_pred_train)
    print_eval(y_val, rf_ex_pred_val)

''' 결론: 순서대로 넣으면 초반에 train의 r^2도 0.4이렇게 나오다가 특성이 들어올수록 r^2 상승, 그렇다고
    val의 r^2는 올라가지 않음 그냥 이대로 진행 또한, 이 정도면 하이퍼파라미터 크게 신경 쓸 필요 없을 듯'''
    
#%%
# heatmap
f, ax = plt.subplots(figsize = (15, 15))
sns.heatmap(pre_df.corr(), cmap='coolwarm', annot=True, linewidths=.5, fmt='.2f', ax=ax)
plt.show()

# pairplot
sns.pairplot(pre_df, hue = "matchResult", height = 2.5)
plt.tight_layout()
plt.show()

# boxplot
plt.figure(figsize = (30, 27))
sns.boxplot(data = pre_df, orient='h')
plt.show()

sns.distplot(pre_df['passTry'])

# 이거 몇개 뺴고 전부 이산형이라 크게 의미 없음
X.columns
sns.distplot(pre_df['goalFreekick'])

for col in X: 
    print(f'{col:24}', '\t',
          f'Skewness: {X[col].skew():.2f}', '\t',
          f'Kurtosis: {X[col].kurt():.2f}')

#%% random forest
''' random search '''
clf = RandomForestRegressor(random_state=13, n_jobs=-1)
h_para = {'n_estimators': range(100, 1500, 15),
          'max_depth': range(1, 150, 15),
          'max_features': range(1, 8)}
rf_rs = RandomizedSearchCV(clf, h_para, scoring='neg_mean_squared_error', n_iter = 10, cv = 5, 
                           verbose=1, random_state=13, n_jobs = -1)
rf_rs.fit(X_train_mm, y_train)
print(rf_rs.best_params_, f'{rf_rs.best_score_:.4f}')
# {'n_estimators': 1240, 'max_features': 3, 'max_depth': 16} -0.9434

''' The score of random search best_parameters '''
rf_rs_best = rf_rs.best_estimator_
rf_rs_best.fit(X_train_mm, y_train)

rf_rs_best_pred_train = rf_rs_best.predict(X_train_mm)
rf_rs_best_pred_val = rf_rs_best.predict(X_val_mm)
print_eval(y_train, rf_rs_best_pred_train) # 0.8173557708659263
print_eval(y_val, rf_rs_best_pred_val) # 0.4964133926649701

#%% logistic regression
from sklearn.linear_model import LogisticRegression

''' gridsearch '''
clf = LogisticRegression(max_iter=10000, random_state=13, n_jobs=-1)
h_para = [{'C': np.logspace(-3, 3, 25), 'penalty': [ 'l1', 'l2'], 'solver': ['liblinear']},
          {'C': np.logspace(-3, 3, 25), 'penalty': ['l2'], 'solver': ['lbfgs']}]
lr_gs = GridSearchCV(clf, h_para, scoring = 'neg_mean_squared_error', cv = 5, n_jobs = -1)
lr_gs.fit(X_train_mm, y_train)
print(lr_gs.best_params_, f'{lr_gs.best_score_:.4f}')
# {'C': 10.0, 'penalty': 'l2', 'solver': 'lbfgs'} -1.0290

''' The score of grid search best_parameters '''
lr_gs_best = lr_gs.best_estimator_
lr_gs_best.fit(X_train_mm, y_train)

lr_gs_best_pred_train = lr_gs_best.predict(X_train_mm)
lr_gs_best_pred_val = lr_gs_best.predict(X_val_mm)
print_eval(y_train, lr_gs_best_pred_train) # 0.4494692439207324
print_eval(y_val, lr_gs_best_pred_val) # 0.4559410298603491

#%% SVM
from sklearn.svm import SVR

''' random search '''
clf = SVR()
h_para = [{'kernel': [ 'linear' ], 'C': np.logspace(-3, 3, 6), 'epsilon': np.logspace(-3, 3, 6)},
          {'kernel': ['rbf'], 'C': np.logspace(-3, 3, 6), 'epsilon': np.logspace(-3, 3, 6), 'gamma': np.logspace(-3, 3, 6)}]
svm_rs = RandomizedSearchCV(clf, h_para, scoring = 'neg_mean_squared_error', cv = 5, verbose=1, n_jobs = -1)
svm_rs.fit(X_train_mm, y_train)
print(svm_rs.best_params_, f'{svm_rs.best_score_:.4f}')
# {'kernel': 'rbf', 'gamma': 0.015848931924611134, 'epsilon': 0.015848931924611134, 'C': 63.0957344480193} -0.9202

''' The score of random search best_parameters '''
svm_rs_best = svm_rs.best_estimator_
svm_rs_best.fit(X_train_mm, y_train)

svm_rs_best_pred_train = svm_rs_best.predict(X_train_mm)
svm_rs_best_pred_val = svm_rs_best.predict(X_val_mm)
print_eval(y_train, svm_rs_best_pred_train) # 0.5089214480857307
print_eval(y_val, svm_rs_best_pred_val) # 0.507081536303746

#%% LightGBM
from lightgbm import LGBMRegressor

clf = LGBMRegressor(random_state=13, n_jobs=-1)
h_para = {'n_estimators': range(1, 1500, 100), 'max_depth': range(1, 1500, 15),
          'min_child_samples': range(1, 5000, 100), 'learning_rate': np.logspace(-3, 3, 20)}
lgbm_rs = RandomizedSearchCV(clf, h_para, cv = 5, scoring = 'neg_mean_squared_error', n_jobs = -1)
lgbm_rs.fit(X_train_mm, y_train)
print(lgbm_rs.best_params_, f'{lgbm_rs.best_score_:.4f}')
# {'n_estimators': 1201, 'min_child_samples': 1101, 'max_depth': 361, 'learning_rate': 0.07847599703514611} -0.9524

''' The score of random search best_parameters '''
lgbm_rs_best = lgbm_rs.best_estimator_
lgbm_rs_best.fit(X_train_mm, y_train)

lgbm_rs_best_pred_train = lgbm_rs_best.predict(X_train_mm)
lgbm_rs_best_pred_val = lgbm_rs_best.predict(X_val_mm)
print_eval(y_train, lgbm_rs_best_pred_train) # 0.5660083590987288
print_eval(y_val, lgbm_rs_best_pred_val) # 0.4934274150599618

#%% 최종 Model Evaluation, ( r^2  / RMSE  )

''' RandomForestRegressor '''
bp =  {'n_estimators': 1240, 'max_features': 3, 'max_depth': 16} # -0.9434
rf_rs_fin_best = RandomForestRegressor(**bp, random_state=13, n_jobs=-1)
rf_rs_fin_best.fit(X_train_mm, y_train)

rf_rs_fin_best_pred_train = rf_rs_fin_best.predict(X_train_mm)
rf_rs_fin_best_pred_val = rf_rs_fin_best.predict(X_val_mm)
print_eval(y_train, rf_rs_fin_best_pred_train) # 0.8173557708659263 / 0.5841853575658337
print_eval(y_val, rf_rs_fin_best_pred_val) # 0.4964133926649701 / 0.9720898889784317

''' LogisticRegression '''
bp = {'C': 6.158482110660261, 'penalty': 'l2', 'solver': 'lbfgs'} # -0.8503
lr_gs_fin_best = LogisticRegression(**bp, n_jobs=-1)
lr_gs_fin_best.fit(X_train_mm, y_train)

lr_gs_fin_best_pred_train = lr_gs_fin_best.predict(X_train_mm)
lr_gs_fin_best_pred_val = lr_gs_fin_best.predict(X_val_mm)
print_eval(y_train, lr_gs_fin_best_pred_train) # 0.4481404948208837 / 1.0154579722867405
print_eval(y_val, lr_gs_fin_best_pred_val) # 0.4539563478087689 / 1.012238896156851

''' SVR '''
bp = {'kernel': 'rbf', 'gamma': 0.015848931924611134, 'epsilon': 0.015848931924611134, 'C': 63.0957344480193} # -0.9202
svm_rs_fin_best = SVR(**bp,)
svm_rs_fin_best.fit(X_train_mm, y_train)

svm_rs_fin_best_pred_train = svm_rs_fin_best.predict(X_train_mm)
svm_rs_fin_best_pred_val = svm_rs_fin_best.predict(X_val_mm)
print_eval(y_train, svm_rs_fin_best_pred_train) # 0.5089214480857307 / 0.9579066168842726
print_eval(y_val, svm_rs_fin_best_pred_val) # 0.507081536303746 / 0.9617382370097467

''' LGBMRegressor '''
bp = {'n_estimators': 1201, 'min_child_samples': 1101, 'max_depth': 361, 'learning_rate': 0.07847599703514611} # -0.9524
lgbm_rs_fin_best = LGBMRegressor(**bp, random_state=13, n_jobs=-1)
lgbm_rs_fin_best.fit(X_train_mm, y_train)

lgbm_rs_fin_best_pred_train = lgbm_rs_fin_best.predict(X_train_mm)
lgbm_rs_fin_best_pred_val = lgbm_rs_fin_best.predict(X_val_mm)
print_eval(y_train, lgbm_rs_fin_best_pred_train) # 0.5660083590987288 / 0.900509652368427
print_eval(y_val, lgbm_rs_fin_best_pred_val) # 0.4934274150599618 / 0.9749675951794595

''' 과적합이 되더라도 차라리 train MSE가 가장 낮은 RandomForestRegressor 선택 '''
rf_rs_fin_best_pred_test = rf_rs_fin_best.predict(X_test_mm)
print_eval(y_test, rf_rs_fin_best_pred_test) # 0.5018588883775862 / 0.9482042238571879

#%% final pro
raw_fin_pro_df = pd.read_csv('zero_ml_pro_raw.csv', encoding='cp949', index_col=0)

print(len(raw_fin_pro_df[raw_fin_pro_df['matchResult'] == '무']))
print(len(raw_fin_pro_df[raw_fin_pro_df['matchEndType'] != 0]))

raw_fin_pro_df.drop(index=raw_fin_pro_df[raw_fin_pro_df['matchEndType'] != 0].index, inplace=True)
raw_fin_pro_df.drop(index=raw_fin_pro_df[raw_fin_pro_df['matchResult'] == '무'].index, inplace=True)
raw_fin_pro_df.duplicated().sum()
raw_fin_pro_df = raw_fin_pro_df.drop_duplicates()
raw_fin_pro_df.reset_index(drop=True, inplace=True)
raw_fin_pro_df.columns
drop_col = ['matchId', 'nickname', 'seasonId', 'matchEndType', 'controller',
            'systemPause', 'goalTotalDisplay', 'matchResult', 'goalInPenalty',
            'goalOutPenalty', 'goalPenaltyKick', 'averageRating', 'dribble', 
            'throughPassSuccess', 'tackleSuccess', 'passSuccess', 'shortPassSuccess']

pred_goal_list = []
pred_win_list = []
for i in range(int(len(raw_fin_pro_df)/2)):
    level1 = raw_fin_pro_df.iloc[2*i:2*i+2]
    for j in range(2):
        level2 = pd.DataFrame(level1.iloc[j]).T
        level2 = level2.drop(drop_col, axis=1)

        X_pro = level2.drop(['goalTotal'], axis=1)
        y_pro = level2['goalTotal']
        
        X_pro_fi = X_pro[topn_cols.index]

        X_pro_fi_mm = mm.transform(X_pro_fi)
        X_pro_fi_mm = pd.DataFrame(X_pro_fi_mm)
        X_pro_fi_mm.columns = X_pro_fi.columns
        
        rf_rs_best_pred_fin = rf_rs_fin_best.predict(X_pro_fi_mm)
        pred_goal_list.append(rf_rs_best_pred_fin[0])
    
    if pred_goal_list[-1] < pred_goal_list[-2]:
        pred_win_list.append(1)
        pred_win_list.append(0)
    else:
        pred_win_list.append(0)
        pred_win_list.append(1)

pro_real_win = raw_fin_pro_df['matchResult']
pro_real_goal = raw_fin_pro_df['goalTotal']
raw_fin_pro_df['matchResult'].replace('승', 1, inplace=True)
raw_fin_pro_df['matchResult'].replace('패', 0, inplace=True)
pro_pred_goal_list = pd.Series(pred_goal_list)
pro_pred_win_list = pd.Series(pred_win_list)

pro_result_df = pd.concat([pro_real_win, pro_real_goal, pro_pred_win_list, pro_pred_goal_list], axis=1)
pro_result_df.rename(columns={0: 'pred_matchResult', 1: 'pred_goalTotal'}, inplace=True)

pro_real_win_rate = len(pro_result_df[pro_result_df['matchResult'] == 1]) / len(pro_result_df['matchResult'])
pro_pred_win_rate = len(pro_result_df[pro_result_df['pred_matchResult'] == 1]) / len(pro_result_df['pred_matchResult'])

print_clf_eval(pro_real_win, pro_pred_win_list)
print('진짜 승률', pro_real_win_rate)
print('예측 승률', pro_real_win_rate)

#%% final ama
raw_fin_ama_df = pd.read_csv('zero_ml_ama_raw.csv', encoding='cp949', index_col=0)

print(len(raw_fin_ama_df[raw_fin_ama_df['matchResult'] == '무']))
print(len(raw_fin_ama_df[raw_fin_ama_df['matchEndType'] != 0]))

raw_fin_ama_df.drop(index=raw_fin_ama_df[raw_fin_ama_df['matchEndType'] != 0].index, inplace=True)
raw_fin_ama_df.drop(index=raw_fin_ama_df[raw_fin_ama_df['matchResult'] == '무'].index, inplace=True)
raw_fin_ama_df.duplicated().sum()
raw_fin_ama_df = raw_fin_ama_df.drop_duplicates()
raw_fin_ama_df.reset_index(drop=True, inplace=True)

drop_col = ['matchId', 'nickname', 'seasonId', 'matchEndType', 'controller',
            'systemPause', 'goalTotalDisplay', 'matchResult', 'goalInPenalty',
            'goalOutPenalty', 'goalPenaltyKick', 'averageRating', 'dribble', 
            'throughPassSuccess', 'tackleSuccess', 'passSuccess', 'shortPassSuccess']

pred_goal_list = []
pred_win_list = []
for i in range(int(len(raw_fin_ama_df)/2)):
    level1 = raw_fin_ama_df.iloc[2*i:2*i+2]
    for j in range(2):
        level2 = pd.DataFrame(level1.iloc[j]).T
        level2 = level2.drop(drop_col, axis=1)

        X_pro = level2.drop(['goalTotal'], axis=1)
        y_pro = level2['goalTotal']
        
        X_pro_fi = X_pro[topn_cols.index]

        X_pro_fi_mm = mm.transform(X_pro_fi)
        X_pro_fi_mm = pd.DataFrame(X_pro_fi_mm)
        X_pro_fi_mm.columns = X_pro_fi.columns
        
        rf_rs_best_pred_fin = rf_rs_fin_best.predict(X_pro_fi_mm)
        pred_goal_list.append(rf_rs_best_pred_fin[0])
    
    if pred_goal_list[-1] < pred_goal_list[-2]:
        pred_win_list.append(1)
        pred_win_list.append(0)
    else:
        pred_win_list.append(0)
        pred_win_list.append(1)

ama_real_win = raw_fin_ama_df['matchResult']
ama_real_goal = raw_fin_ama_df['goalTotal']
raw_fin_ama_df['matchResult'].replace('승', 1, inplace=True)
raw_fin_ama_df['matchResult'].replace('패', 0, inplace=True)
ama_pred_goal_list = pd.Series(pred_goal_list)
ama_pred_win_list = pd.Series(pred_win_list)

ama_result_df = pd.concat([ama_real_win, ama_real_goal, ama_pred_win_list, ama_pred_goal_list], axis=1)
ama_result_df.rename(columns={0: 'pred_matchResult', 1: 'pred_goalTotal'}, inplace=True)

ama_real_win_rate = len(ama_result_df[ama_result_df['matchResult'] == 1]) / len(ama_result_df['matchResult'])
ama_pred_win_rate = len(ama_result_df[ama_result_df['pred_matchResult'] == 1]) / len(ama_result_df['pred_matchResult'])

print_clf_eval(ama_real_win, ama_pred_win_list)
print('진짜 승률', ama_real_win_rate)
print('예측 승률', ama_pred_win_rate)

#%% 결론 
''' 득점한 골수는 정확히 예측하진 못하나 승률을 정확히 예측한다는 것을 알수 있다 
또한, 승률이 50%인 것이 적은 프로선수나 아마추어의test 데이터 수라고 판단되어 더 많은 데이터를 수집하여 예측해봤지만 
계속 승률이 50%였다. 이에 내린 결론은 50%인 이유는 아무래도 프로선수나 랭커나 동일한 플레이어 이기 떄문에 
넥슨에서의 MMR에 의해서 실력이 비슷한 사람끼리 경기를 진행하기 떄문에 승률이 50%인것으로 사료된다.
이로써 프로선수와 전시즌 랭킹1위가 대결 할때의 경기력은 구축한 모델로 알 수 없다. '''