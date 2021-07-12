from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np





#讀取資料
df = pd.read_csv('data.csv', encoding = 'big5', thousands=',')
df = df.dropna(subset=['首充日期'], axis=0)

#數值之空值以0填補
df['當日充值點數'] = df['當日充值點數'].fillna(0.00)
df['運動流水'] = df['運動流水'].fillna(0.00)
df['運動結果'] = df['運動結果'].fillna(0.00)
df['Casino流水'] = df['Casino流水'].fillna(0.00)
df['Casino結果'] = df['Casino結果'].fillna(0.00)
df['快開流水'] = df['快開流水'].fillna(0.00)
df['快開結果'] = df['快開結果'].fillna(0.00)
df['電子流水'] = df['電子流水'].fillna(0.00)
df['電子結果'] = df['電子結果'].fillna(0.00)
df['棋牌流水'] = df['棋牌流水'].fillna(0.00)
df['棋牌結果'] = df['棋牌結果'].fillna(0.00)



#數值欄位轉為float
df['當日充值點數'] = pd.to_numeric(df['當日充值點數'], errors='coerce')
df['運動流水'] = pd.to_numeric(df['運動流水'], errors='coerce')
df['運動結果'] = pd.to_numeric(df['運動結果'], errors='coerce')
df['Casino流水'] = pd.to_numeric(df['Casino流水'], errors='coerce')
df['Casino結果'] = pd.to_numeric(df['Casino結果'], errors='coerce')
df['快開流水'] = pd.to_numeric(df['快開流水'], errors='coerce')
df['快開結果'] = pd.to_numeric(df['快開結果'], errors='coerce')
df['電子流水'] = pd.to_numeric(df['電子流水'], errors='coerce')
df['電子結果'] = pd.to_numeric(df['電子結果'], errors='coerce')
df['棋牌流水'] = pd.to_numeric(df['棋牌流水'], errors='coerce')
df['棋牌結果'] = pd.to_numeric(df['棋牌結果'], errors='coerce')
df['首充點數'] = pd.to_numeric(df['首充點數'], errors='coerce')



#日期欄位轉為datetime
df['註冊日期'] = pd.to_datetime(df['註冊日期'], format='%Y-%m-%d', errors='coerce')
df['首充日期'] = pd.to_datetime(df['首充日期'], format='%Y-%m-%d', errors='coerce')
df["日期"] = df["日期"].str.replace("月", "/")
df["日期"] = df["日期"].str.replace("日", "")
df["日期"] = '2021/' + df["日期"]
df['日期'] = pd.to_datetime(df['日期'], format='%Y-%m-%d', errors='coerce')




#增加天數欄位
df['天數'] = (df['日期']-df['首充日期']).map(lambda x: x.days)


#刪除VIP等級欄位
df = df.drop(['VIP等級'], axis=1)


#處理業務負責人員欄位
df["負責人員"] = df["負責人員"].str.replace("公司-離職", "無")
df["負責人員"] = df["負責人員"].str.replace("公司-離職", "無")

#處裡有效客戶欄位
df["有效客戶"] = df["有效客戶"].str.replace("V", '1')
df["有效客戶"] = df["有效客戶"].str.replace("~", '0')
df["有效客戶"] = df["有效客戶"].str.replace("-", '0')
df["有效客戶"] = df["有效客戶"].str.replace("0", '0')
df["有效客戶"] = df["有效客戶"].fillna('0')

#刪除日期和會員號
df = df.drop(['註冊日期'], axis=1)
df = df.drop(['日期'], axis=1)
df = df.drop(['首充日期'], axis=1)
df = df.drop(['會員號'], axis=1)


#把負責人員欄位以及平台欄位變成dummy，並且合併後砍掉原本欄位
df1 = pd.get_dummies(df['負責人員'])
df = pd.concat([df, df1], axis=1)
df = df.drop(['負責人員'], axis=1)

df2 = pd.get_dummies(df['平台'])
df = pd.concat([df, df2], axis=1)
df = df.drop(['平台'], axis=1)

print(df.info())
print(df)

#x為訓練用特徵

x = df.drop(['有效客戶'], axis=1)

#y為預測目標

y = df['有效客戶']

#將資料分為70%訓練集，30%測試集，參數:test_size=0.3
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)


#建立模型並加以預測
forest = RandomForestClassifier()
forest.fit(x_train, y_train)
predictions = forest.predict(x_test)


#顯示訓練集測驗結果
print(forest.score(x_train, y_train))
#準確度: 1.0


#顯示測驗集測驗結果(重要)
print(forest.score(x_test, y_test))
#準確度: 0.9575835475578406
