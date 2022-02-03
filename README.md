# cassino
#在此資料模型中，欄位['有效客戶']為預測之目標，為判斷此客戶是會繼續消費。將以決策樹、隨機森林、KNN、SVC支持向量分類、羅吉斯回歸等多種演算法找出準確度最高之分類演算法，做為日後機器學習分辨之判斷模型:
#資料於data.csv中:
![image](https://user-images.githubusercontent.com/71545529/152302012-7531e86e-5879-418b-9af2-582c6fd17fac.png)

#經過資料清洗以及建模後，隨機森林為準確度最高之分類演算法(準確度: 95.75%)。
#個別演算法程式如下:(決策樹)decision_tree.py、(隨機森林)Randomforest.py、(KNN演算法)KNN.py、(羅吉斯回歸)logistic_reggression.py、(SVC支持向量分類)SVC.py
#另外以power BI進行財務及資源配置之分析，以及利用相關係數矩陣(corr)對客戶消費行為進行分析:

#Cassino、快開、棋盤、運動、電子，此為5種遊戲類別，客戶有A、B、C、D，4個平台作為遊戲場地。
![image](https://user-images.githubusercontent.com/71545529/152303838-ebb54b67-2945-4ce2-9fc6-9d4f50615ea6.png)
![image](https://user-images.githubusercontent.com/71545529/152304253-3d0372ff-d61c-493f-8763-67a936e35d1d.png)
![image](https://user-images.githubusercontent.com/71545529/152304350-2adb0468-d52a-4891-811f-41454bee96e0.png)
![image](https://user-images.githubusercontent.com/71545529/152304390-036aae1d-d4da-46d1-b072-1868f8c6a3b2.png)
![image](https://user-images.githubusercontent.com/71545529/152304419-f05e9324-31c6-454f-b80d-84ddbef44b8c.png)

#	相關係數圖(corr):

![image](https://user-images.githubusercontent.com/71545529/152304726-5c892afc-9cc2-41a0-a136-629f822cd4b7.png)
![image](https://user-images.githubusercontent.com/71545529/152304780-12d2fbb6-678a-467c-a8fb-914f79c45b23.png)


![image](https://user-images.githubusercontent.com/71545529/152304879-17dd8c5f-44be-4496-9266-791c979580f3.png)

