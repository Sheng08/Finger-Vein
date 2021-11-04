# Finger-Vein

### 作品介紹
本作品由本人獨力完成，利用OpenCV針對FVUSM、SDUMLA指靜脈的開放資料集進行影像處理並擷取ROI區域，其目的是為了獲取含有指靜脈區域圖片，排除手指邊緣與雜訊等，其處理後的資料圖像可做為日後訓練CNN模型分類使用。

###  作品之實驗內容如下：
>1.	利用OpenCV對指靜脈圖片進行：高斯雙邊濾波→Sobel邊緣檢測→圖像線性融合與疊加→Canny多級邊緣檢測→形態學轉換(開運算)，去除雜訊與提取手指邊緣
>2.	利用邊緣設計感興趣區域(ROI)定位演算法，確保只擷取含指靜脈區域之圖像
>3.	對指靜脈之圖像進行Haar小波轉換
>4.	利用AlexNet CNN模型進行指靜脈分類訓練

###  成果截圖：
影像處理各過程結果<br>
![1](https://user-images.githubusercontent.com/58781800/140339004-c328761d-dd71-4c25-a788-f57463903179.png)

感興趣區域(ROI)定位矯正+區域旋轉矯正<br>
![2](https://user-images.githubusercontent.com/58781800/140339007-a865e8b3-fecc-4ef5-b55c-c6884421b2fb.png)

ROI定位比較<br>
![3](https://user-images.githubusercontent.com/58781800/140339011-36953e8e-712f-4c91-b191-24491bea172d.png)

Haar小波轉換加強影像特徵<br>
![4](https://user-images.githubusercontent.com/58781800/140339016-de61c2bc-9798-4f90-b3dc-648a5549c6eb.png)

Alexnet訓練結果準確度達約97%<br>
![螢幕擷取畫面 2021-11-04 230518](https://user-images.githubusercontent.com/58781800/140339020-fc039fa5-4f14-4752-91cd-7d67f845f59e.jpg)
