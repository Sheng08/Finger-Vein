主程式:
python ROI_main2.py -f ./FVUSM1_raw_data/ -s ./result1/ -e ./expc1/ -r ./draw1/

需要clahe:取消註解 430 431行
-------------------------------------------------------------------------------------
例外圖片程式:
python ROI_main_expc.py -e ./expc1/ -s ./result1/ -r ./draw1/
-------------------------------------------------------------------------------------
...小波處理後...
-------------------------------------------------------------------------------------
python file_injection_resize.py -s ./result1_dwt/ -r ./CNN_data/train_data/ -t ./CNN_data/test_data/


!!需要清除 _DS_Store 檔案