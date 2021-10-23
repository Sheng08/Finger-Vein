主程式:
python SUMLA_ROI_main.py -f ./SUMLA_raw_data/ -s ./SUMLA_result/ -e ./SUMLA_expc/ -r ./SUMLA_draw/

需要clahe:註解 496~498行 將506~508行取消註解
-------------------------------------------------------------------------------------
...小波處理後...
-------------------------------------------------------------------------------------
python file_injection_resize.py -s ./SUMLA_result_dwt/ -r ./CNN_data/train_data/ -t ./CNN_data/test_data/

