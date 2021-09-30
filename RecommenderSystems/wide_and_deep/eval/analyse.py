import pandas as pd
import os
import math

COLUMNS_BASEINFO=['batch_size','deep_vocab_size','deep_embedding_vec_size','hidden_units_num']
COLUMNS_ANA=['latency/ms','memory_usage/MB']
currentPath=os.path.dirname(os.path.abspath(__file__))

def analyse_test(csvFiles=[],ana_fileName=''):
    data=[]
    for file in csvFiles:
        df=pd.read_csv(file)
        tmp_baseinfo_data=df.loc[[0],COLUMNS_BASEINFO].values.flatten().tolist()
        tmp_ana_data=df[COLUMNS_ANA].mean().values.flatten().tolist()
        tmp_ana_data[0]=round(tmp_ana_data[0],3)
        tmp_ana_data[1]=int(tmp_ana_data[1])
        data.append(tmp_baseinfo_data+tmp_ana_data)
    result_df=pd.DataFrame(data,columns=COLUMNS_BASEINFO+COLUMNS_ANA)
    dir_path=os.path.join(currentPath,'csv/analysis/%s'%(ana_fileName))
    result_df.to_csv(dir_path,index=False)



if __name__ == "__main__":
    batch_size_array=[]
    for i in range(10):
        batch_size_array.append(int(512*math.pow(2,i)))
    csvFiles=[]
    for batch_size in batch_size_array:
        csvFiles.append('/home/shiyunxiao/models/wide_and_deep/eval/csv/n1g1_batch_size_x2_tests_eager/record_%s_0.csv'%batch_size)
    analyse_test(csvFiles,ana_fileName='n1g1_batch_size_x2_test_eager.csv')