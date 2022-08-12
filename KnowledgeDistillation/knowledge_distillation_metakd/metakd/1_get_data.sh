cd data
if [ ! -f ./SENTI/dev.tsv ];then
wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/datasets/domain_sentiment_data.tar.gz
tar -zxvf domain_sentiment_data.tar.gz
fi
cd ..

if [ ! -f data/SENTI/dev.tsv ];then
python generate_senti_data.py
fi