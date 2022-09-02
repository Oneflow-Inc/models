if [ ! -f data/SENTI/train.embeddings.tsv ];then
python extract_embeddings.py \
--bert_path "bert-base-uncased" \
--input data/SENTI/train.tsv \
--output data/SENTI/train.embeddings.tsv \
--task_name senti
fi

if [ ! -f data/SENTI/train_with_weights.tsv ];then
python generate_meta_weights.py \
data/SENTI/train.embeddings.tsv \
data/SENTI/train_with_weights.tsv \
books,dvd,electronics,kitchen
fi

if [ ! -f data/SENTI/dev_standard.tsv ];then
python generate_dev_file.py \
--input data/SENTI/dev.tsv \
--output data/SENTI/dev_standard.tsv
fi