# DLRM
[DLRM](https://arxiv.org/pdf/1906.00091.pdf) is a deep learning-based recommendation model that exploits categorical data model for CTR recommendation. Its model structure is as follows. Based on this structure, this project uses OneFlow distributed deep learning framework to realize training the modle in graph mode and eager mode respectively on Crioteo data set. 
![image](https://user-images.githubusercontent.com/63446546/158937131-1a057659-0d49-4bfb-aee2-5568e605fa01.png)
## Directory description
```
#待补充
```
## Arguments description
待补充

## Prepare running
### Environment
Running DLRM model requires downloading [OneFlow](https://github.com/Oneflow-Inc/oneflow), [scikit-learn](https://scikit-learn.org/stable/install.html) for caculating mertics, and tool package [numpy](https://numpy.org/)。


### Dataset
[Criteo](https://figshare.com/articles/dataset/Kaggle_Display_Advertising_Challenge_dataset/5732310) dataset is the online advertising dataset published by criteo labs. It contains the function value and click feedback of millions of display advertisements, which can be used as the benchmark for CTR prediction. Each advertisement has the function of describing data. The dataset has 40 attributes. The first attribute is the label, where a value of 1 indicates that the advertisement has been clicked and a value of 0 indicates that the advertisement has not been clicked. This attribute contains 13 integer columns and 26 category columns.

### Prepare ofrecord format data 
Please view [how_to_make_ofrecord_for_wdl](https://github.com/Oneflow-Inc/OneFlow-Benchmark/blob/master/ClickThroughRate/WideDeepLearning/how_to_make_ofrecord_for_wdl.md)

## Start training by Oneflow

### Train by using ddp
```
bash train_ddp.sh
```
### Train by graph mode in global view 
```
bash train_global_graph.sh
```
### Train by eager mode in global view 
```
bash train_global_eager.sh
```
## Dataset preparation

```
本模块待替换
```
Currently OneFlow-WDL supports two types of dataset format: ofrecord and onerec, both can be tranformed from HugeCTR parquet format dataset.
Following two steps to process dataset:

Step 1: [Preprocess the Dataset Through NVTabular](https://github.com/NVIDIA-Merlin/HugeCTR/tree/master/samples/wdl#preprocess-the-dataset-through-nvtabular)

Run NVTabular docker and execute the following preprocessing command to parquet format dataset
```bash
$ bash preprocess.sh 1 criteo_data nvt 1 0 1 # parquet output
```
Step 2: Convert parquet format dataset to ofrecord or(and) onerec format dataset

This step needs to be executed in the spark 2.4.x environment. Please download the dependent jar package from [here](http://oneflow-public.oss-cn-beijing.aliyuncs.com/tools/spark-oneflow-connector-assembly-0.2.0-SNAPSHOT.jar) first, then launch spark shell environment with this jar support.

In spark shell environment, execute following scripts. You may modify `db_path`(parquet dataset path as input), `output_path`(ofrecord or onerec dataset path for output).
```scala
import org.oneflow.spark.functions._
import org.apache.spark.sql.types.{IntegerType}

def convert_parquet(input_path: String, output_path: String, db_type: String) = {
  val NUM_CATEGORICAL_COLUMNS = 26
  val categorical_cols = (1 to NUM_CATEGORICAL_COLUMNS).map{id=>s"C$id"}
  val sparse_cols = Seq("C1_C2", "C3_C4") ++ categorical_cols
  val slot_size_array = Array(225945, 354813, 202260, 18767, 14108, 6886, 18578, 4, 6348, 1247, 51, 186454, 71251, 66813, 11, 2155, 7419, 60, 4, 922, 15, 202365, 143093, 198446, 61069, 9069, 74, 34)

  var df = spark.read.parquet(input_path)
  df = df.withColumn("label", col("label").cast(IntegerType)).withColumnRenamed("label", "labels")
  var offset: Long = 0 
  var i = 0
  for(col_name <- sparse_cols) {
    println(col_name)
    df = df.withColumn(col_name, df(col_name) + lit(offset))
           .withColumn(col_name, col(col_name).cast(IntegerType))
    offset += slot_size_array(i)
    i += 1
  }
  
  val NUM_INTEGER_COLUMNS = 13
  val integer_cols = (1 to NUM_INTEGER_COLUMNS).map{id=>s"I$id"} 
  df = df.select($"labels", array(integer_cols map col: _*).as("dense_fields"), array(categorical_cols map col: _*).as("deep_sparse_fields"), array("C1_C2", "C3_C4").as("wide_sparse_fields"))

  df = df.repartition(256)
  
  var path = output_path + "wdl_ofrecord/" + db_type
  println(path)
  df.write.mode("overwrite").ofrecord(path)
  sc.formatFilenameAsOneflowStyle(path)
  
  path = output_path + "wdl_onerec/" + db_type
  println(path)
  df.write.mode("overwrite").onerec(path)
}

val output_path = "/dataset/8c609a13/day_0/"
val db_path = "/dataset/d4f7e679/criteo_day_0_parquet/train/part_*.parquet"
convert_parquet(db_path, output_path, "train")

val db_path = "/dataset/d4f7e679/criteo_day_0_parquet/val/part_*.parquet"
convert_parquet(db_path, output_path, "val")
```

Note: slot_size_array is generated in step 1, please find more description [here](https://github.com/NVIDIA-Merlin/HugeCTR/blob/master/docs/python_interface.md#parquet).
- We provide an option to add offset for each slot by specifying slot_size_array. slot_size_array is an array whose length is equal to the number of slots. To avoid duplicate keys after adding offset, we need to ensure that the key range of the i-th slot is between 0 and slot_size_array[i]. We will do the offset in this way: for i-th slot key, we add it with offset slot_size_array[0] + slot_size_array[1] + ... + slot_size_array[i - 1]. In the configuration snippet noted above, for the 0th slot, offset 0 will be added. For the 1st slot, offset 278899 will be added. And for the third slot, offset 634776 will be added.





