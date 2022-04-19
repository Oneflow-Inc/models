import org.apache.spark.sql.functions.udf

val categorical_names = (1 to 26).map{id=>s"C$id"}
val dense_names = (1 to 13).map{id=>s"I$id"}
val integer_names = Seq("label") ++ dense_names
val col_names = integer_names ++ categorical_names

val src_dir = "/path/to/unziped/criteo1t"
val dst_dir = "/path/to/output"
val tmp_dir = "/path/to/tmp_spark"

// split day_23() to test.csv and val.csv
// # head -n 89137319 src/day_23 > dst/test.csv
// # tail -n +89137320 src/day_23 > dst/val.csv
// total 178274637, test 89137319, val 89137318

val day_23 = s"${src_dir}/day_23"
val test_csv = s"${tmp_dir}/test.csv"
val val_csv = s"${tmp_dir}/val.csv"

val make_label = udf((str:String) => str.toFloat)
val make_dense = udf((str:String) => if (str == null) 1 else str.toFloat + 1)
val make_sparse = udf((str:String, i:Long) => (if (str == null) (i+1) * 40000000L else Math.floorMod(Integer.parseUnsignedInt(str, 16).toLong, 40000000L)) +  i * 40000000L)
val label_cols = Seq(make_label($"label").as("label"))
val dense_cols = 1.to(13).map{i=>make_dense(col(s"I$i")).as(s"I${i}")}
val sparse_cols = 1.to(26).map{i=>make_sparse(col(s"C$i"), lit(i-1)).as(s"C${i}")}
val cols = label_cols ++ dense_cols ++ sparse_cols

spark.read.option("delimiter", "\t").csv(test_csv).toDF(col_names: _*).select(cols:_*).repartition(256).write.parquet(s"${dst_dir}/test")
spark.read.option("delimiter", "\t").csv(val_csv).toDF(col_names: _*).select(cols:_*).repartition(256).write.parquet(s"${dst_dir}/val")

val day_files = 0.until(23).map{day=>s"${src_dir}/day_${day}"}
// in one line
spark.read.option("delimiter", "\t").csv(day_files:_*).toDF(col_names: _*).select(cols:_*).orderBy(rand()).repartition(2560).write.parquet(s"${dst_dir}/train")

// or save temporary data to tmp_dir
// spark.read.option("delimiter", "\t").csv(day_files:_*).toDF(col_names: _*).select(cols:_*).write.parquet(s"${tmp_dir}/tmp1")
// spark.read.parquet(s"${tmp_dir}/tmp1").orderBy(rand()).write.parquet(s"${tmp_dir}/tmp2")
// spark.read.parquet(s"${tmp_dir}/tmp2").repartition(2560).write.parquet(s"${dst_dir}/train")

// print table size array
val df = spark.read.parquet(s"${dst_dir}/train", s"${dst_dir}/val", s"${dst_dir}/test")
println(1.to(26).map{i=>df.select(s"C$i").as[Long].distinct.count}.mkString(","))
