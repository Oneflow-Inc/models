import org.apache.spark.sql.functions.udf

def makeDCNDataset(srcDir: String, dstDir:String) = {
    val train_csv = s"${srcDir}/train.csv"
    val test_csv = s"${srcDir}/test.csv"
    val val_csv = s"${srcDir}/valid.csv"

    val make_label = udf((str:String) => str.toFloat)
    val label_cols = Seq(make_label($"Label").as("Label"))

    val dense_cols = 1.to(13).map{i=>xxhash64(lit(i), col(s"I$i")).as(s"I${i}")}

    var sparse_cols = 1.to(26).map{i=>xxhash64(lit(i), col(s"C$i")).as(s"C${i}")}

    val cols = label_cols ++ dense_cols ++ sparse_cols

    spark.read.option("header","true").csv(test_csv).select(cols:_*).repartition(32).write.parquet(s"${dstDir}/test")
    spark.read.option("header","true").csv(val_csv).select(cols:_*).repartition(32).write.parquet(s"${dstDir}/val")

    spark.read.option("header","true").csv(train_csv).select(cols:_*).orderBy(rand()).repartition(256).write.parquet(s"${dstDir}/train")

    // print the number of samples
    val train_samples = spark.read.parquet(s"${dstDir}/train").count()
    println(s"train samples = $train_samples")
    val val_samples = spark.read.parquet(s"${dstDir}/val").count()
    println(s"validation samples = $val_samples")
    val test_samples = spark.read.parquet(s"${dstDir}/test").count()
    println(s"test samples = $test_samples")

    // print table size array
    val df = spark.read.parquet(s"${dstDir}/train", s"${dstDir}/val", s"${dstDir}/test")
    println("table size array: ")
    println(1.to(13).map{i=>df.select(s"I$i").as[Long].distinct.count}.mkString(","))
    println(1.to(26).map{i=>df.select(s"C$i").as[Long].distinct.count}.mkString(","))
}


