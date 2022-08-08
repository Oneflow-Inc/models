import org.apache.spark.sql.functions.udf

def makeCriteo1tC39Int32(srcDir: String, dstDir:String) = {
    val categorical_names = (1 to 26).map{id=>s"C$id"}
    val dense_names = (1 to 13).map{id=>s"I$id"}
    val integer_names = Seq("label") ++ dense_names
    val col_names = integer_names ++ categorical_names
    
    val test_csv = s"${srcDir}/test.csv"
    val val_csv = s"${srcDir}/val.csv"
    
    val make_label = udf((str:String) => str.toFloat)
    val make_dense = udf((str:String, i:Int) => (if (str == null) 1  else str.toFloat + 1) + i * 40000000)
    val make_sparse = udf((str:String, i:Int) => (if (str == null) 0 else Math.floorMod(Integer.parseUnsignedInt(str, 16).toInt, 40000000)) +  i * 40000000)
    val label_cols = Seq(make_label($"label").as("label"))
    val dense_cols = 1.to(13).map{i=>make_dense(col(s"I$i"), lit(i)).as(s"C${i}")}
    val sparse_cols = 1.to(26).map{i=>make_sparse(col(s"C$i"), lit(i + 13)).as(s"C${i+13}")}
    val cols = label_cols ++ dense_cols ++ sparse_cols
    
    spark.read.option("delimiter", "\t").csv(test_csv).toDF(col_names: _*).select(cols:_*).repartition(256).write.parquet(s"${dstDir}/test")
    spark.read.option("delimiter", "\t").csv(val_csv).toDF(col_names: _*).select(cols:_*).repartition(256).write.parquet(s"${dstDir}/val")
    
    val day_files = 0.until(23).map{day=>s"${srcDir}/day_${day}"}
    spark.read.option("delimiter", "\t").csv(day_files:_*).toDF(col_names: _*).select(cols:_*).orderBy(rand()).repartition(2560).write.parquet(s"${dstDir}/train")
    
    // print table size array
    val df = spark.read.parquet(s"${dstDir}/*/*.parquet")
    println(1.to(39).map{i=>df.select(s"C$i").as[Int].distinct.count}.mkString(","))
}

