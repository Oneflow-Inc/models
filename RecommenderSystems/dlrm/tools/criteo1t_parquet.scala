import org.apache.spark.sql.functions.udf

def makeDlrmDataset(srcDir: String, dstDir:String, tmpDir:String, modIdx:Long = 40000000L, stepByStep:Boolean = false) = {
    val categorical_names = (1 to 26).map{id=>s"C$id"}
    val dense_names = (1 to 13).map{id=>s"I$id"}
    val integer_names = Seq("label") ++ dense_names
    val col_names = integer_names ++ categorical_names

    val day_23 = s"${srcDir}/day_23"
    val test_csv = s"${tmpDir}/test.csv"
    val val_csv = s"${tmpDir}/val.csv"

    val make_label = udf((str:String) => str.toFloat)
    val label_cols = Seq(make_label($"label").as("label"))

    val make_dense = udf((str:String) => if (str == null) 1 else str.toFloat + 1)
    val dense_cols = 1.to(13).map{i=>make_dense(col(s"I$i")).as(s"I${i}")}

    var sparse_cols = if (modIdx > 0){
        def make_sparse = udf((str:String, i:Long, mod:Long) => (if (str == null) mod else Math.floorMod(Integer.parseUnsignedInt(str, 16).toLong, mod)) +  i * (mod + 1))
        1.to(26).map{i=>make_sparse(col(s"C$i"), lit(i-1), lit(modIdx)).as(s"C${i}")}
    } else {
        1.to(26).map{i=>xxhash64(lit(i), col(s"C$i")).as(s"C${i}")}
    }

    val cols = label_cols ++ dense_cols ++ sparse_cols

    spark.read.option("delimiter", "\t").csv(test_csv).toDF(col_names: _*).select(cols:_*).repartition(256).write.parquet(s"${dstDir}/test")
    spark.read.option("delimiter", "\t").csv(val_csv).toDF(col_names: _*).select(cols:_*).repartition(256).write.parquet(s"${dstDir}/val")

    val day_files = 0.until(23).map{day=>s"${srcDir}/day_${day}"}
    if (stepByStep) {
        // save temporary intermediate data to tmpDir
        spark.read.option("delimiter", "\t").csv(day_files:_*).toDF(col_names: _*).select(cols:_*).write.parquet(s"${tmpDir}/tmp1")
        spark.read.parquet(s"${tmpDir}/tmp1").orderBy(rand()).write.parquet(s"${tmpDir}/tmp2")
        spark.read.parquet(s"${tmpDir}/tmp2").repartition(2560).write.parquet(s"${dstDir}/train")
    } else {
        spark.read.option("delimiter", "\t").csv(day_files:_*).toDF(col_names: _*).select(cols:_*).orderBy(rand()).repartition(2560).write.parquet(s"${dstDir}/train")
    }

    // print table size array
    val df = spark.read.parquet(s"${dstDir}/train", s"${dstDir}/val", s"${dstDir}/test")
    println(1.to(26).map{i=>df.select(s"C$i").as[Long].distinct.count}.mkString(","))
} 
