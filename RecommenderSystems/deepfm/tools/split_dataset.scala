import org.apache.spark.sql.functions.udf

def splitDataset(srcDir: String, dstDir:String) = {
   val categorical_names = (1 to 26).map{id=>s"C$id"}
   val dense_names = (1 to 13).map{id=>s"I$id"}
   val integer_names = Seq("label") ++ dense_names
   val col_names = integer_names ++ categorical_names

   val df = spark.read.option("delimiter", "\t").csv(s"${srcDir}/train.txt").toDF(col_names: _*)

   val splits = df.randomSplit(Array(0.8, 0.1, 0.1), seed=2018)

   splits(0).write.option("header", "true").csv(s"${dstDir}/train.csv")
   splits(1).write.option("header", "true").csv(s"${dstDir}/valid.csv")
   splits(2).write.option("header", "true").csv(s"${dstDir}/test.csv")
}
