import org.apache.spark.sql.functions.udf
import org.apache.spark.storage.StorageLevel

def splitCriteoKaggle(srcDir: String, dstDir:String) = {
   val categorical_names = (1 to 26).map{id=>s"C$id"}
   val dense_names = (1 to 13).map{id=>s"I$id"}
   val integer_names = Seq("label") ++ dense_names
   val col_names = integer_names ++ categorical_names

   val inputDF = spark.read.option("delimiter", ",").csv(s"${srcDir}/train.txt").toDF(col_names: _*)

   val df = inputDF.persist(StorageLevel.MEMORY_AND_DISK)

   val splits = df.randomSplit(Array(0.8, 0.1, 0.1), seed=2020)
   val train_samples = splits(0).count()
   println(s"train samples = $train_samples")
   val valid_samples = splits(1).count()
   println(s"valid samples = $valid_samples")
   val test_samples = splits(2).count()
   println(s"test samples = $test_samples")

   splits(0).write.option("header", "true").csv(s"${dstDir}/train.csv")
   splits(1).write.option("header", "true").csv(s"${dstDir}/valid.csv")
   splits(2).write.option("header", "true").csv(s"${dstDir}/test.csv")
}
