import java.io.File
import scala.util.Random
import org.apache.spark.sql.functions.udf
import java.io.PrintWriter


def getParquetFiles(root_dir: String): List[String] = {
    def getRecursiveListOfFiles(dir: File): Array[File] = {
        val these = dir.listFiles
        these ++ these.filter(_.isDirectory).flatMap(getRecursiveListOfFiles)
    }
    val file = new File(root_dir)
    val files = getRecursiveListOfFiles(file)
    files.filter(f => """.*\.parquet$""".r.findFirstIn(f.getName).isDefined)
        .map(_.getPath).toList
}


def makeDlrmDataset(srcDir: String, dstDir:String, tmpDir:String, modIdx:Int = 40000000, num_day_parts: Int = 256) = {
    val categorical_names = (1 to 26).map{id=>s"C$id"}
    val dense_names = (1 to 13).map{id=>s"I$id"}
    val integer_names = Seq("label") ++ dense_names
    val col_names = integer_names ++ categorical_names

    val day_23 = s"${srcDir}/day_23"
    val test_csv = s"${tmpDir}/test.csv"
    val val_csv = s"${tmpDir}/val.csv"

    val day_23_lines = scala.io.Source.fromFile(day_23).getLines

    new PrintWriter(test_csv) {
      day_23_lines.slice(0, 89137319).foreach{println}
      close
    }

    new PrintWriter(val_csv) {
      day_23_lines.foreach(println)
      close
    }

    val make_label = udf((str:String) => str.toFloat)
    val label_cols = Seq(make_label($"label").as("label"))

    val make_dense = udf((str:String) => if (str == null) 1 else str.toFloat + 1)
    val dense_cols = 1.to(13).map{i=>make_dense(col(s"I$i")).as(s"I${i}")}

    var sparse_cols = if (modIdx > 0){
        def make_sparse = udf((str:String, i:Int, mod:Int) => (if (str == null) mod else Math.floorMod(Integer.parseUnsignedInt(str, 16).toInt, mod)) +  i * (mod + 1))
        1.to(26).map{i=>make_sparse(col(s"C$i"), lit(i-1), lit(modIdx)).as(s"C${i}")}
    } else {
        1.to(26).map{i=>xxhash64(lit(i), col(s"C$i")).as(s"C${i}")}
    }

    val cols = label_cols ++ dense_cols ++ sparse_cols

    spark.read.option("delimiter", "\t").csv(test_csv).toDF(col_names: _*).select(cols:_*).repartition(256).write.parquet(s"${dstDir}/test")
    spark.read.option("delimiter", "\t").csv(val_csv).toDF(col_names: _*).select(cols:_*).repartition(256).write.parquet(s"${dstDir}/val")

    0.to(22).map{day=>spark.read.option("delimiter", "\t").csv(s"${srcDir}/day_${day}").toDF(col_names: _*).select(cols:_*).orderBy(rand()).repartition(num_day_parts).write.mode("overwrite").parquet(s"${tmpDir}/shuffled_days/day_${day}")}
    
    val files = Random.shuffle(getParquetFiles(s"${tmpDir}/shuffled_days"))
    val grouped_files = files.grouped(files.length / 23).toList
    grouped_files.zipWithIndex.foreach{case(files, id) => spark.read.parquet(files:_*).orderBy(rand()).repartition(256).write.parquet(s"${dstDir}/shuffled_day_parts/day_part_${id}")}
} 
