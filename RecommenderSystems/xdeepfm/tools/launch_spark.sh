export SPARK_LOCAL_DIRS=/tmp/tmp_spark
spark-shell \
    --master "local[*]" \
    --conf spark.driver.maxResultSize=0 \
    --driver-memory 360G
