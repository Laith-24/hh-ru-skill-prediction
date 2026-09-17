$env:PATH = "C:\Users\LAITH AHMAD\AppData\Local\Programs\Python\Python312;C:\Users\LAITH AHMAD\AppData\Local\Programs\Python\Python312\Scripts;C:\hadoop\bin;" + $env:PATH
$env:SPARK_HOME = "C:\Users\LAITH AHMAD\AppData\Local\Programs\Python\Python312\Lib\site-packages\pyspark"
$env:PYSPARK_PYTHON = "C:\Users\LAITH AHMAD\AppData\Local\Programs\Python\Python312\python.exe"
$env:PYSPARK_DRIVER_PYTHON = "C:\Users\LAITH AHMAD\AppData\Local\Programs\Python\Python312\python.exe"
$env:HADOOP_HOME = "C:\hadoop"
cd "C:\Users\LAITH AHMAD\Desktop\BD\hh-ru-skill-prediction"
spark-submit --master local[2] --driver-memory 4g --conf spark.sql.parquet.columnarReaderBatchSize=512 --conf spark.sql.shuffle.partitions=8 --conf spark.default.parallelism=8 --conf spark.local.dir="C:\Users\LAITH AHMAD\Desktop\BD\hh-ru-skill-prediction\spark-tmp" src/retrain_on_new_data.py --new-data "data/demo/demo_new.parquet" --old-data "data/demo/demo_old.parquet" --spark-models-dir "models/spark_demo" --nonspark-models-dir "models/nonspark_demo" --tuned-params "configs/tuned_params.json" --pipeline-path "models/feature_pipeline_demo"
