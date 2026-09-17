$env:PATH = "C:\Users\LAITH AHMAD\AppData\Local\Programs\Python\Python312;C:\Users\LAITH AHMAD\AppData\Local\Programs\Python\Python312\Scripts;C:\hadoop\bin;" + $env:PATH
$env:SPARK_HOME = "C:\Users\LAITH AHMAD\AppData\Local\Programs\Python\Python312\Lib\site-packages\pyspark"
$env:PYSPARK_PYTHON = "C:\Users\LAITH AHMAD\AppData\Local\Programs\Python\Python312\python.exe"
$env:PYSPARK_DRIVER_PYTHON = "C:\Users\LAITH AHMAD\AppData\Local\Programs\Python\Python312\python.exe"
$env:HADOOP_HOME = "C:\hadoop"
cd "C:\Users\LAITH AHMAD\Desktop\BD\hh-ru-skill-prediction\src"
python train_nonspark_models.py --data "../data/vacancies_sample.parquet" --output "../results/nonspark_results.csv" --models-dir "../models/nonspark" --tuned-params "../configs/tuned_params.json"
