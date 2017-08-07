# weather-describe
describes weather conditions based on input images

To Run Locally:
From weather_describe/ run:

spark-submit --master=local[1] --num-executors 5 --driver-memory 8g --executor-memory 8g --packages databricks:spark-deep-learning:0.1.0-spark2.1-s_2.11 weather_describe.py yvr-weather katkam-scaled output