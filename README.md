# weather-describe
describes weather conditions based on input images

Setup Instructions:
Install Anaconda.  If Anaconda is already installed, skip this step
    download the Anaconda installer from https://www.continuum.io/downloads#linux
    Follow the installation instructions included here: https://docs.continuum.io/anaconda/install/linux
Install Tensorflow.  Enter the following commands
    conda create -n tensorflow
    source activate tensorflow #your prompt will change
    pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.2.1-cp36-cp36m-linux_x86_64.whl

To Run Locally:
From weather_describe/ run:

spark-submit --master=local[1] --num-executors 5 --driver-memory 8g --executor-memory 8g --packages databricks:spark-deep-learning:0.1.0-spark2.1-s_2.11 weather_describe.py yvr-weather katkam-scaled output