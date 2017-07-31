import sys
from pyspark.sql import SparkSession, functions, types, Row
import re
from difflib import get_close_matches

spark = SparkSession.builder.appName('weather describe').getOrCreate()

assert sys.version_info >= (3, 4) # make sure we have Python 3.4+
assert spark.version >= '2.1' # make sure we have Spark 2.1+

schema = types.StructType([
    types.StructField('Date/Time', types.StringType(), False),
    types.StructField('Year', types.StringType(), False),
    types.StructField('Month', types.StringType(), False),
    types.StructField('Day', types.StringType(), False),
    types.StructField('Time', types.StringType(), False),
    types.StructField('Data Quality', types.StringType(), False),
    types.StructField('Temp (°C)', types.StringType(), False),
    types.StructField('Temp Flag', types.StringType(), False),
    types.StructField('Dew Point Temp (°C)', types.StringType(), False),
    types.StructField('Dew Point Temp Flag', types.StringType(), False),
    types.StructField('Rel Hum (%)', types.StringType(), False),
    types.StructField('Rel Hum Flag', types.StringType(), False),
    types.StructField('Wind Dir (10s deg)', types.StringType(), False),
    types.StructField('Wind Dir Flag', types.StringType(), False),
    types.StructField('Wind Spd (km/h)', types.StringType(), False),
    types.StructField('Wind Spd Flag', types.StringType(), False),
    types.StructField('Visibility (km)', types.StringType(), False),
    types.StructField('Visibility Flag', types.StringType(), False),
    types.StructField('Stn Press (kPa)', types.StringType(), False),
    types.StructField('Stn Press Flag', types.StringType(), False),
    types.StructField('Hmdx', types.StringType(), False),
    types.StructField('Hmdx Flag', types.StringType(), False),
    types.StructField('Wind Chill', types.StringType(), False),
    types.StructField('Wind Chill Flag', types.StringType(), False),
    types.StructField('Weather', types.StringType(), False),
])

weather_list = ['Clear', 'Rain', 'Cloudy']

def fix_weather(weather_list, observed_weather):
    '''
    Apply this to the column of weathers to get a more limited vocabulary of observations
    We need to decide what kind of things we are looking for, but this is good for now.
    '''
    return get_close_matches(weather_list, observed_weather)

udf_fix_weather = functions.udf(fix_weather, types.StringType())

def main():
    data_directory = sys.argv[1]
    pic_directory = sys.argv[2]
    output_directory = sys.argv[3]

    weather_data = spark.read.csv(data_directory, schema=schema)
    weather_data_no_NA = weather_data.filter(weather_data['Weather'] != 'NA')
    weather_data_selected = weather_data_no_NA.select(
        weather_data_no_NA['Date/Time'],
        weather_data_no_NA['Weather']
    )
    weather_data_selected.show()
    #The following is in progress. currently doesn't work as intended. returns empty list.
    #weather_array = weather_data_selected.select('Weather').rdd.flatMap(lambda x: x).collect()
    #weather_array = fix_weather(weather_list, weather_array)
    #print(weather_array)

if __name__ == '__main__':
    main()
