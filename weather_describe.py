import sys
from pyspark.sql import SparkSession, functions, types, Row
from sparkdl import readImages
import re
#from difflib import get_close_matches

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

#weather_list = ['Clear', 'Rain', 'Cloudy']

def fix_weather(observed_weather):
    '''
    Apply this to the column of weathers to get a more limited vocabulary of observations
    We need to decide what kind of things we are looking for, but this is good for now.
    The fog line should be at the end so it only ever occurs if no other weather is observed.
    (getclosematches wasnt working so i did this the ugly way)
    '''
    if 'Pellet' in observed_weather:
        return 'Ice Pellets'
    elif 'Thunder' in observed_weather:
        return 'Stormy'
    elif 'Drizzle' in observed_weather:
        return 'Drizzle'
    elif 'Fog' in observed_weather:
        return 'Fog'
    elif 'Snow' in observed_weather:
        return 'Snow'
    elif 'Shower' in observed_weather:
        return 'Shower'
    elif 'Cloud' in observed_weather:
        return 'Cloudy'
    elif 'Clear' in observed_weather:
        return 'Clear'
    elif 'Rain' in observed_weather:
        return 'Rain'

def get_fog(observed_weather):
    '''
    Apply this to weather column to determine if the day was foggy or not
    This should probably be in a secondary column to make scoring a bit easier
    '''
    if 'Fog' in observed_weather:
        return True
    else:
        return False

udf_fix_weather = functions.udf(fix_weather, types.StringType())
udf_get_fog = functions.udf(get_fog, types.StringType())

def main():
    data_directory = sys.argv[1]
    pic_directory = sys.argv[2]
    output_directory = sys.argv[3]

    weather_data = spark.read.csv(data_directory, schema=schema)
    weather_data_no_NA = weather_data.filter(weather_data['Weather'] != 'NA')
    weather_data_selected = weather_data_no_NA.select(#include more columns as we determine what we need
        weather_data_no_NA['Date/Time'],
        weather_data_no_NA['Weather']
    ).cache()
    weather_less_vocab = weather_data_selected.withColumn('Reduced Weather', udf_fix_weather(weather_data_selected['Weather']))
    weather_w_fog = weather_less_vocab.withColumn('Fog', udf_get_fog(weather_data_selected['Weather']))

    #weather_w_fog.write.csv(output_directory, mode='overwrite')

    pics_data = readImages(pic_directory)
    pics_data.show()

if __name__ == '__main__':
    main()
