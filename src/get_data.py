import ee
import pandas as pd
import os
from gee_utils import export_images, wait_on_tasks
import configparser
import math
import time

config = configparser.ConfigParser()
config.read('config.ini')

ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')

data_dir = './data'
dhs_cluster_file_path = os.path.join(data_dir, 'dhs_clusters.csv')
df = pd.read_csv(dhs_cluster_file_path)
df.head()

surveys = list(df.groupby(['country', 'year']).groups.keys())

def test_export(df, country, year):
    test_df = df[(df['country'] == country) & (df['year'] == year)].sample(10, random_state=0)
    test_tasks = export_images(test_df,
                               country=country,
                               year=year,
                               export_folder='',  # 'data/dhs_tfrecords_raw',
                               export='gcs',
                               bucket='config['GCS']['BUCKET']',
                               ms_bands=['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2', 'TEMP1'],
                               include_nl=True,
                               start_year=1990,
                               end_year=2020,
                               span_length=3,
                               chunk_size=5)
    wait_on_tasks(test_tasks, poll_interval=60)

test_export(df, surveys[0][0], surveys[0][1])

latest_tasks = {}
for survey in surveys:
    latest_tasks[survey] = -1

for survey in surveys:
    files_path = f'gs://{config['GCS']['BUCKET']}/{config['GCS']['EXPORT_FOLDER']}/{survey[0]}_{survey[1]}'
    files_in_bucket = !gsutil ls {files_path}*
    if files_in_bucket[-1].startswith(files_path):
        latest_file = files_in_bucket[-1]
        latest_file_nr = int(latest_file[len(files_path)+1:len(files_path)+5])
        latest_tasks[survey] = latest_file_nr

print('Latest tasks already in bucket:\n', latest_tasks)

# Get task list from GEE
gee_tasks = !earthengine task list

# Loop over these tasks. Save the latest in "last_tasks", if it's higher than what is already in the GCS bucket.
for line in gee_tasks:
    if 'Export.table' in line:
        task = line.split()[2]
        survey_string = task.split('_')[:2]
        survey = (survey_string[0], int(survey_string[1]))
        if survey not in surveys:
            continue
        task_nr = int(task.split('_')[2][:4])
        if task_nr > latest_tasks[survey]:
            latest_tasks[survey] = task_nr


print('Latest tasks already started in GEE:\n', latest_tasks)

chunk_size = 5
all_tasks = dict()

for survey in surveys:
    last_started = latest_tasks[survey]
    survey_df = df[(df['country'] == survey[0]) & (df['year'] == survey[1])]
    expected_nr_of_tasks = int(math.ceil(len(survey_df) / chunk_size))
    if last_started < expected_nr_of_tasks - 1:
        # Some tasks have not been started. Starts them here:
        country = survey[0]
        year = survey[1]
        already_in_bucket = list(range(last_started + 1))
        survey_tasks = export_images(df,
                                     country=country,
                                     year=year,
                                     export_folder=config['GCS']['EXPORT_FOLDER'],
                                     export='gcs',
                                     bucket=config['GCS']['BUCKET'],
                                     ms_bands=['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2', 'TEMP1'],
                                     include_nl=True,
                                     start_year=1990,
                                     end_year=2020,
                                     span_length=3,
                                     chunk_size=5,
                                     already_in_bucket=already_in_bucket)
        all_tasks.update(survey_tasks)

wait_on_tasks(all_tasks, poll_interval=60)