import argparse
from spark_session_builder import build_spark_session
from pyspark.sql import SparkSession
from pyspark import SparkContext
from extract_from_warc import process_warc
import jsonlines
import os
from pyspark.sql.types import StructType, StructField, StringType
import uuid

output_schema = StructType([
    StructField('url', StringType(), True),
    StructField('value', StructType([
        StructField('text', StringType(), True),
        StructField('html', StringType(), True),
        StructField('warc_path', StringType(), True),
        StructField('metadata', StringType(), True),
        StructField('date', StringType(), True),
    ]), True),
])

def process_filename(filename):
    if filename.startswith('file:'):
        filename = filename[5:]
    return filename

def test_process_warc(warc_file):
    yield ('test_url', ('test_html', 'test_text'))

def main(warc_file_list,
         output_dir,
         master='local',
         driver_memory=33,
         driver_cores=1,
         executor_memory=33,
         num_executors=29,
         executor_cores=5,
         num_cpus_per_task=5,
         num_output_partitions=1,
         output_format='parquet',
         output_compression='gzip'):
    spark = SparkSession.getActiveSession()
    if spark is not None:
        spark.stop()
    spark = build_spark_session(master=master, 
                                driver_memory=driver_memory, 
                                executor_memory=executor_memory, 
                                num_cpus_per_task=num_cpus_per_task,
                                executor_cores=executor_cores,
                                driver_cores=driver_cores)
    sc = SparkContext.getOrCreate()
    # warc_file_list is a text file with one warc file per line
    # Get the filename of warc_file_list
    warc_file_list_name = os.path.basename(warc_file_list).split('.')[0]
    warc_files = spark.read.text(warc_file_list).rdd.map(lambda r: r[0]).collect()
    warc_files = [process_filename(w) for w in warc_files]
    warc_count = len(warc_files)
    print('Found {} warc files'.format(warc_count))
    warc_rdd = sc.parallelize(warc_files, warc_count)

    df = warc_rdd.flatMap(process_warc).toDF(output_schema)

    # Split into partitions
    df = df.repartition(num_output_partitions)
    # Write the output
    unique_id = str(uuid.uuid4())
    output_file = os.path.join(output_dir, f'math_{warc_file_list_name}_{unique_id}')
    df.write.format(output_format).option('compression', output_compression).save(output_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--warc_file_list', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--master', type=str, default='local')
    parser.add_argument('--driver_memory', type=int, default=33)
    parser.add_argument('--driver_cores', type=int, default=1)
    parser.add_argument('--executor_memory', type=int, default=33)
    parser.add_argument('--num_executors', type=int, default=29)
    parser.add_argument('--executor_cores', type=int, default=5)
    parser.add_argument('--num_cpus_per_task', type=int, default=2)
    parser.add_argument('--mem_gb', type=int, default=16)
    parser.add_argument('--num_output_partitions', type=int, default=1)
    parser.add_argument('--output_format', type=str, default='json')
    parser.add_argument('--output_compression', type=str, default='gzip')
    args = parser.parse_args()
    main(warc_file_list=args.warc_file_list, 
         output_dir=args.output_dir, 
         master=args.master, 
         driver_memory=args.driver_memory,
         driver_cores=args.driver_cores,
         executor_memory=args.executor_memory,
         num_executors=args.num_executors,
         executor_cores=args.executor_cores,
         num_cpus_per_task=args.num_cpus_per_task,
         num_output_partitions=args.num_output_partitions,
         output_format=args.output_format,
         output_compression=args.output_compression)