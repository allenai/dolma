from pyspark.sql import SparkSession
import os
import sys


def build_spark_session(master, driver_memory, executor_memory, num_cpus_per_task, executor_cores, driver_cores):
    """Build a spark session based on the master url and the number of cores and memory to use"""
    if master == "local":
        spark = local_session(executor_cores, driver_memory)
    else:
        spark = aws_ec2_s3_spark_session(driver_memory, executor_memory, num_cpus_per_task, executor_cores, driver_cores)

    return spark


def local_session(num_cores=4, mem_gb=16):
    """Build a local spark session"""
    spark = (
        SparkSession.builder.config("spark.driver.memory", str(mem_gb) + "G")
        .master("local[" + str(num_cores) + "]")
        .appName("extract_math")
        .getOrCreate()
    )
    return spark

def aws_ec2_s3_spark_session(driver_memory, executor_memory, num_cpus_per_task, executor_cores, driver_cores):
    """Build a spark session on AWS EC2"""
    driver_memory = str(int(driver_memory)) + 'g'
    # executor_memory = str(int(executor_memory)) + 'g'
    main_memory = str(int(executor_memory * 0.9)) + 'g'
    memory_overhead = str(executor_memory - int(executor_memory * 0.9)) + 'g'
    spark = (
        SparkSession.builder.appName("extractmath")
        .config("spark.executor.memory", main_memory)
        .config("spark.driver.memory", driver_memory)
        # .config("spark.driver.cores", str(driver_cores))
        .config("spark.executor.memoryOverhead", memory_overhead)
        .config("spark.executor.cores", str(executor_cores)) # Number of cpu cores available basically
        .config("spark.task.cpus", str(num_cpus_per_task))
        .config("spark.task.maxFailures", "10")
        .config("spark.dynamicAllocation.enabled", "true")
        .config("spark.shuffle.service.enabled", "true")
        .getOrCreate()
    )
    return spark