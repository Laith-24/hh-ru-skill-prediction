"""
Loads a raw parquet file's rows on the Spark driver (via pyarrow) and hands
them to Spark as a distributed DataFrame via parallelize()+createDataFrame(),
instead of spark.read.parquet().

Needed because k8s executor pods don't share the driver's NFS filesystem --
normally that means input data must live in HDFS, but the driver itself
has NFS access, so this reads the file there and lets Spark ship the data
to executors over the network instead. Prefer spark.read.parquet(
"hdfs://...") when HDFS is healthy (executors then read their own
partitions in parallel); this is the fallback data_pipeline.load_and_
engineer() uses under --k8s otherwise.

_arrow_type_to_spark() exists because PySpark 3.1.1's Arrow<->Spark schema
conversion predates Arrow's large_string/large_binary/large_list types and
raises on them. Rows are built from Arrow's ChunkedArray.to_pylist()
rather than through pandas, which silently upcasts nullable ints to float
and breaks Spark's strict type checking.
"""
import logging

import pyarrow as pa
import pyarrow.parquet as pq
from pyspark.sql.types import (
    ArrayType, BinaryType, BooleanType, DateType, DoubleType, FloatType,
    IntegerType, LongType, MapType, NullType, ShortType, StringType,
    StructField, StructType, TimestampType,
)

logger = logging.getLogger(__name__)


def _arrow_type_to_spark(t):
    if pa.types.is_null(t):
        return NullType()
    if pa.types.is_boolean(t):
        return BooleanType()
    if pa.types.is_int16(t):
        return ShortType()
    if pa.types.is_int32(t):
        return IntegerType()
    if pa.types.is_int64(t):
        return LongType()
    if pa.types.is_float32(t):
        return FloatType()
    if pa.types.is_float64(t):
        return DoubleType()
    if pa.types.is_string(t) or pa.types.is_large_string(t):
        return StringType()
    if pa.types.is_binary(t) or pa.types.is_large_binary(t):
        return BinaryType()
    if pa.types.is_date(t):
        return DateType()
    if pa.types.is_timestamp(t):
        return TimestampType()
    if pa.types.is_list(t) or pa.types.is_large_list(t):
        return ArrayType(_arrow_type_to_spark(t.value_type))
    if pa.types.is_struct(t):
        return StructType([StructField(f.name, _arrow_type_to_spark(f.type), nullable=f.nullable) for f in t])
    if pa.types.is_map(t):
        return MapType(_arrow_type_to_spark(t.key_type), _arrow_type_to_spark(t.item_type))
    raise TypeError(f"Unhandled arrow type: {t}")


def load_local_parquet_distributed(spark, path: str, num_partitions: int = 64):
    """Read `path` via pyarrow on the driver and return a Spark DataFrame,
    without requiring executors to see the file at all."""
    logger.info(f"Reading {path} via pyarrow on the driver (executors can't see this path directly) ...")
    table = pq.read_table(path)
    spark_schema = StructType([
        StructField(f.name, _arrow_type_to_spark(f.type), nullable=f.nullable) for f in table.schema
    ])

    columns = table.column_names
    col_values = [table.column(name).to_pylist() for name in columns]
    rows = [dict(zip(columns, values)) for values in zip(*col_values)]
    logger.info(f"Converted {len(rows)} rows to native Python; distributing across {num_partitions} partitions ...")

    rdd = spark.sparkContext.parallelize(rows, numSlices=num_partitions)
    return spark.createDataFrame(rdd, schema=spark_schema)
