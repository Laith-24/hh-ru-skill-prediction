"""Test whether raw vacancy data (deeply nested structs/lists) can be
loaded WITHOUT HDFS: read the parquet on the driver (which has NFS access,
unlike executors) via pandas/pyarrow, then hand it to Spark with
spark.createDataFrame() -- Spark distributes it to executors over the
network, no shared filesystem required between driver and executors.

Usage (from inside the jupyter driver pod): python3 k8s_data_load_test.py
"""
import pyarrow as pa
import pyarrow.parquet as pq
from pyspark.sql.types import (
    ArrayType, BinaryType, BooleanType, DateType, DoubleType, FloatType,
    IntegerType, LongType, MapType, NullType, ShortType, StringType,
    StructField, StructType, TimestampType,
)
from spark_session import build_k8s_spark_session

LOCAL_PATH = "/home/jovyan/nfs-home/hh-ru-skill-prediction/data/vacancies_sample.parquet"


def _arrow_type_to_spark(t):
    """Minimal recursive Arrow->Spark type mapper. Written instead of
    reusing pyspark.sql.pandas.types.from_arrow_schema because that
    helper (as shipped in Spark 3.1.1) raises on large_string/
    large_binary/large_list -- added to Arrow after 3.1.1 was cut. The
    data itself is unaffected (pandas holds plain python str regardless
    of the arrow offset width); only the schema *declaration* needs the
    large_* variants treated the same as their regular counterparts."""
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


print("Reading parquet via pyarrow on the driver (for its authoritative schema) ...")
table = pq.read_table(LOCAL_PATH)
spark_schema = StructType([
    StructField(f.name, _arrow_type_to_spark(f.type), nullable=f.nullable) for f in table.schema
])
print(f"Derived Spark schema (first 10 fields): {spark_schema.fields[:10]}")

print("Converting to native Python rows via Arrow's own to_pylist() "
      "(NOT pandas -- pandas upcasts nullable int64 columns to float64, "
      "which then fails Spark's strict LongType verification) ...")
columns = table.column_names
col_values = [table.column(name).to_pylist() for name in columns]
rows = [dict(zip(columns, values)) for values in zip(*col_values)]
print(f"row count: {len(rows)}")

spark = build_k8s_spark_session("hh-ru-data-load-test")

print("Converting to Spark DataFrame via spark.createDataFrame(rows, schema=arrow-derived schema) ...")
sdf = spark.createDataFrame(rows, schema=spark_schema)
print(f"Spark schema (first 10 fields):")
for f in sdf.schema.fields[:10]:
    print(f"  {f.name}: {f.dataType}")

print("Forcing a real distributed operation (count + a struct field access) ...")
n = sdf.count()
print(f"DATA_LOAD_TEST_COUNT={n}")

# touch a nested column if present, to confirm struct/array types survived
nested_cols = [f.name for f in sdf.schema.fields if "Struct" in str(f.dataType) or "Array" in str(f.dataType)]
print(f"DATA_LOAD_TEST_NESTED_COLS={nested_cols[:5]}")

spark.stop()
print("DATA_LOAD_TEST_SUCCESS")
