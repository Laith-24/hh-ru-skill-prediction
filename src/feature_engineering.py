"""
Feature Engineering - Extract features from raw data for ML models
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col, when, size, array_contains, explode, count, desc,
    regexp_extract, lit, split, trim, lower, coalesce, expr, array,
    array_remove
)
from pyspark.sql.types import ArrayType, StringType
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler,
    CountVectorizer, IDF, StandardScaler
)
from pyspark.ml import Pipeline
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, DoubleType, TimestampType
from collections import Counter
from itertools import combinations
import logging

logger = logging.getLogger(__name__)

# ============================================================
# Main Feature Engineering Functions
# ============================================================

def extract_basic_features(df: DataFrame) -> DataFrame:
    """Extract basic features from raw data"""
    
    # Extract seniority from job title
    df = df.withColumn("seniority",
        when(col("name").rlike("(?i)junior|trainee|intern|entry|начальный"), "Junior")
        .when(col("name").rlike("(?i)senior|lead|principal|head|старший|ведущий"), "Senior")
        .when(col("name").rlike("(?i)middle|mid"), "Middle")
        .otherwise("Unknown")
    )
    
    # ============================================================
    # SAFE SKILLS EXTRACTION - Handles both struct and string
    # ============================================================
    
    # Check if skills column exists
    if "key_skills_translated" in df.columns:
        skills_col = "key_skills_translated"
    elif "skills_translated" in df.columns:
        skills_col = "skills_translated"
    else:
        skills_col = "key_skills"
    
    # Get a sample to check the data type
    sample = df.select(col(skills_col)).limit(1).collect()
    is_struct = False
    if sample and sample[0][0] and len(sample[0][0]) > 0:
        first_skill = sample[0][0][0]
        if isinstance(first_skill, dict) or hasattr(first_skill, 'asDict'):
            is_struct = True
    
    if is_struct:
        # Skills are structs → extract name field
        df = df.withColumn("skills_clean",
            when(
                size(col(skills_col)) > 0,
                expr(f"transform({skills_col}, s -> lower(trim(s.name)))")
            ).otherwise(array())
        )
    else:
        # Skills are strings → use directly
        df = df.withColumn("skills_clean",
            when(
                size(col(skills_col)) > 0,
                expr(f"transform({skills_col}, s -> lower(trim(s)))")
            ).otherwise(array())
        )
    
    # Remove "unknown" and empty strings
    df = df.withColumn("skills_clean",
        when(
            array_contains(col("skills_clean"), lit("unknown")),
            array_remove(col("skills_clean"), "unknown")
        ).otherwise(col("skills_clean"))
    )

    df = df.withColumn("skills_clean",
        when(
            array_contains(col("skills_clean"), lit("")),
            array_remove(col("skills_clean"), "")
        ).otherwise(col("skills_clean"))
    )
    
    df = df.withColumn("skills_count", size(col("skills_clean")))
    
    # Extract city and employer
    df = df.withColumn("city", col("area.name"))
    df = df.withColumn("employer", col("employer.name"))
    
    # Salary
    df = df.withColumn("salary_avg",
        when(col("salary.from").isNotNull() & col("salary.to").isNotNull(),
             (col("salary.from") + col("salary.to")) / 2)
        .otherwise(None)
    )
    
    df = df.withColumn("has_salary", when(col("salary_avg").isNotNull(), 1).otherwise(0))
    
    # Top employer flag
    top_employers = ["СБЕР", "Яндекс", "Ozon", "Т-Банк", "VK", "МТС"]
    df = df.withColumn("is_top_employer", 
                       when(col("employer").isin(top_employers), 1).otherwise(0))
    
    return df


def build_ml_pipeline():
    """Build Spark ML pipeline for feature engineering"""
    
    # Index categorical features
    city_indexer = StringIndexer(inputCol="city", outputCol="city_index", handleInvalid="keep")
    employer_indexer = StringIndexer(inputCol="employer", outputCol="employer_index", handleInvalid="keep")
    seniority_indexer = StringIndexer(inputCol="seniority", outputCol="seniority_index", handleInvalid="keep")
    
    # One-hot encode categoricals
    city_encoder = OneHotEncoder(inputCol="city_index", outputCol="city_vector")
    employer_encoder = OneHotEncoder(inputCol="employer_index", outputCol="employer_vector")
    seniority_encoder = OneHotEncoder(inputCol="seniority_index", outputCol="seniority_vector")
    
    # TF-IDF for skills (using cleaned skills)
    cv = CountVectorizer(inputCol="skills_clean", outputCol="raw_features", vocabSize=500, minDF=5)
    idf = IDF(inputCol="raw_features", outputCol="skills_tfidf")
    
    # Assemble all features
    assembler = VectorAssembler(
        inputCols=["city_vector", "employer_vector", "seniority_vector", "skills_count", "skills_tfidf"],
        outputCol="features"
    )
    
    # Scale features
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=True)
    
    # Build pipeline
    pipeline = Pipeline(stages=[
        city_indexer, employer_indexer, seniority_indexer,
        city_encoder, employer_encoder, seniority_encoder,
        cv, idf, assembler, scaler
    ])
    
    return pipeline


def get_top_skills(df: DataFrame, n: int = 20):
    """Get top N most frequent skills using Spark"""
    
    top_skills = df.select(explode("skills_clean").alias("skill")) \
        .filter(col("skill").isNotNull() & (col("skill") != "") & (col("skill") != "unknown")) \
        .groupBy("skill") \
        .agg(count("*").alias("count")) \
        .orderBy(desc("count")) \
        .limit(n) \
        .collect()
    
    return [row.skill for row in top_skills]


def create_multi_label_columns(df: DataFrame, skill_list):
    """Create binary label columns for each skill"""
    for skill in skill_list:
        df = df.withColumn(
            f"label_{skill}",
            when(array_contains(col("skills_clean"), skill), 1).otherwise(0)
        )
    return df

def define_schema():
    """Define strictly typed schema for data validation"""
    schema = StructType([
        StructField("id", StringType(), nullable=False),
        StructField("name", StringType(), nullable=False),
        StructField("key_skills", ArrayType(StringType()), nullable=True),
        StructField("area", StructType([
            StructField("id", StringType(), nullable=True),
            StructField("name", StringType(), nullable=False)
        ]), nullable=False),
        StructField("employer", StructType([
            StructField("id", StringType(), nullable=True),
            StructField("name", StringType(), nullable=False)
        ]), nullable=False),
        StructField("salary", StructType([
            StructField("from", DoubleType(), nullable=True),
            StructField("to", DoubleType(), nullable=True)
        ]), nullable=True),
        StructField("published_at", TimestampType(), nullable=False)
    ])
    return schema

def analyze_skill_cooccurrence(df, top_n=20):
    """Analyze which skills frequently appear together"""
    skill_pairs = []
    for skills in df['skills_clean']:
        if skills and len(skills) > 1:
            for pair in combinations(skills, 2):
                skill_pairs.append(tuple(sorted(pair)))
    
    pair_counts = Counter(skill_pairs)
    top_pairs = pair_counts.most_common(top_n)
    
    print(f"Top {top_n} Skill Pairs:")
    for pair, count in top_pairs:
        print(f"  {pair[0]} + {pair[1]}: {count}")
    
    return top_pairs