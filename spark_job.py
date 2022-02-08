
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--bucket", help="bucket for input and output")
args = parser.parse_args()

BUCKET = args.bucket
from pyspark.sql import SparkSession, SQLContext, Row
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType, FloatType, DateType
from pyspark.sql.functions import regexp_replace
from pyspark.sql.functions import when

# Creating Spark Session
spark = SparkSession.builder.appName("LinearRegression_with_HouseData").getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("ERROR")

gcs_bucket='dataproc-staging-us-central1-1012630253058-dtwj7n8q/notebooks/jupyter/'
data_file = "gs://" + gcs_bucket + "kc_house_data.csv"

# reading csv
house_df = spark.read.option("header", True).csv(data_file) 

# changing date rows 20141013T000000 ---> 20141013
house_df = house_df.withColumn("date"
                        ,when(house_df.date.endswith("T000000"), regexp_replace(house_df.date,"T000000","")))

# changing data types required
house_df = house_df.withColumn("id", col("id").cast(IntegerType())) \
                   .withColumn("price", col("price").cast(FloatType())) \
                   .withColumn("bedrooms", col("bedrooms").cast(IntegerType())) \
                   .withColumn("bathrooms", col("bathrooms").cast(FloatType())) \
                   .withColumn("sqft_living", col("sqft_living").cast(IntegerType())) \
                   .withColumn("sqft_lot", col("sqft_lot").cast(IntegerType())) \
                   .withColumn("floors", col("floors").cast(FloatType())) \
                   .withColumn("waterfront", col("waterfront").cast(IntegerType())) \
                   .withColumn("view", col("view").cast(IntegerType())) \
                   .withColumn("condition", col("condition").cast(IntegerType())) \
                   .withColumn("grade", col("grade").cast(IntegerType())) \
                   .withColumn("sqft_above", col("sqft_above").cast(IntegerType())) \
                   .withColumn("sqft_basement", col("sqft_basement").cast(IntegerType())) \
                   .withColumn("yr_built", col("yr_built").cast(DateType())) \
                   .withColumn("yr_renovated", col("yr_renovated").cast(DateType())) \
                   .withColumn("zipcode", col("zipcode").cast(IntegerType())) \
                   .withColumn("lat", col("lat").cast(FloatType())) \
                   .withColumn("long", col("long").cast(FloatType())) \
                   .withColumn("sqft_living15", col("sqft_living15").cast(IntegerType())) \
                   .withColumn("sqft_lot15", col("sqft_lot15").cast(IntegerType()))
# Data exploration
columns_for_lr = ["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living", "price"]
print(house_df.printSchema())
print(house_df.select(columns_for_lr).show(5, True))
# Data Describe
house_df[columns_for_lr].describe().toPandas().T
# Correlation between features and target 
import six

for i in house_df[columns_for_lr].columns:
    if not( isinstance(house_df.select(i).take(1)[0][0], six.string_types)):
        print("Correlation to price for", i, house_df.stat.corr('price',i))
from pyspark.ml.feature import VectorAssembler

# Preparing to Linear Regression
features = ["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"] 
vectorAssembler = VectorAssembler(inputCols = features, outputCol = 'Features')
vhouse_df = vectorAssembler.transform(house_df)
vhouse_df = vhouse_df.select(['Features', 'price'])
print("\nAfter Vector Assembling")
vhouse_df.show(5)
# Training, Test Split
(trainingData, testData) = vhouse_df.randomSplit([0.7, 0.3])

print("Training data")
print(trainingData.show(5))

print("\nTest data")
print(testData.show(5))
print("Total trainingData instance :", trainingData.count(),"\nTotal testData instance :", testData.count())
from pyspark.ml.regression import LinearRegression

print("*** Linear Regression ***")
lr = LinearRegression(featuresCol = 'Features', labelCol='price', maxIter = 20, regParam = 0.3, elasticNetParam = 0.8)
lr_model = lr.fit(trainingData)
print("Coefficients: " + str(lr_model.coefficients))
print("\nIntercept: " + str(lr_model.intercept))
trainingSummary = lr_model.summary
print("Model Summary for Linear Regression")
print("-----------------------------------")
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)
print("Prediction for Linear Regression Model")
print("--------------------------------------")
lr_predictions = lr_model.transform(testData)
lr_predictions.select("features","price","prediction").show(5)

from pyspark.ml.evaluation import RegressionEvaluator

# Evaluator for test data
lr_evaluator = RegressionEvaluator(labelCol="price",
                                   predictionCol="prediction",
                                   metricName="r2")

# R2 score for test data
r2 = lr_evaluator.evaluate(lr_predictions)
print("R Squared (R2) on test data = %g" % lr_evaluator.evaluate(lr_predictions))
from pyspark.ml.regression import GBTRegressor

# Gradient-boosted Tree Regression
gbt = GBTRegressor(featuresCol = 'Features', labelCol = 'price', maxIter=10)
gbt_model = gbt.fit(trainingData)
gbt_predictions = gbt_model.transform(testData)
print("\n*** Gradient-boosted Tree Regression *** ")
gbt_predictions.select('prediction', 'price', 'features').show(5)
# Gradient-boosted Tree Regression Evaluator
gbt_evaluator = RegressionEvaluator(
    labelCol="price", predictionCol="prediction", metricName="r2")
r2 = gbt_evaluator.evaluate(gbt_predictions)
print("\n*** Gradient-boosted Tree Regression - R Squared (R2) Score *** ")
print("---------------------------------------------------------------")
print("R Squared (R2) on test data = %g" % r2)

import google.cloud.storage as gcs
bucket = gcs.Client().get_bucket(BUCKET)
for blob in bucket.list_blobs(prefix='sparkml/'):
    blob.delete()
