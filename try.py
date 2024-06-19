from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

# Assuming your dataset is stored in a CSV file named 'your_dataset.csv'
file_path = 'dataset2.csv'

# Create a Spark session
spark = SparkSession.builder.appName("RestaurantRecommendation").getOrCreate()

# Load the dataset
df = spark.read.csv(file_path, header=True, inferSchema=True)

# Assuming 'name' is the restaurant, 'rating' is the user rating
df = df.selectExpr("name as restaurant", "rating")

# Rename 'name' to 'user' to match ALS user column
df = df.withColumnRenamed("name", "user")

# Split the dataset into train and test sets
train, test = df.randomSplit([0.8, 0.2], seed=42)

# Build the ALS model
als = ALS(maxIter=5, regParam=0.01, userCol="user", itemCol="restaurant", ratingCol="rating")
model = als.fit(train)

# Make predictions on the test set
predictions = model.transform(test)

# Evaluate the model
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Generate top N recommendations for a user
user_id = 1  # You can replace this with the actual user ID
user_recommendations = model.recommendForUserSubset(spark.createDataFrame([[user_id]], ["user"]), 5)
print(f"Top 5 recommended restaurants for user {user_id}:")
user_recommendations.show(truncate=False)
