# Databricks notebook source
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('Titanic').getOrCreate()
df = spark.read.csv('/FileStore/tables/titanic.csv',header=True,inferSchema=True)
df.show()



structure = [['1) df'],['2) Missing & Categorical'],
            ['3) StringIndexer & OneHotEncoder'], ['4) VectorAssembler'],['5) Random Split'],
            ['6) Model'], ['7) Pipeline'], ['8) Fit & Transform'],
            ['9) Evaluate']]

df.printSchema()

df.columns



cols = df.select(['Survived',
 'Pclass',
 'Sex',
 'Age',
 'SibSp',
 'Parch',
 'Fare',
 'Cabin',
 'Embarked'])



final_df = cols.na.drop()



from pyspark.ml.feature import VectorAssembler,StringIndexer,VectorIndexer,OneHotEncoder



genderIndexer = StringIndexer(inputCol='Sex', outputCol='SexIndex')
genderEncoder = OneHotEncoder(inputCol='SexIndex',outputCol='SexVec')



embarkIndexer = StringIndexer(inputCol='Embarked', outputCol='EmbarkIndex')
embarkEncoder = OneHotEncoder(inputCol='EmbarkIndex',outputCol='EmbarkVec')



assembler = VectorAssembler(inputCols=['Pclass','EmbarkVec', 'SexIndex', 'Age',
 'SibSp',
 'Parch',
 'Fare',], outputCol ='features')



from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline



logModel = LogisticRegression(featuresCol='features', labelCol='Survived' )



pipeline = Pipeline(stages=[genderIndexer,embarkIndexer,genderEncoder,embarkEncoder,assembler,logModel])



train_data,test_data = final_df.randomSplit([0.8,0.2])



fitted = pipeline.fit(train_data)



results = fitted.transform(test_data)



from pyspark.ml.evaluation import BinaryClassificationEvaluator



binaryEval = BinaryClassificationEvaluator(rawPredictionCol='prediction',labelCol='Survived')
AUC = binaryEval.evaluate(results)



AUC




