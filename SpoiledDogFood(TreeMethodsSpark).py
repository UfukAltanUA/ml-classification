# Databricks notebook source
from pyspark.sql import SparkSession



spark = SparkSession.builder.appName('TreeForCollege').getOrCreate()
df = spark.read.csv('/FileStore/tables/dog_food.csv',header=True,inferSchema=True)
df.show()



structure = [['1) df'],['2) Missing & Categorical'],
            ['3) StringIndexer & OneHotEncoder'], ['4) VectorAssembler'],['5) Random Split'],
            ['6) Model'], ['7) Pipeline'], ['8) Fit & Transform'],
            ['9) Evaluate']]



df.printSchema()



df.describe().show()



from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols=['A','B','C','D'], outputCol='features')
output = assembler.transform(df)



output.printSchema()



from pyspark.ml.classification import RandomForestClassifier,GBTClassifier, DecisionTreeClassifier
rfc = RandomForestClassifier(labelCol='Spoiled', featuresCol='features')



final_data = output.select('features', 'Spoiled')



final_data.show()



rfcModel = rfc.fit(final_data)



structure



rfcModel.featureImportances




