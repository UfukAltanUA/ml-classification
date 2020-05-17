# Databricks notebook source
from pyspark.sql import SparkSession



spark = SparkSession.builder.appName('TreeForCollege').getOrCreate()
df = spark.read.csv('/FileStore/tables/College.csv',header=True,inferSchema=True)
df.show()



df.columns



df.printSchema()



from pyspark.ml.feature import VectorAssembler



assembler = VectorAssembler(inputCols=['Apps',
 'Accept',
 'Enroll',
 'Top10perc',
 'Top25perc',
 'F_Undergrad',
 'P_Undergrad',
 'Outstate',
 'Room_Board',
 'Books',
 'Personal',
 'PhD',
 'Terminal',
 'S_F_Ratio',
 'perc_alumni',
 'Expend',
 'Grad_Rate'], outputCol='features')



output = assembler.transform(df)



from pyspark.ml.feature import StringIndexer
indexer = StringIndexer(inputCol='Private',outputCol='PrivateIndex')



outputFixed = indexer.fit(output).transform(output)



outputFixed.printSchema()



final_data = outputFixed.select('features','PrivateIndex')



train_data,test_data=final_data.randomSplit([0.75,0.25])



from pyspark.ml.classification import RandomForestClassifier,GBTClassifier, DecisionTreeClassifier
from pyspark.ml import Pipeline



dtc = DecisionTreeClassifier(labelCol='PrivateIndex', featuresCol='features')
rfc = RandomForestClassifier(numTrees=25,labelCol='PrivateIndex', featuresCol='features')
gbt = GBTClassifier(labelCol='PrivateIndex', featuresCol='features')



dtcModel = dtc.fit(train_data)
rfcModel = rfc.fit(train_data)
gbtModel = gbt.fit(train_data)



dtcPred = dtcModel.transform(test_data)
rfcPred = rfcModel.transform(test_data)
gbtPred = gbtModel.transform(test_data)



from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
binaryEval = BinaryClassificationEvaluator(labelCol='PrivateIndex')
multiEval = MulticlassClassificationEvaluator(metricName ='accuracy')



print('DTC Accuracy:')
binaryEval.evaluate(dtcPred)



print('RFC Accuracy:')
binaryEval.evaluate(rfcPred)



print('GBT Accuracy:')
binaryEval.evaluate(gbtPred)



cols = ['Apps',
 'Accept',
 'Enroll',
 'Top10perc',
 'Top25perc',
 'F_Undergrad',
 'P_Undergrad',
 'Outstate',
 'Room_Board',
 'Books',
 'Personal',
 'PhD',
 'Terminal',
 'S_F_Ratio',
 'perc_alumni',
 'Expend',
 'Grad_Rate']



for i,j in zip(cols,rfcModel.featureImportances):
  print(i + '\t\t' + str(j))



gbtPred.printSchema()



binaryEval2 = BinaryClassificationEvaluator(labelCol='PrivateIndex', rawPredictionCol='prediction')



print('GBT Accuracy2:')
binaryEval2.evaluate(gbtPred)



from pyspark.ml.evaluation import MulticlassClassificationEvaluator




