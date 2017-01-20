from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from pyspark.ml.feature import VectorAssembler
from keras.models import Sequential
from keras.layers.core import *
from distkeras.trainers import *
from pyspark import SQLContext
from distkeras.transformers import *
from distkeras.predictors import *


def get_index(vector):
    if vector > 0.55:
        index = 1
    else:
        index = 0
    return index


def _transform(row):
    prediction = row["prediction"]
    index = float(get_index(prediction))
    from distkeras.utils import new_dataframe_row
    new_row = new_dataframe_row(row, "prediction_index", index)
    return new_row




sqlContext = SQLContext(sc)
reader = sqlContext
raw_dataset = reader.read.format('com.databricks.spark.csv').options(header='false', inferSchema='true').load(
    '/home/hpcc/test/11.csv')
# raw_dataset = raw_dataset.repartition(4)
features = raw_dataset.columns
features.remove('C0')
vector_assembler = VectorAssembler(inputCols=features, outputCol="features")
dataset = vector_assembler.transform(raw_dataset)
dataset = dataset.select("features", "C0")
dense_transformer = DenseTransformer(input_col="features", output_col="features_dense")
dataset = dense_transformer.transform(dataset)
model1 = Sequential()
model1.add(Embedding(input_dim=52965, output_dim=256))
model1.add(LSTM(128))
model1.add(Dropout(0.5))
model1.add(Dense(1))
model1.add(Activation('sigmoid'))
model1.summary()
optimizer_mlp = 'adam'
loss_mlp = 'binary_crossentropy'
dataset.cache()
# trainer = DOWNPOUR(keras_model=model1, worker_optimizer=optimizer_mlp, loss=loss_mlp, num_workers=7,
#                    batch_size=8, communication_window=10, learning_rate=0.1, num_epoch=10,
#                    features_col="features_dense", label_col="C0")
# trainer.set_parallelism_factor(1)
# trainer = ADAG(keras_model=model1, worker_optimizer=optimizer_mlp, loss=loss_mlp, num_workers=7,
#                batch_size=16, communication_window=5, learning_rate=0.01, num_epoch=1,
#                features_col="features_dense", label_col="C0")
# trainer.set_parallelism_factor(1)
# trained_model = trainer.train(training_set)
trainer = SingleTrainer(keras_model=model1, worker_optimizer=optimizer_mlp, loss=loss_mlp, batch_size=8,
                        num_epoch=5, features_col="features_dense", label_col="C0")
trained_model = trainer.train(dataset)
print("Training time: " + str(trainer.get_training_time()))
predictor = ModelPredictor(keras_model=trained_model, features_col="features_dense")
test_set = dataset.select("features_dense", "C0")
test_set = predictor.predict(test_set)
test_set.select("prediction", "C0").show(20, False)
test_set2 = test_set.rdd.map(_transform).toDF()
from distkeras.evaluators import *
evaluator = AccuracyEvaluator(prediction_col="prediction_index", label_col="C0")
score = evaluator.evaluate(test_set2)
print("Training time: " + str(trainer.get_training_time()))
print("Accuracy: " + score)


