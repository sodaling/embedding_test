from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from pyspark.ml.feature import VectorAssembler
from keras.models import Sequential
from keras.layers.core import *
from distkeras.trainers import *
from pyspark import SQLContext
from distkeras.transformers import *
from distkeras.predictors import *
from keras.layers import Convolution1D, GlobalMaxPooling1D

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


nb_filter = 250
filter_length = 3
hidden_dims = 250
nb_epoch = 2

sqlContext = SQLContext(sc)
reader = sqlContext
raw_dataset = reader.read.format('com.databricks.spark.csv').options(header='true', inferSchema='true').load(
    '/home/hpcc/test/mnist8.csv')
# raw_dataset = raw_dataset.repartition(4)
features = raw_dataset.columns
features.remove('label')
vector_assembler = VectorAssembler(inputCols=features, outputCol="features")
dataset = vector_assembler.transform(raw_dataset)
dataset = dataset.select("features", "label")

transformer = MinMaxTransformer(n_min=0.0, n_max=1.0, \
                                o_min=0.0, o_max=250.0, \
                                input_col="features", \
                                output_col="features_normalized")
# Transform the dataset.
dataset = transformer.transform(dataset)
dense_transformer = DenseTransformer(input_col="features_normalized", output_col="features_dense")
dataset = dense_transformer.transform(dataset)
reshape_transformer = ReshapeTransformer("features_dense", "matrix", (50, 1))
dataset = reshape_transformer.transform(dataset)
model1 = Sequential()
model1.add(Embedding(input_dim=256, output_dim=256, dropout=0.2))
# model1.add(LSTM(128,input_shape=(50,1)))
model1.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
# we use max pooling:
model1.add(GlobalMaxPooling1D())
model1.add(Dense(hidden_dims))
model1.add(Dropout(0.2))
model1.add(Activation('relu'))
model1.add(Dropout(0.5))
model1.add(Dense(1))
model1.add(Activation('softmax'))
model1.summary()
optimizer_mlp = 'adam'
loss_mlp = 'binary_crossentropy'
dataset.cache()
trainer = DOWNPOUR(keras_model=model1, worker_optimizer=optimizer_mlp, loss=loss_mlp, num_workers=7,
                   batch_size=8, communication_window=10, learning_rate=0.1, num_epoch=10,
                   features_col="features_dense", label_col="C0")
trainer.set_parallelism_factor(1)
trainer = ADAG(keras_model=model1, worker_optimizer=optimizer_mlp, loss=loss_mlp, num_workers=7,
               batch_size=16, communication_window=5, num_epoch=2,
               features_col="features_dense", label_col="C0")
trainer.set_parallelism_factor(1)
trained_model = trainer.train(dataset)
trainer = SingleTrainer(keras_model=model1, worker_optimizer=optimizer_mlp, loss=loss_mlp, batch_size=8,
                        num_epoch=1, features_col="features_dense", label_col="label")
trained_model = trainer.train(dataset)
print("Training time: " + str(trainer.get_training_time()))
predictor = ModelPredictor(keras_model=trained_model, features_col="features_dense")
test_set = dataset.select("features_dense", "label")
test_set = predictor.predict(test_set)
test_set.select("prediction", "label").show(20, False)
test_set2 = test_set.rdd.map(_transform).toDF()
from distkeras.evaluators import *
evaluator = AccuracyEvaluator(prediction_col="prediction", label_col="label")
score = evaluator.evaluate(test_set)
print("Training time: " + str(trainer.get_training_time()))
print("Accuracy: " + score)


