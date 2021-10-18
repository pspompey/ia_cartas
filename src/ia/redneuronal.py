#!/usr/bin/python3
import numpy as np
import tensorflow as tf
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from src.config import config
from IPython.display import display
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model

class RedNeuronal:
    def __init__(self, x_e, y_e, x_p, y_p, map):
        self.__x_e = x_e
        self.__y_e = y_e
        self.__x_p = x_p
        self.__y_p = y_p
        self.__map = map

    def crear_modelo(self, inputs, outputs, learn_rate):
        # Modelo con tres layers ocultos.
        hidden_layer_nodes = [inputs // 2, inputs // 4, inputs // 8]
        print("Se creara una RNA multiperceptrón con backpropagation")

        input_layer = Input(shape=(inputs,), name="input_img")
        previous_layer = input_layer

        # Hidden layers
        for x in range(len(hidden_layer_nodes)):
            layer_name = "hidden_" + str(x + 1)
            previous_layer = Dense(hidden_layer_nodes[x], activation="sigmoid", name=layer_name)(previous_layer)

        output_layer = Dense(outputs, activation="softmax", name="output")(previous_layer)

        decr_gradient = tf.keras.optimizers.Adam(learning_rate=learn_rate)

        model = Model(input_layer, output_layer, name="RedNeuronalArtificial")
        model.compile(optimizer=decr_gradient, loss="categorical_crossentropy", metrics=["accuracy"])

        print("Modelo creado con " + str(len(model.layers)) + " capas")
        model.summary()

        tf.keras.utils.plot_model(model, show_layer_names=True, show_shapes=True)

        return model

    def entrenar(self, model, epochs):
        x_train, x_verify, y_train, y_verify = train_test_split(self.__x_e, self.__y_e, test_size=0.1)
        print("Comenzando entrenamiento...")
        history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_verify, y_verify,), batch_size=4)
        print("Entrenamiento finalizado...")
        model.save_weights(config["archivo_weights"])
        self.__mostrar_datos_entrenamiento(history)

    def probar_modelo(self, x, y, mapa_clases, model):
        predicted_class = model.predict(x)
        class_predicted_array = []
        class_real_array = []

        for i in range(len(x)):
            real_class = mapa_clases[y[i]]
            idcl_pred = int(np.argmax(predicted_class[i], axis=0))
            idcl_pred_rnd = idcl_pred

            if idcl_pred_rnd < 0 or idcl_pred_rnd >= len(mapa_clases):
                cl_pred = "CLASE " + str(idcl_pred_rnd) + " INVÁLIDA!"
            else:
                cl_pred = mapa_clases[idcl_pred_rnd]

            class_real_array.append(real_class)
            class_predicted_array.append(cl_pred)

        print("Reporte de clasificacion:")
        print(classification_report(class_real_array, class_predicted_array))
        conf_matrix = confusion_matrix(class_real_array, class_predicted_array, labels=mapa_clases)
        confusion_matrix_dataframe = pd.DataFrame(
            conf_matrix,
            index=["r:{:}".format(x) for x in mapa_clases],
            columns=["p:{:}".format(x) for x in mapa_clases],
        )
        confusion_matrix_dataframe = confusion_matrix_dataframe.sort_index()
        cols = list(confusion_matrix_dataframe.columns.values)
        cols.sort(key=lambda l: int(l.replace("p:", "")))
        display(confusion_matrix_dataframe[cols])

    def __mostrar_datos_entrenamiento(self, history):
        plt.figure(figsize=(15, 8))
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Gráfico del Error del Entrenamiento')
        plt.ylabel('')
        plt.xlabel('epoch')
        plt.legend(['entrenamiento', 'validación'], loc='upper left')
        plt.show()

        plt.figure(figsize=(15, 8))
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Gráfico de la Exactitud del Entrenamiento')
        plt.ylabel('')
        plt.xlabel('epoch')
        plt.legend(['entrenamiento', 'validación'], loc='upper left')
        plt.show()
