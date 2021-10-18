#!/usr/bin/python3
import os
import pathlib
import numpy as np

from PIL import Image
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

from src.config import config
from src.model.carta import Carta
from src.model.cartas import cartas
from src.ai.redneuronal import RedNeuronal


class Reconocedor:
    def __init__(self, aumentador):
        self.__aumentador = aumentador
        self.__red = None

    def iniciar_aumentacion(self):
        self.__aumentador.preparar_procesamiento(pathlib.Path(__file__).parent.parent / "imagenes")

        # Primera etapa: populamos con el augmentator imagenes para cada carta.
        for carta, path in cartas.items():
            print("Invocando data augmentator para carta " + str(carta))
            self.__aumentador.procesar_imagen(int(Carta.to_number(carta)), path)

    def crear_sets(self):
        # Segunda etapa, cargamos las imagenes de entrenamiento y pruebas
        clases_entrenamiento, imagenes_entrenamiento = self.__cargar_imagenes(pathlib.Path(__file__).parent.parent / "imagenes" / "tmp" / "train")
        clases_pruebas, imagenes_pruebas = self.__cargar_imagenes(pathlib.Path(__file__).parent.parent / "imagenes" / "tmp" / "test")

        x_entrenamiento = self.__preparar_imagenes(imagenes_entrenamiento)
        x_pruebas = self.__preparar_imagenes(imagenes_pruebas)

        y_util_entr, y_entrenamiento, diccionario_mapeo = self.__preparar_clases(clases_entrenamiento)
        y_util_prue, y_pruebas, _ = self.__preparar_clases(clases_pruebas, diccionario_mapeo)

        mapa_clases = [x for x, y in diccionario_mapeo.items()]

        return x_entrenamiento, y_entrenamiento, x_pruebas, y_pruebas, mapa_clases, y_util_entr, y_util_prue

    def procesar_sets(self, x_e, y_e, x_p, y_p, mapa, inputs, outputs, epochs, learn_rate):
        self.__red = RedNeuronal(x_e, y_e, x_p, y_p, mapa)
        modelo = self.__red.crear_modelo(inputs, outputs, learn_rate)
        if os.path.exists(config["archivo_weights"]):
            modelo.load_weights(config["archivo_weights"])
        else:
            self.__red.entrenar(modelo, epochs)

        return modelo

    def probar_modelo(self, x, y, mapa, model):
        self.__red.probar_modelo(x, y, mapa, model)

    def __preparar_imagenes(self, lista_imagenes):
        array_imagenes = np.array(lista_imagenes).astype("float32") / 255.
        array_imagenes = array_imagenes.reshape((len(array_imagenes), config["ancho_imagenes_a_procesar"] * config["alto_imagenes_a_procesar"] * 1))
        return np.array(array_imagenes)

    def __preparar_clases(self, lista_clases, diccionario_mapeo=None):
        if diccionario_mapeo is None:
            aux_dict = list(set(lista_clases))
            diccionario_mapeo = dict(zip(aux_dict, range(len(aux_dict))))

        y = []

        for clase in lista_clases:
            y.append(diccionario_mapeo[clase])

        y_dummy = np_utils.to_categorical(y)
        return np.array(y), np.array(y_dummy), diccionario_mapeo

    def __cargar_imagenes(self, path_imagenes):
        clases = []
        imagenes = []

        print("Iniciando carga de imagenes a memoria")
        directorios = os.listdir(path_imagenes)
        directorios.sort()

        for directorio in directorios:
            print("Iniciando carga de clase " + str(directorio))
            path_dir = path_imagenes / directorio
            images = os.listdir(path_dir)

            for image_path in images:
                image = Image.open(path_dir / image_path)
                image = image.convert("L")
                image_array = np.array(image)

                clases.append(directorio)
                imagenes.append(image_array)

            print("Finalizando carga de clase " + str(directorio))
        return clases, imagenes
