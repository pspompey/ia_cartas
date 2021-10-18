#!/usr/bin/python3
import os
import shutil

import cv2
import matplotlib.pyplot as plt
from PIL import Image
from PIL._imaging import display

from src.config import config
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib.pyplot import imshow, subplots, show

class Aumentador:

    def preparar_procesamiento(self, source):
        if not os.path.exists(source / "tmp"):
            os.makedirs(source / "tmp")

    def procesar_imagen(self, label, source):
        print("Iniciando procesamiento de " + str(source))
        tmp_dir = source.parent.parent / "tmp"
        dest_inicial = tmp_dir / (str(label) + ".png")

        if not os.path.exists(dest_inicial):
            self.__transformar_imagen_inicial(source, dest_inicial)
        print("Validada existencia de source preprocesado para label " + str(label))

        self.__preparar_filesystem(dest_inicial, dest_inicial.parent / "train" / str(label))
        self.__preparar_filesystem(dest_inicial, dest_inicial.parent / "test" / str(label))
        print("Filesystem preparado para label " + str(label))

        # image_generator = ImageDataGenerator(
        #    rotation_range=180,
        #    width_shift_range=0.2,
        #    height_shift_range=0.2,
        #    brightness_range=(0.5, 1.5),
        #    shear_range=0.5,
        #    zoom_range=0.8,
        #    fill_mode="nearest",
        #)

        image_generator = ImageDataGenerator(
                           rescale=1. / 255,
                           rotation_range=90,
                           shear_range=0.3,
                           zoom_range=0.3,
                           horizontal_flip=False,
                           vertical_flip=False)

        path_imagen = str(dest_inicial)
        self.__procesar_imagen(image_generator, tmp_dir / "train", label, int(config["imagenes_por_carta"] * 1.1))
        self.__procesar_imagen(image_generator, tmp_dir / "test", label, int(config["imagenes_por_carta"] * 0.25))
        os.remove(tmp_dir / "train" / str(label) / "orig.png")
        os.remove(tmp_dir / "test" / str(label) / "orig.png")

        print("Finalizada validacion de existencia de archivos por data augmentation")

    def __transformar_imagen_inicial(self, source, dest):
        image = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (config["ancho_imagenes"], config["alto_imagenes"]))
        delta_w = config["ancho_imagenes_a_procesar"] - config["ancho_imagenes"]
        delta_h = config["alto_imagenes_a_procesar"] - config["alto_imagenes"]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        color = [255, 255, 255]
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        cv2.imwrite(str(dest), image)

    def __mostrar_imagen(self, path):
        print("Mostrando imagen " + str(path))
        image = Image.open(path)
        image.show()

    def __mostrar_todas_las_imagenes(self, path):
        print("Mostrando imagenes en ruta " + str(path))
        for image in os.listdir(path):
            self.__mostrar_imagen(path / image)

    def __preparar_filesystem(self, path_imagen, path_destino):
        print("Preparando filesystem para imagen " + str(path_imagen))
        _, image_name = os.path.split(path_imagen)
        print("File: " + image_name)

        if not os.path.exists(path_destino):
            os.makedirs(path_destino)

        shutil.copy(path_imagen, str(path_destino / "orig.png"))

    def __procesar_imagen(self, image_generator, path_destino, label, cantidad):
        print("Procesando destino " + str(path_destino) + " para label " + str(label))

        if not (os.path.exists(path_destino / str(label)) and len(os.listdir(path_destino / str(label))) == cantidad + 1):
            for f in os.listdir(path_destino / str(label)):
                if str(f) != "orig.png":
                    try:
                        os.remove(path_destino / str(label) / f)
                    except:
                        print("Fallo al borrar " + str(f))

            image_data = image_generator.flow_from_directory(
                str(path_destino),
                classes=[str(label)],
                save_to_dir=str(path_destino / str(label)),
                save_prefix="da_",
                target_size=(config["ancho_imagenes_a_procesar"], config["alto_imagenes_a_procesar"]),
                color_mode="grayscale")

            for i in range(cantidad):
                image_data.next()

            if len(os.listdir(path_destino / str(label))) != cantidad + 1:
                # Reprocesamos si no juntamos la cantidad exacta hasta que funcione bn
                self.__procesar_imagen(image_generator, path_destino, label, cantidad)
