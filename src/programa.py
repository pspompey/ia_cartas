#!/usr/bin/python3

from src.config import config
from src.ai.reconocedor import Reconocedor
from src.ai.aumentador import Aumentador

aumentador = Aumentador()
reconocedor = Reconocedor(aumentador)

datos = reconocedor.iniciar_aumentacion()
x_entrenamiento, y_entrenamiento, x_pruebas, y_pruebas, mapa_clases, y_util_entr, y_util_prue = reconocedor.crear_sets()

modelo = reconocedor.procesar_sets(
    x_entrenamiento,
    y_entrenamiento,
    x_pruebas,
    y_pruebas,
    mapa_clases,
    config["ancho_imagenes_a_procesar"] * config["alto_imagenes_a_procesar"] * 1,
    52,
    config["epochs"],
    config["tasa_aprendizaje"])

print("Resultados con datos de entrenamiento:")
# Prueba de modelo con los datos de las pruebas
reconocedor.probar_modelo(x_entrenamiento, y_util_entr, mapa_clases, modelo)

evaluacion = modelo.evaluate(x_pruebas, y_pruebas)
print("\n>Evaluación del Modelo: ")
print("    - Error: ", evaluacion[0])
print("    - Exactitud: ", evaluacion[1]*100)
print("\n")

print("Resultados con datos de prueba:")
reconocedor.probar_modelo(x_pruebas, y_util_prue, mapa_clases, modelo)
