#!/usr/bin/python3
from model.tipocarta import TipoCarta


class Carta:
    def __init__(self, numero, tipo):
        self.numero = numero
        self.tipo = tipo
    
    def __str__(self):
        return str(self.numero) + " de " + str(self.tipo)

    def __hash__(self):
        return hash((self.numero, self.tipo))

    def __eq__(self, other):
        return (self.numero, self.tipo) == (other.numero, other.tipo)

    def __ne__(self, other):
        return not(self == other)

    # Metodos estaticos para convertir las cartas en numeros
    def to_number(carta):
        return float(carta.numero + (int(carta.tipo) - 1) * 6)

    def to_carta(numero):
        numero_redondeado = int(numero + 0.5)
        numero = numero_redondeado % 6
        if numero == 0:
            numero = 6
        tipo = ((numero_redondeado - 1) // 6) + 1
        return Carta(numero, TipoCarta(tipo))
