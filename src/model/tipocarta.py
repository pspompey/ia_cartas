#!/usr/bin/python3

from enum import IntEnum

class TipoCarta(IntEnum):
    Espada = 1,
    Basto = 2,
    Oro = 3,
    Copa = 4,

    def __str__(self):
        if self == TipoCarta.Espada:   return "Espada"
        elif self == TipoCarta.Basto:  return "Basto"
        elif self == TipoCarta.Oro:    return "Oro"
        elif self == TipoCarta.Copa:   return "Copa"
        else:                             raise TypeError(self)
