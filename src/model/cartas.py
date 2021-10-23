#!/usr/bin/python3

from model.carta import Carta
from model.tipocarta import TipoCarta
import pathlib

__path = pathlib.Path(__file__).parent.parent / "imagenes"

cartas = {
    Carta(1, TipoCarta.Espada): (__path / "espada" / "1.png").absolute(),
    Carta(2, TipoCarta.Espada): (__path / "espada" / "2.png").absolute(),
    Carta(3, TipoCarta.Espada): (__path / "espada" / "3.png").absolute(),
    Carta(4, TipoCarta.Espada): (__path / "espada" / "4.png").absolute(),
    Carta(5, TipoCarta.Espada): (__path / "espada" / "5.png").absolute(),
    Carta(6, TipoCarta.Espada): (__path / "espada" / "6.png").absolute(),
    Carta(7, TipoCarta.Espada): (__path / "espada" / "7.png").absolute(),
    Carta(8, TipoCarta.Espada): (__path / "espada" / "8.png").absolute(),
    Carta(9, TipoCarta.Espada): (__path / "espada" / "9.png").absolute(),
    Carta(10, TipoCarta.Espada): (__path / "espada" / "10.png").absolute(),
    Carta(11, TipoCarta.Espada): (__path / "espada" / "11.png").absolute(),
    Carta(12, TipoCarta.Espada): (__path / "espada" / "12.png").absolute(),
    Carta(1, TipoCarta.Basto): (__path / "basto" / "1.png").absolute(),
    Carta(2, TipoCarta.Basto): (__path / "basto" / "2.png").absolute(),
    Carta(3, TipoCarta.Basto): (__path / "basto" / "3.png").absolute(),
    Carta(4, TipoCarta.Basto): (__path / "basto" / "4.png").absolute(),
    Carta(5, TipoCarta.Basto): (__path / "basto" / "5.png").absolute(),
    Carta(6, TipoCarta.Basto): (__path / "basto" / "6.png").absolute(),
    Carta(7, TipoCarta.Basto): (__path / "basto" / "7.png").absolute(),
    Carta(8, TipoCarta.Basto): (__path / "basto" / "8.png").absolute(),
    Carta(9, TipoCarta.Basto): (__path / "basto" / "9.png").absolute(),
    Carta(10, TipoCarta.Basto): (__path / "basto" / "10.png").absolute(),
    Carta(11, TipoCarta.Basto): (__path / "basto" / "11.png").absolute(),
    Carta(12, TipoCarta.Basto): (__path / "basto" / "12.png").absolute(),
    Carta(1, TipoCarta.Oro): (__path / "oro" / "1.png").absolute(),
    Carta(2, TipoCarta.Oro): (__path / "oro" / "2.png").absolute(),
    Carta(3, TipoCarta.Oro): (__path / "oro" / "3.png").absolute(),
    Carta(4, TipoCarta.Oro): (__path / "oro" / "4.png").absolute(),
    Carta(5, TipoCarta.Oro): (__path / "oro" / "5.png").absolute(),
    Carta(6, TipoCarta.Oro): (__path / "oro" / "6.png").absolute(),
    Carta(7, TipoCarta.Oro): (__path / "oro" / "7.png").absolute(),
    Carta(8, TipoCarta.Oro): (__path / "oro" / "8.png").absolute(),
    Carta(9, TipoCarta.Oro): (__path / "oro" / "9.png").absolute(),
    Carta(10, TipoCarta.Oro): (__path / "oro" / "10.png").absolute(),
    Carta(11, TipoCarta.Oro): (__path / "oro" / "11.png").absolute(),
    Carta(12, TipoCarta.Oro): (__path / "oro" / "12.png").absolute(),
    Carta(1, TipoCarta.Copa): (__path / "copa" / "1.png").absolute(),
    Carta(2, TipoCarta.Copa): (__path / "copa" / "2.png").absolute(),
    Carta(3, TipoCarta.Copa): (__path / "copa" / "3.png").absolute(),
    Carta(4, TipoCarta.Copa): (__path / "copa" / "4.png").absolute(),
    Carta(5, TipoCarta.Copa): (__path / "copa" / "5.png").absolute(),
    Carta(6, TipoCarta.Copa): (__path / "copa" / "6.png").absolute(),
    Carta(7, TipoCarta.Copa): (__path / "copa" / "7.png").absolute(),
    Carta(8, TipoCarta.Copa): (__path / "copa" / "8.png").absolute(),
    Carta(9, TipoCarta.Copa): (__path / "copa" / "9.png").absolute(),
    Carta(10, TipoCarta.Copa): (__path / "copa" / "10.png").absolute(),
    Carta(11, TipoCarta.Copa): (__path / "copa" / "11.png").absolute(),
    Carta(12, TipoCarta.Copa): (__path / "copa" / "12.png").absolute(),
}
