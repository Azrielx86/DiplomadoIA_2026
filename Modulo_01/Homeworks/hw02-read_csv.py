"""
Tarea 2 - Diplomado de Inteligencia Artificial

La solución se basa en el uso de sets para omitir duplicados.
Se mapean las filas del csv en clases de Python que posteriormente se serializan en JSON usando el método __dict__ que
representa los atributos del objeto como un diccionario, lo cual puede ser serializado utilizando `json.dumps()`.

Equipo (nombres en classroom):
    - Edgar Chalico
    - Elsa Energy
    - Aldo Quiroz

"""

import csv
import json
from typing import Set
from sys import argv


class Data:
    def __init__(self, name: str, age: int, city: str) -> None:
        self.name = name
        self.age = age
        self.city = city

    # Para que set.add pueda hacer la comparación y validar que no hay datos repetidos
    # es necesario sobreescribir los métodos __hash__ y __eq__
    def __hash__(self) -> int:
        return hash((self.name, self.age, self.city))

    def __eq__(self, o: object) -> bool:
        return isinstance(o, Data) and (self.name == o.name
                                        and self.age == o.age
                                        and self.city == o.city)


if __name__ == '__main__':
    if len(argv) != 2:
        print("Ingrese el nombre del archivo")
        exit(1)

    filename: str = argv[1]

    set_data: Set[Data] = set()

    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Para escapar el header

        for row in reader:
            objeto = Data(row[0], int(row[1]), row[2])
            set_data.add(objeto)

    # __dict__ retorna el objeto como un diccionario de pares atributo:valor, con eso
    # json.dumps puede crear el archivo json
    # indent le agrega indentación al texto final
    json_data = json.dumps([obj.__dict__ for obj in set_data], indent=4)

    with open(filename.replace(".csv", ".json"), "w") as outfile:
        outfile.write(json_data)
