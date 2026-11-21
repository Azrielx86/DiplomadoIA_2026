from typing import Literal
from numpy import ndarray
import numpy as np
from typing import List


def activation(weights: ndarray, x: ndarray, b: float) -> Literal[0, 1]:
    """
    :arg weights: Pesos
    :arg x: Información de prueba
    :arg b: Sesgo
    """
    z = weights * x
    return 1 if z.sum() + b > 0 else 0


if __name__ == "__main__":
    w = np.random.uniform(-1, 1, size=2)
    b = np.random.uniform(-1, 1)

    # Características, por ejemplo una tarjeta de crédito
    # ¿Cómo saber si se le dará una TDC a alguien?
    # Edad (0.2 - 0.9) y Sueldo (0.1 == $10,000 MXN, 0.9 == $90,000 MXN)
    features_ejemplo = np.array([0.3, 0.4])

    result = activation(w, features_ejemplo, b)

    print(w)
    print(f"Result: {result}")
