# Visualizador 3D ─ Control Bimanual con MediaPipe

Este proyecto permite visualizar un modelo 3D en **formato `.ply`** y controlarlo en tiempo real mediante gestos de ambas manos usando **MediaPipe**.

## Requisitos

1. Python ≥ 3.10
2. Tarjeta gráfica compatible con OpenGL 2.1 (o superior)
3. Webcam HD (integrada o USB)
4. Sistema operativo Windows 10/11, macOS o Linux

### Dependencias de Python

Instala los paquetes especificados en `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate   # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **Nota:** En Windows puede que necesites instalar los *wheels* oficiales de `mediapipe` y `PyOpenGL` desde [PyPI](https://pypi.org/).

## Ejecución

```bash
python viewer_mediapipe.py --model Espino.ply
```

Parámetros opcionales:

* `--model RUTA` – ruta al modelo principal `.ply`. Si no se indica usará la variable de entorno `PLY_MODEL` o `Espino.ply` por defecto.

## Controles y Gestos

| Entrada | Acción |
|---------|--------|
| **Mano derecha** | |
| ‑ Desplazar (x) | Rota cámara `rotY` |
| ‑ Desplazar (y) | Rota cámara `rotX` |
| ‑ Pinch (pulgar-índice) | Zoom (5 → 25 u) |
| **Mano izquierda** | |
| ‑ Pinch | Activa `EFFECT_1` (demo) |
| **Ambas manos** | |
| ‑ Tocar palma izquierda con índice derecho | Cicla modo (`NEUTRAL → EFFECT_1 → EFFECT_2 → …`) |
| **Teclado** | |
| `F` | Pantalla completa on/off |
| `C` | Mostrar/ocultar cámara |
| `Esc` | Salir |

## Estructura del Proyecto

```text
ANTRO03/
│ viewer_mediapipe.py   # script principal
│ Espino.ply            # modelo de ejemplo
│ requirements.txt      # dependencias
└─ README.md            # este archivo
```

## Extensión y Personalización

* Sustituye `Espino.ply` por cualquier otra malla (ASCII o binaria); el cargador la normalizará (centrado y escala).
* Añade nuevos estados en `ViewerState` y define efectos en `_apply_state_visuals_pre`.
* Ajusta constantes (auto-rotación, umbrales de gestos, etc.) en la sección **CONFIGURATION** del script.

## Solución de Problemas

1. **Pantalla negra / sin puntos:** Asegúrate de que tu GPU soporta OpenGL 2.1+ y que los drivers están actualizados.
2. **Webcam no encontrada:** Cambia el índice `cv2.VideoCapture(0)` a otro número (`1`, `2`, …).
3. **`ModuleNotFoundError: mediapipe`** – instala la versión indicada en `requirements.txt`.

## Créditos

* Google MediaPipe – detección de manos.
* PyOpenGL, Pygame, OpenCV, NumPy, plyfile.
