# TP2 - MLP en TNG: Clasificación de texto con 20 Newsgroups

Trabajo práctico de **Redes Neuronales en Bioingeniería** (ITBA). Continúa el
[TP1](https://github.com/fperel22/tp-softmax-20newsgroups): se reemplaza la regresión
softmax por un MLP en PyTorch (activaciones, profundidad, neuronas y optimizadores
configurables), con búsqueda de hiperparámetros en Optuna usando pruning.

## Cómo correrlo

1. Abrir `TP2_MLP_20Newsgroups.ipynb` en [Google Colab](https://colab.research.google.com/)
   con GPU (`Entorno de ejecución > Cambiar tipo de entorno de ejecución`).
2. Ejecutar las celdas en orden. La primera instala `mlflow` y `optuna`.
3. La búsqueda de 100 trials tardó unos 15 minutos en una GPU T4. El estudio se guarda
   en `optuna_tp2.db`, así que se puede retomar si se corta.

## Estructura del notebook

0-3. Setup, datos, preprocesamiento configurable y TF-IDF (reutilizados del TP1)
4. Modelo: MLP · 5. Entrenamiento · 6. Búsqueda con Optuna y pruning
7. Entrenamiento final · 8. Evaluación final
9. Experimento de overfitting · 10. Figuras · 11. Análisis

## Resultados principales

- **Búsqueda:** 100 trials (51 completos, 49 podados). Mejor configuración: 1 capa de
  512 neuronas, GELU, RMSprop, `batch_size` 128 y 10.000 features (val_acc 0.7388).
- **Test:** MLP 0.6808, contra 0.6569 del softmax del TP1 y 0.6559 de un softmax con los
  hiperparámetros del TP2. La capa oculta aporta unos 2.5 puntos, con una sola semilla.
- **Overfitting:** se adelanta con el ancho y la profundidad; `3x256` es un caso claro.
  Más capas empeoran el rendimiento a igual ancho.
- **Pruning:** ahorró unos 177 s de entrenamiento (~17% del tiempo total estimado).
- **Reproducibilidad:** dos corridas limpias con la misma semilla dieron resultados idénticos.
