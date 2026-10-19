# TP2 - MLP en TNG: Clasificación de texto con 20 Newsgroups

Trabajo práctico de **Redes Neuronales en Bioingeniería** (ITBA). Continúa el
[TP1](https://github.com/fperel22/tp-softmax-20newsgroups): se reemplaza la regresión
softmax por un MLP en PyTorch (activaciones, profundidad, neuronas y optimizadores
configurables), con búsqueda de hiperparámetros en Optuna usando pruning.

## Cómo correrlo

1. Abrir `TP2_MLP_20Newsgroups.ipynb` en [Google Colab](https://colab.research.google.com/)
   con GPU (`Entorno de ejecución > Cambiar tipo de entorno de ejecución`).
2. Ejecutar las celdas en orden. La primera instala `mlflow` y `optuna`.
3. La búsqueda de 100 trials tardó unos 12 minutos en una GPU T4. El estudio se guarda
   en `optuna_tp2.db`, así que se puede retomar si se corta.

## Estructura del notebook

0-3. Setup, datos, preprocesamiento configurable y TF-IDF (reutilizados del TP1)
4. Modelo: MLP · 5. Entrenamiento · 6. Búsqueda con Optuna y pruning
7. Entrenamiento final · 8. Evaluación final
9. Experimento de overfitting · 10. Figuras · 11. Análisis

## Resultados principales

- **Búsqueda:** 100 trials (63 completos, 37 podados). Mejor configuración: 1 capa de
  512 neuronas, tanh, Adam, `batch_size` 128 y 10.000 features (val_acc 0.7556).
- **Test:** MLP 0.6977, contra 0.6569 del softmax del TP1. Un softmax con los
  hiperparámetros del TP2 llega a 0.6834, así que gran parte de la mejora no se debe
  a la capa oculta.
- **Overfitting:** aparece desde `1x64`; `1x1024` es el caso más claro. Más capacidad
  adelanta el sobreajuste, y más capas empeoran el rendimiento a igual ancho.
- **Pruning:** ahorró unos 124 s de entrenamiento, aunque el tiempo total lo dominan el
  TF-IDF y la conversión a tensores.
