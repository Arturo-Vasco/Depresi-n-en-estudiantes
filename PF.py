import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

# Cargar el dataset
file_path = r"C:\Users\artur\Documents\FIA\Proyecto_Final\Student_Depression_Dataset.csv"
dataframe = pd.read_csv(file_path)

# Preprocesamiento de datos
# 1. Eliminar columna 'id' porque no aporta valor predictivo
dataframe.drop(columns=['id'], inplace=True)

# 2. Convertir 'Sleep Duration' a numérico
dataframe['Sleep Duration'] = pd.to_numeric(dataframe['Sleep Duration'], errors='coerce')

# 3. Imputar valores faltantes en 'Financial Stress'
dataframe['Financial Stress'].fillna(dataframe['Financial Stress'].median(), inplace=True)

# 4. Verificar valores faltantes en todo el dataset
print("Valores nulos restantes:\n", dataframe.isnull().sum())

# 5. Eliminar filas con valores nulos restantes
dataframe.dropna(inplace=True)

# Separar características (X) y la variable objetivo (y)
X = dataframe.drop(columns=['Depression'])  # Todas las columnas excepto la variable objetivo
y = dataframe['Depression']  # Variable objetivo

# Dividir los datos en entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo de RandomForestClassifier
modelo = RandomForestClassifier(random_state=42)

# Inicializar variables para la selección de características
print("\nIniciando selección de características...")
best_features = pd.Index([])
scores = []
values = []
n_features = 0
roc = 0

# Probar diferentes cantidades de características seleccionadas
for i in range(2, X.shape[1] + 1):
    # Crear el selector RFE
    selector_caracteristicas = RFE(modelo, n_features_to_select=i, step=1)
    selector_caracteristicas = selector_caracteristicas.fit(X_train, y_train)

    # Seleccionar las características
    selected_features = X.columns[selector_caracteristicas.support_]

    # Crear nuevo dataset con las características seleccionadas
    X_new = X[selected_features]

    # Dividir nuevamente en entrenamiento y prueba
    X_ntrain, X_ntest, y_ntrain, y_ntest = train_test_split(X_new, y, test_size=0.2, random_state=42)

    # Entrenar el modelo con las características seleccionadas
    modelo.fit(X_ntrain, y_ntrain)

    # Evaluar con ROC AUC
    y_pred = modelo.predict(X_ntest)
    roc_auc = roc_auc_score(y_ntest, y_pred)

    # Guardar resultados para graficar
    scores.append(roc_auc)
    values.append(i)

    # Guardar las mejores características
    if roc_auc > roc:
        roc = roc_auc
        best_features = selected_features
        n_features = i

# Graficar los resultados de la selección de características
plt.figure(figsize=(10, 6))
plt.plot(values, scores, marker='o')
plt.xlabel('Número de características seleccionadas')
plt.ylabel('ROC AUC')
plt.title('Selección de características usando RFE')
plt.grid(True)
plt.show()

# Mostrar las mejores características seleccionadas
print("\nMejor ROC AUC: ", roc)
print(f"Mejores {n_features} características seleccionadas:\n")
print(best_features)

# Crear nuevo dataset con las mejores características
X_best = X[best_features]

# Dividir los datos finales con las mejores características
X_btrain, X_btest, y_btrain, y_btest = train_test_split(X_best, y, test_size=0.2, random_state=42)

# Entrenar el modelo final
modelo.fit(X_btrain, y_btrain)

# Realizar predicciones finales
y_pred = modelo.predict(X_btest)

# Calcular métricas de rendimiento
accuracy = accuracy_score(y_btest, y_pred)
precision = precision_score(y_btest, y_pred)
recall = recall_score(y_btest, y_pred)
f1 = f1_score(y_btest, y_pred)
roc_auc = roc_auc_score(y_btest, y_pred)

# Imprimir métricas
print("\nMétricas de rendimiento:\n")
print("Exactitud (Accuracy):", accuracy)
print("Precisión (Precision):", precision)
print("Sensibilidad (Recall):", recall)
print("Puntuación F1 (F1 Score):", f1)
print("ROC AUC:", roc_auc)

# Mostrar matriz de confusión
conf_matrix = confusion_matrix(y_btest, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap=plt.cm.PuBuGn)
plt.show()

# Guardar el modelo entrenado
output_file_name = r"C:\Users\artur\Documents\FIA\Proyecto_Final\depression_random_forest_model.joblib"
model_and_features = {'model': modelo, 'feature_names': best_features}
joblib.dump(model_and_features, output_file_name)
print(f"Modelo guardado como {output_file_name}")
