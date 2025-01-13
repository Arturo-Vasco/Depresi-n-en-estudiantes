import pandas as pd
import joblib
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import LabelEncoder

# Cargar el modelo y las características seleccionadas
model_and_features = joblib.load(r"C:\Users\artur\Documents\FIA\Proyecto_Final\depression_random_forest_model.joblib")
list_features = list(model_and_features['feature_names'])

# Cargar el dataset  
dataframe = pd.read_csv(r"C:\Users\artur\Documents\FIA\Proyecto_Final\Student_Depression_Dataset.csv")  

# Limpiar los nombres de las columnas para evitar problemas  
dataframe.columns = dataframe.columns.str.strip().str.replace(" ", "_").str.lower()  # Normaliza a minúsculas.  

# Imprimir nombres de las columnas cargadas  
print("Columnas del DataFrame:")  
print(dataframe.columns.tolist())  

# Mostrar las características seleccionadas  
print("\nLas características seleccionadas son:")  
print(list_features)  

# Normalizar las características seleccionadas  
list_features = [feature.strip().replace(" ", "_").lower() for feature in list_features]  

# Validar que las características seleccionadas existan en el DataFrame  
missing_features = [feature for feature in list_features if feature not in dataframe.columns]  
if missing_features:  
    print("Las siguientes características no están en el dataset:", missing_features)  # Imprime características faltantes  
    raise ValueError(f"Las siguientes características no están en el dataset: {missing_features}")  

# Obtener las variables predictoras y objetivo  
X = dataframe[list_features]  
y = dataframe['depression']

# Dividir el conjunto de datos en entrenamiento (80%) y prueba (20%)  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  

print("Dimensiones de X_train:", X_train.shape)  
print("Dimensiones de y_train:", y_train.shape)  
print("¿X_train contiene valores nulos?", X_train.isnull().values.any())  
print("¿y_train contiene valores nulos?", y_train.isnull().values.any())  
# Eliminar valores nulos  
X_train = X_train.dropna()  
y_train = y_train[X_train.index]  # Asegúrate de que y_train tenga el mismo índice tras la eliminación.  

# O rellenar valores nulos  
X_train = X_train.fillna(0)  # O cualquier otro valor apropiado.

# Convertir columnas categóricas a valores numéricos (si es necesario)
label_encoder = LabelEncoder()
for column in X_train.select_dtypes(include=['object']).columns:
    X_train[column] = label_encoder.fit_transform(X_train[column])
    
# Entrenamiento del modelo de árbol de decisión
tree_model = DecisionTreeClassifier(criterion="gini", random_state=42)
tree_model.fit(X_train, y_train)

# Prueba del modelo
y_pred = tree_model.predict(X_test)

# Verificar si las clases son las mismas en y_test y y_pred
print("Clases de y_test:", set(y_test))
print("Clases de y_pred:", set(y_pred))

# Verificar si hay valores nulos en y_test o y_pred
print("¿Hay valores nulos en y_test?", y_test.isnull().any())
print("¿Hay valores nulos en y_pred?", pd.Series(y_pred).isnull().any())

# Calcular las métricas solo si hay al menos dos clases en y_test
if len(set(y_test)) > 1:
    # Calcular las métricas ('y' reales vs 'y' predichas)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    # Imprimir las métricas
    print("\nMétricas de rendimiento:\n")
    print("Exactitud (Accuracy):", accuracy)
    print("Precisión (Precision):", precision)
    print("Sensibilidad (Recall):", recall)
    print("Puntuación F1 (F1 Score):", f1)
    print("ROC AUC:", roc_auc)

    # Despliega la matriz de confusión
    print("Matriz de confusión:")
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    disp.plot(cmap="BuGn")
else:
    print("No se pueden calcular las métricas porque 'y_test' solo tiene una clase.")

# Visualización del árbol de decisión
px = 1 / plt.rcParams["figure.dpi"]  # Pixel en pulgadas

# Ajustes de imagen
fig_size = 1500
font_size = 2
dots_per_inch = 400

fig = plt.figure(figsize=(fig_size * px, fig_size * px))
_ = plot_tree(
    tree_model,
    feature_names=list_features,
    class_names=["Negative", "Positive"],
    filled=True,
    fontsize=font_size,
    rounded=True,
)

# Guardar la imagen del árbol
plt.savefig(
    r"C:\Users\artur\Documents\FIA\Proyecto_Final\arbol.png", dpi=dots_per_inch, bbox_inches="tight"
)
plt.show()

# Opción para probar el modelo con datos del usuario
resp = input("¿Desea probar el modelo con sus datos? (SI/NO): ")

if resp.upper() == "SI":
    print("\nIndique los datos que se le soliciten")
    prod_features = []

    for feat in list_features:
        value = float(input(f"{feat}: "))
        prod_features.append(value)

    # Crear un dataframe con los datos capturados
    df_prod = pd.DataFrame([prod_features], columns=list_features)

    # Predicción
    prediccion = tree_model.predict(df_prod)

    print("\nCon los datos:")
    print("Datos originales:", prod_features)

    # Muestra diagnóstico
    if prediccion[0] == 1:
        print("\nEl modelo predice que es muy propenso a tener depresión :(\n")
    else:
        print("\nEl modelo predice que no es propenso a tener depresión :)\n")
