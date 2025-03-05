import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu, pearsonr, spearmanr, shapiro, kruskal
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import t
from statsmodels.stats.proportion import proportion_confint

df = pd.read_csv("data/ensanut2020_limpio.csv")

name_ent = df["Desc_ent"]
name_ent
df = df.drop(columns=["Desc_ent"])

glucosa_min = 126
hb1ac_min = 6.5

predictoras = ['Edad', 'Entidad', 'Sexo', 'Valor.trig', 'Valor.col_hdl', 'Valor.col_ldl', 
               'Valor.creat', 'Valor.ac_urico', 'Valor.ggt', 'Valor.alt', 'Valor.aat']

nombre_predictoras = ["Edad","Entidad","Sexo","Trigliceridos","Colesterol HDL", "Colesterol LDL","Creatinina","Ácido úrico","GGT","Trasaminasas ALT","Aspartato aminotransferasa"]

df_new = df

df_new["Diabetes"] = ((df_new["Valor.glu_suero"] >= glucosa_min) | (df_new["Hb1ac.valor"] >= hb1ac_min))
df["Diabetes"] = ((df["Valor.glu_suero"] >= glucosa_min) | (df["Hb1ac.valor"] >= hb1ac_min))

df.describe()
df_new.describe()
df_new["Sexo"]

# Selección de variables predictoras
# Excluimos 'diabetes' como etiqueta objetivo y 'hb1ac.valor', 'glucosa' para evitar circularidad

X = df_new[predictoras]
y = df_new['Diabetes']

df_new["Diabetes_n"] = df_new["Diabetes"].astype(int)
df["Diabetes_n"] = df["Diabetes"].astype(int)

diabetes_str = df_new["Diabetes"].astype(str)

df_new.describe()

X = X.dropna()
y = y[X.index]

# División de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

# --- Modelo: Random Forest con GridSearch ---
model = RandomForestClassifier(random_state=42)

# Optimización
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 10],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc')
grid_search.fit(X_train, y_train)

# Mejor modelo encontrado
best_model = grid_search.best_estimator_
print("Mejores parámetros:", grid_search.best_params_)

# --- Evaluación ---
# predicciones y probando
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

# Reporte de clasificación
print(classification_report(y_test, y_pred))

# ROC-AUC
roc_auc = roc_auc_score(y_test, y_proba)
print("ROC-AUC:", roc_auc)

# Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure()
plt.plot(fpr, tpr, label="ROC Curve (AUC = {:.3f})".format(roc_auc))
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="best")
plt.show()

# -- Importancia de variables --
importances = best_model.feature_importances_
importance_df = pd.DataFrame({'Variable': predictoras, 'Importancia': importances})
importance_df = importance_df.sort_values(by='Importancia', ascending=False)

# Gráfico

plt.figure(figsize=(8, 6))
sns.barplot(x='Importancia', y='Variable', data=importance_df)
plt.title('Importancia de Variables')
plt.show()

# -- Correlaciones entre variables --
plt.figure(figsize=(10, 8))
sns.heatmap(df_new[predictoras + ['Diabetes']].corr(), annot=True, cmap="coolwarm", fmt=".2f")
#plt.xticks(ticks=range(len(nombre_predictoras)),labels=nombre_predictoras)
#plt.yticks(ticks=range(len(nombre_predictoras)),labels=nombre_predictoras)
plt.title("Mapa de Calor")
plt.show()

# distribución variables
fig, ax = plt.subplots(figsize=(6, 6))
sns.histplot(
    data    = df_new["Edad"],
    stat    = "density",
    kde     = True,
    line_kws= {'linewidth': 1},
    color   = "firebrick",
    alpha   = 0.3,
)
ax.set_title('Distribución de edades', fontsize = 10,
                     fontweight = "bold")
ax.set_xlabel("Edad"),ax.set_ylabel("Densidad")
ax.tick_params(labelsize = 7)
plt.show()

fig, ax = plt.subplots(figsize=(6, 6))
sns.histplot(
    data    = diabetes_str,
    stat    = "density",
    color   = "firebrick",
    alpha   = 0.3,
)
ax.set_title('Distribución de diabetes diagnosticada', fontsize = 10,
                     fontweight = "bold")
ax.set_xlabel("Diabetes"), ax.set_ylabel("Densidad")
ax.tick_params(labelsize = 7)
plt.show()

# Calcular límites para cada columna y eliminar outlier
# Mostrar resumen después de eliminar outliers
print(df_new.describe())

# Punto 1: Comparaciones entre grupos
def comparar_grupos(df, variable, grupo, group_labels):
    
    grupo1 = df[df[grupo] == group_labels[0]][variable].dropna()
    grupo2 = df[df[grupo] == group_labels[1]][variable].dropna()
    
    # Prueba de normalidad
    normal_grupo1 = shapiro(grupo1).pvalue > 0.05
    normal_grupo2 = shapiro(grupo2).pvalue > 0.05
    
    if normal_grupo1 and normal_grupo2:
        # Prueba t
        stat, p = ttest_ind(grupo1, grupo2)
        test = "T-test"
    else:
        # Prueba U de Mann-Whitney
        stat, p = mannwhitneyu(grupo1, grupo2)
        test = "Mann-Whitney U"

    print(f"Comparación entre {group_labels[0]} y {group_labels[1]} para {variable}:")
    print(f"Test: {test}, Estadístico: {stat:.2f}, p-valor: {p:.4f}")
    print()

# Ejemplo: Comparar triglicéridos entre diabéticos y no diabéticos
comparar_grupos(df_new, 'Valor.trig', 'Diabetes_n', [0, 1])
comparar_grupos(df_new, 'Valor.ac_urico', 'Diabetes_n', [0, 1])
comparar_grupos(df_new, 'Valor.creat', 'Diabetes_n', [0, 1])
comparar_grupos(df_new, 'Valor.ggt', 'Diabetes_n', [0, 1])
comparar_grupos(df_new, 'Edad', 'Diabetes_n', [0, 1])

df.describe()

# Punto 2: Correlaciones
def calcular_correlacion(df, var1, var2):
    
    data1 = df[var1].dropna()
    data2 = df[var2].dropna()
    
    if shapiro(data1).pvalue > 0.05 and shapiro(data2).pvalue > 0.05:
        # Pearson
        coef, p = pearsonr(data1, data2)
        metodo = "Pearson"
    else:
        # Spearman
        coef, p = spearmanr(data1, data2)
        metodo = "Spearman"

    print(f"Correlación ({metodo}) entre {var1} y {var2}: Coeficiente: {coef:.2f}, p-valor: {p:.4f}")
    print()

# Ejemplo: Correlación entre edad y colesterol LDL
calcular_correlacion(df_new, 'Valor.trig', 'Diabetes_n')
calcular_correlacion(df_new, 'Valor.ac_urico', 'Diabetes_n')
calcular_correlacion(df_new, 'Valor.creat', 'Diabetes_n')
calcular_correlacion(df_new, 'Valor.ggt', 'Diabetes_n')
calcular_correlacion(df_new, 'Edad', 'Diabetes_n')


# Punto 6: Segmentación por rangos de edad

def analizar_por_grupos(df, variable, nombre, grupo):
    
    grupos = [grupo for grupo in df[grupo].unique() if not pd.isnull(grupo)]
    data = [df[df[grupo] == g][variable].dropna() for g in grupos]
    
    # Prueba de Kruskal-Wallis
    stat, p = kruskal(*data)
    print(f"Análisis por grupos para {variable} según {grupo}:")
    print(f"Kruskal-Wallis Test: Estadístico: {stat:.2f}, p-valor: {p:.4f}")
    print()

    # Visualización
    sns.boxplot(x=grupo, y=variable, data=df)
    plt.xlabel(f"{nombre}"), plt.ylabel("Grupo de edad")
    plt.title(f'{nombre} por grupo de edad')
    plt.xticks(rotation=45)
    plt.show()

# Crear grupos de edad
bins = [0, 30, 45, 60, 110]
labels = ['Jóvenes', 'Adultos', 'Adultos Mayores', 'Ancianos']
df_new['grupo_edad'] = pd.cut(df_new['Edad'], bins=bins, labels=labels)

df_hombres = df_new[df_new["Sexo"] == 1]
df_mujeres = df_new[df_new["Sexo"] == 2]

analizar_por_grupos(df_new, 'Valor.trig', "Triglicéridos", 'grupo_edad')
analizar_por_grupos(df_new, 'Valor.ac_urico',"Ácido úrico", 'grupo_edad')
analizar_por_grupos(df_new, 'Valor.creat',"Creatinina", 'grupo_edad')
analizar_por_grupos(df_mujeres, 'Valor.ggt',"GGT", 'grupo_edad')
analizar_por_grupos(df_hombres, 'Valor.ggt',"GGT", 'grupo_edad')

# Iterar sobre cada variable para calcular el intervalo de confianza
    
df_new.describe()
    
fig, ax = plt.subplots(figsize=(6, 6))
sns.histplot(
    data    = df_new["grupo_edad"],
    stat    = "density",
    kde     = False,
    color   = "firebrick",
    alpha   = 0.3,
)
ax.set_title('Distribución de edad (por grupos de edad)', fontsize = 10,
                     fontweight = "bold")
ax.set_xlabel("Grupo"), ax.set_ylabel("Densidad")
ax.tick_params(labelsize = 7)
plt.show()

df_diab = df_new[df_new["Diabetes_n"] == 1]
df_nodiab = df_new[df_new["Diabetes_n"] == 0]

plt.figure(figsize=(10, 6))
sns.histplot(data=df_diab, x='grupo_edad', stat='density', discrete=True, kde=False, color='firebrick')

# Personalizar el gráfico
plt.title('Densidad de Diagnósticos positivo de Diabetes por grupo de edad', fontsize=14)
plt.xlabel('Grupo de edad', fontsize=12)
plt.ylabel('Densidad', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data=df_nodiab, x='grupo_edad', stat='density', discrete=True, kde=False, color='firebrick')

# Personalizar el gráfico
plt.title('Densidad de Diagnósticos negativo de Diabetes por grupo de edad', fontsize=14)
plt.xlabel('Grupo de edad', fontsize=12)
plt.ylabel('Densidad', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

age_groups = df_new.groupby('Edad').agg(
    total=('Diabetes_n', 'size'),  # Total de personas por edad
    diabetic=('Diabetes_n', lambda x: (x == 1).sum())  # Personas no diabéticas por edad
).reset_index()

# Calcular proporción de no diabéticos
age_groups['proportion_diabetic'] = age_groups['diabetic'] / age_groups['total']

# Calcular intervalos de confianza al 95%
age_groups['ci_lower'], age_groups['ci_upper'] = zip(*age_groups.apply(
    lambda row: proportion_confint(count=row['diabetic'], nobs=row['total'], alpha=0.05, method='normal'),
    axis=1
))

# Encontrar la edad mínima con al menos un 95% de seguridad en el grupo de no diabéticos
threshold_age = age_groups[age_groups['ci_upper'] <= 0.05]['Edad'].min()

# Mostrar los resultados
print("Edad mínima con un 95% de seguridad en el grupo de diabéticos:", threshold_age)

# Graficar proporciones con intervalos de confianza
plt.figure(figsize=(10, 6))
plt.errorbar(age_groups['Edad'], age_groups['proportion_diabetic'],
             yerr=[age_groups['proportion_diabetic'] - age_groups['ci_lower'],
                   age_groups['ci_upper'] - age_groups['proportion_diabetic']],
             fmt='o', label='Proporción de diabéticos (con IC 95%)')
plt.axhline(0.95, color='red', linestyle='--', label='95% de confianza')
plt.axvline(threshold_age, color='green', linestyle='--', label=f'Umbral: {threshold_age} años')
plt.title('Proporción de diabéticos por edad con IC 95%')
plt.xlabel('Edad')
plt.ylabel('Proporción de diabéticos')
plt.legend()
plt.grid()
plt.show()

df_new.describe()