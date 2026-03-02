# %%
# ============================== IMPORT ==============================
# Manipulation des données : numpy and pandas
import numpy as np
import pandas as pd

# File system manangement
import os

# Visualisation : matplotlib and seaborn
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Réglage affichage des datasets
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 125)

# %%
# ============================= DATA LOAD =============================
print(os.listdir("../datas/"))

# %%
# Home Credit columns description
HomeCredit_columns_description = pd.read_csv(
    "../datas/HomeCredit_columns_description.csv",
    encoding="ISO-8859-1"
)
print("Home Credit columns description shape: ", HomeCredit_columns_description.shape)
HomeCredit_columns_description.head()

# %%
# Training data
app_train = pd.read_csv(
    "../datas/application_train.csv",
    encoding="ISO-8859-1"
    )
print("Training data shape: ", app_train.shape)
app_train.head()

# %%
# Test Data
app_test = pd.read_csv(
    "../datas/application_test.csv",
    encoding="ISO-8859-1"
    )
print("Test data shape: ", app_test.shape)
app_test.head()

# %%
# Bureau Balance
bureau_balance = pd.read_csv(
    "../datas/bureau_balance.csv",
    encoding="ISO-8859-1")
print("Bureau Balance shape: ", bureau_balance.shape)
bureau_balance.head()

# %%
# Bureau
bureau = pd.read_csv(
    "../datas/bureau.csv",
    encoding="ISO-8859-1"
    )
print("Bureau Balance shape: ", bureau.shape)
bureau.head()

# %%
# Credit Card Balance
credit_card_balance = pd.read_csv(
    "../datas/credit_card_balance.csv",
    encoding="ISO-8859-1"
    )
print("Credit Card Balance shape: ", credit_card_balance.shape)
credit_card_balance.head()

# %%
# Installements payments
installments_payments = pd.read_csv(
    "../datas/installments_payments.csv",
    encoding="ISO-8859-1"
    )
print("Installements payments shape: ", installments_payments.shape)
installments_payments.head()

# %%
# POS CASH Balance
POS_CASH_balance = pd.read_csv(
    "../datas/POS_CASH_balance.csv",
    encoding="ISO-8859-1"
    )
print("POS CASH Balance shape: ", POS_CASH_balance.shape)
POS_CASH_balance.head()

# %%
# Previous application
previous_application = pd.read_csv(
    "../datas/previous_application.csv",
    encoding="ISO-8859-1"
    )
print("Previous application shape: ", previous_application.shape)
previous_application.head()

# %%
# Sample Submission
sample_submission = pd.read_csv(
    "../datas/sample_submission.csv",
    encoding="ISO-8859-1"
    )
print("Sample Submission shape: ", sample_submission.shape)
sample_submission.head()

# %%
# ============================= EDA functions ==============================
def check_duplicates(df, column_name):
    """
    Vérifie les doublons dans une colonne spécifique et retourne les lignes concernées.
    """
    # Vérification si la colonne existe
    if column_name not in df.columns:
        print(f"Erreur : La colonne '{column_name}' est absente du dataset.")
        return None

    # Identification des doublons (keep=False permet de voir toutes les occurrences)
    duplicates = df[df.duplicated(subset=[column_name], keep=False)]
    
    num_duplicates = duplicates.shape[0]
    
    if num_duplicates > 0:
        print(f"🚩 Attention : {num_duplicates} lignes en doublon détectées pour la colonne '{column_name}'.")
        # On trie par la colonne pour faciliter la comparaison visuelle
        return duplicates.sort_values(by=column_name)
    else:
        print(f"✅ Aucun doublon détecté pour la colonne '{column_name}'.")
        return pd.DataFrame() # Retourne un DF vide si pas de doublon


def analyze_dataset_structure(df, dataset_name="Dataset"):
    """
    Présente un résumé complet du dataset : types, manquants et pourcentages.
    """
    # Calcul des métriques
    data_types = df.dtypes
    missing_count = df.isnull().sum()
    missing_percentage = (missing_count / len(df)) * 100
    
    # Création du tableau de synthèse
    analysis_df = pd.DataFrame({
        'Colonne': df.columns,
        'Type': data_types,
        'Valeurs Manquantes': missing_count,
        'Pourcentage (%)': missing_percentage.round(2)
    }).reset_index(drop=True)
    
    print(f"--- Analyse du dataset : {dataset_name} ---")
    print(f"Nombre total de lignes : {len(df)}")
    print(f"Nombre total de colonnes : {df.shape[1]}")
    
    # On trie par pourcentage de valeurs manquantes pour identifier les problèmes
    return analysis_df.sort_values(by='Pourcentage (%)', ascending=False)


def plot_target_distribution(df, target_col='TARGET'):
    """
    Analyse et affiche la distribution de la variable cible.
    Utile pour quantifier le déséquilibre des classes.
    """
    counts = df[target_col].value_counts()
    percentages = df[target_col].value_counts(normalize=True) * 100
    
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(x=target_col, data=df, palette='viridis')
    plt.title(f'Distribution de la Variable Cible ({target_col})')
    plt.xlabel('Classe (0: Remboursé, 1: Défaillant)')
    plt.ylabel('Nombre de clients')
    
    # Ajout des pourcentages sur les barres
    for i, p in enumerate(ax.patches):
        ax.annotate(f'{percentages.iloc[i]:.1f}%', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', fontsize=11, color='black', xytext=(0, 5),
                    textcoords='offset points')
    plt.show()
    
    print(f"Nombre de clients par classe :\n{counts}")


def get_correlations_with_target(df, target_col='TARGET', n_top=15):
    """
    Calcule et affiche les variables les plus corrélées avec la cible.
    Aide à identifier les features les plus discriminantes.
    """
    # On ne garde que les colonnes numériques
    numeric_df = df.select_dtypes(include=[np.number])
    correlations = numeric_df.corr()[target_col].sort_values()
    
    print(f"--- Top {n_top} Corrélations Positives ---")
    print(correlations.tail(n_top))
    print(f"\n--- Top {n_top} Corrélations Négatives ---")
    print(correlations.head(n_top))
    
    return correlations


def plot_feature_distribution_by_target(df, feature, target_col='TARGET'):
    """
    Affiche une courbe de densité (KDE) pour comparer la distribution
    d'une variable entre les clients sains (0) et en défaut (1).
    """
    plt.figure(figsize=(10, 6))
    sns.kdeplot(df.loc[df[target_col] == 0, feature], label='Cible == 0 (Sain)', shade=True)
    sns.kdeplot(df.loc[df[target_col] == 1, feature], label='Cible == 1 (Défaut)', shade=True)
    
    plt.title(f'Distribution de {feature} selon la solvabilité')
    plt.xlabel(feature)
    plt.ylabel('Densité')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


def detect_anomalies(df, feature):
    """
    Identifie les valeurs suspectes ou aberrantes pour une colonne donnée
    en comparant les extrêmes aux statistiques centrales.
    """
    stats = df[feature].describe()
    print(f"--- Analyse des anomalies pour : {feature} ---")
    print(f"Moyenne : {stats['mean']:.2f} | Médiane : {stats['50%']:.2f}")
    print(f"Minimum : {stats['min']:.2f} | Maximum : {stats['max']:.2f}")
    
    # Exemple de règle métier ou statistique (Z-score simplifié ou IQR)
    q1 = stats['25%']
    q3 = stats['75%']
    iqr = q3 - q1
    outlier_bound = q3 + 3 * iqr # On cherche les valeurs très extrêmes
    
    outliers = df[df[feature] > outlier_bound]
    if not outliers.empty:
        print(f"🚩 Alerte : {len(outliers)} valeurs potentielles au-delà du seuil statistique ({outlier_bound:.2f}).")
    else:
        print("✅ Pas d'anomalies majeures détectées par la méthode IQR.")
    
    return outliers
# %%
# ========================= EDA : Training data =========================
print("--- 1. VÉRIFICATION DES DOUBLONS ---")
df_doublons = check_duplicates(app_train, 'SK_ID_CURR')
# %%
print("\n--- 2. STRUCTURE ET MANQUANTS ---")
resume_structure = analyze_dataset_structure(app_train, "Application Train")
# On affiche le top 50 des colonnes les plus vides
print(resume_structure.head(50))
# %%
print("\n--- 3. DISTRIBUTION DE LA CIBLE ---")
plot_target_distribution(app_train, target_col='TARGET')
# %%
print("\n--- 4. CORRÉLATIONS AVEC LA CIBLE ---")
corrs = get_correlations_with_target(app_train, target_col='TARGET')
# %%
print("\n--- 5. VISUALISATION DES FEATURES ---")
plot_feature_distribution_by_target(app_train, 'EXT_SOURCE_3')
# %%
print("\n--- 6. DÉTECTION D'ANOMALIES ---")
anomalies_emploi = detect_anomalies(app_train, 'DAYS_EMPLOYED')
# %%
