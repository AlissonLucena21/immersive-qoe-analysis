from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from scipy import stats
from collections import Counter
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

print("=== TREINAMENTO OTIMIZADO DO MODELO RANDOM FOREST ===\n")

# Lê os CSVs
print("Carregando datasets...")
metrics_df = pd.read_csv('data/filtered_metrics.csv')
tests_df = pd.read_csv('data/treinamento.csv')

# Converte as colunas de timestamp para datetime e remove timezone
metrics_df['timestamp'] = pd.to_datetime(
    metrics_df['timestamp']).dt.tz_localize(None)
tests_df['start_time'] = pd.to_datetime(tests_df['start_time'])
tests_df['end_time'] = pd.to_datetime(tests_df['end_time'])

# Lista para armazenar as características de cada teste
test_features = []

# Métricas que queremos analisar
metrics_to_analyze = [
    'probe_icmp_jitter_seconds',
    'probe_duration_seconds',
    'download_speed_bps',
    'probe_icmp_packet_loss',
    'oculus_button_press_total',
    'oculus_button_state',
    'vr_fps'
]

print("Processando métricas para cada teste...")
# Para cada teste
for _, test in tests_df.iterrows():
    # Filtra as métricas que estão dentro do intervalo de tempo do teste
    test_metrics = metrics_df[
        (metrics_df['timestamp'] >= test['start_time']) &
        (metrics_df['timestamp'] <= test['end_time'])
    ]

    if len(test_metrics) > 0:
        features = {
            'test_id': test['test_id']
        }

        for metric in metrics_to_analyze:
            metric_data = test_metrics[test_metrics['metric_name'] == metric]
            if len(metric_data) > 0:
                # Remove outliers usando IQR
                Q1 = metric_data['value'].quantile(0.25)
                Q3 = metric_data['value'].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                clean_data = metric_data[(metric_data['value'] >= lower_bound) &
                                         (metric_data['value'] <= upper_bound)]

                if len(clean_data) > 0:
                    features.update({
                        f'{metric}_mean': clean_data['value'].mean(),
                        f'{metric}_max': clean_data['value'].max(),
                        f'{metric}_min': clean_data['value'].min(),
                        f'{metric}_std': clean_data['value'].std()
                    })
            else:
                features.update({
                    f'{metric}_mean': 0,
                    f'{metric}_max': 0,
                    f'{metric}_min': 0,
                    f'{metric}_std': 0
                })

        features['mos_score'] = test['mos_score']
        test_features.append(features)

# Converte para DataFrame
features_df = pd.DataFrame(test_features)

# Converte MOS scores para classes inteiras
le = LabelEncoder()
X = features_df.drop(['test_id', 'mos_score'], axis=1)
y = le.fit_transform(features_df['mos_score'].round())

# Pré-processamento avançado
print("\nRealizando pré-processamento robusto dos dados...")

# Trata valores NaN com estratégia robusta
# Usando mediana para ser mais robusto a outliers
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# Normalização robusta
X_imputed_df = pd.DataFrame(X_imputed, columns=X.columns)

# Usando normalização robusta sem remoção de outliers
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed_df)
y_processed = y

# Feature selection mais conservadora
print("\nRealizando seleção de features de forma adaptativa...")
pre_selector = RandomForestClassifier(n_estimators=500,
                                      max_depth=None,
                                      random_state=42,
                                      class_weight='balanced')
pre_selector.fit(X_scaled, y_processed)

# Usar threshold adaptativo com base na distribuição das importâncias
importances = pre_selector.feature_importances_
threshold = max(np.mean(importances) * 0.5, np.median(importances) * 0.7)
print(f"Threshold adaptativo para seleção de features: {threshold:.6f}")

selector = SelectFromModel(pre_selector, prefit=True, threshold=threshold)
X_selected = selector.transform(X_scaled)
selected_feature_mask = selector.get_support()
selected_features = X.columns[selected_feature_mask].tolist()

print(f"\nFeatures selecionadas ({len(selected_features)} de {X.shape[1]}):")
print(selected_features)

# Se muito poucas features foram selecionadas, ajustar o threshold
if len(selected_features) < 10:
    print("Poucas features selecionadas, ajustando threshold...")
    threshold = max(np.mean(importances) * 0.3, np.median(importances) * 0.5)
    selector = SelectFromModel(pre_selector, prefit=True, threshold=threshold)
    X_selected = selector.transform(X_scaled)
    selected_feature_mask = selector.get_support()
    selected_features = X.columns[selected_feature_mask].tolist()
    print(f"Features atualizadas ({len(selected_features)} de {X.shape[1]}):")
    print(selected_features)

# Divide em conjunto de treino e validação
X_train, X_val, y_train, y_val = train_test_split(
    X_selected, y_processed, test_size=0.25, random_state=42, stratify=y_processed)

# Aplica balanceamento de classes mais simples e estável
print("\nAplicando técnicas de balanceamento equilibradas...")

# Verificar a distribuição original
print("Distribuição original das classes:", Counter(y_train))

# Usar SMOTE padrão, que é mais estável e confiável que ADASYN
print("Aplicando SMOTE + RandomUnderSampler para balanceamento equilibrado...")
smote = SMOTE(random_state=42, k_neighbors=5)
rus = RandomUnderSampler(random_state=42)

# Primeiro oversample as classes minoritárias
X_smote, y_smote = smote.fit_resample(X_train, y_train)

# Depois undersample a classe majoritária para evitar muito desbalanceamento
X_train_balanced, y_train_balanced = rus.fit_resample(X_smote, y_smote)

print("Distribuição após balanceamento:", Counter(y_train_balanced))

# Busca por hiperparâmetros ótimos
print("\nRealizando busca por hiperparâmetros ideais...")

# Definir o espaço de busca
param_space = {
    'n_estimators': [500, 1000, 1500],
    'max_depth': [8, 10, 12, 15, None],
    'min_samples_split': [2, 5, 8],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True],
    'class_weight': ['balanced', 'balanced_subsample', None],
    'criterion': ['gini', 'entropy']
}

# Criar o modelo base
rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)

# Criar o objeto de busca aleatória
rf_random = RandomizedSearchCV(
    estimator=rf_base,
    param_distributions=param_space,
    n_iter=20,  # Número de combinações para testar
    cv=5,       # 5-fold cross-validation
    verbose=1,
    random_state=42,
    n_jobs=-1,
    scoring='f1_weighted'  # Otimizar para F1 score
)

# Executar a busca
rf_random.fit(X_train_balanced, y_train_balanced)

# Mostrar os melhores parâmetros
print("\nMelhores hiperparâmetros encontrados:")
for param, value in rf_random.best_params_.items():
    print(f"  {param}: {value}")

# Criar o modelo final com os melhores parâmetros
best_params = rf_random.best_params_
rf_model = RandomForestClassifier(
    **best_params,
    random_state=42,
    n_jobs=-1,
    oob_score=True
)

# Treinar o modelo final
print("\nTreinando modelo final com os melhores parâmetros...")
rf_model.fit(X_train_balanced, y_train_balanced)

# Avaliação no conjunto de validação
print("\nAvaliando modelo no conjunto de validação...")
y_val_pred = rf_model.predict(X_val)

# Calcula métricas
accuracy = accuracy_score(y_val, y_val_pred)
precision = precision_score(
    y_val, y_val_pred, average='weighted', zero_division=0)
f1 = f1_score(y_val, y_val_pred, average='weighted')

print("\nMétricas de Avaliação do Modelo (Validação):")
print(f"Acurácia: {accuracy:.4f}")
print(f"Precisão: {precision:.4f}")
print(f"F1-Score: {f1:.4f}")
if hasattr(rf_model, 'oob_score_'):
    print(f"Out-of-bag Score: {rf_model.oob_score_:.4f}")

print("\nRelatório de Classificação Detalhado:")
print(classification_report(y_val, y_val_pred,
                            target_names=[f'MOS {i}' for i in le.classes_],
                            zero_division=0))

# Validação cruzada no dataset completo
print("\nRealizando validação cruzada...")
cv_scores = cross_val_score(rf_model, X_selected, y_processed,
                            cv=5, scoring='f1_weighted')
print(f"Resultados da Validação Cruzada (F1-Score):")
print(f"Média: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Retreinar o modelo com todo o dataset
print("\nRetreinando o modelo com todo o dataset...")

# Pipeline de balanceamento e treinamento final
balancer = SMOTE(random_state=42, k_neighbors=5)
X_full_balanced, y_full_balanced = balancer.fit_resample(
    X_selected, y_processed)

# Treinamento com todo o dataset
rf_model.fit(X_full_balanced, y_full_balanced)

# Importância das features
feature_importance = pd.DataFrame({
    'feature': selected_features,
    'importance': rf_model.feature_importances_
})
print("\nImportância das Features (Top 10):")
print(feature_importance.sort_values(by='importance', ascending=False).head(10))

# Visualizar a importância das features
plt.figure(figsize=(10, 8))
sns.barplot(x='importance', y='feature', data=feature_importance.sort_values(
    'importance', ascending=False).head(15))
plt.title('Top 15 Features Mais Importantes', fontsize=14)
plt.xlabel('Importância Relativa', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.tight_layout()
plt.savefig('features_importantes_otimizado.png', dpi=300, bbox_inches='tight')
plt.show()

# Salva o modelo e objetos necessários
print("\nSalvando modelo e componentes do pipeline...")
joblib.dump(rf_model, 'random_forest_model_otimizado.joblib')
joblib.dump(scaler, 'scaler_otimizado.joblib')
joblib.dump(imputer, 'imputer_otimizado.joblib')
joblib.dump(le, 'label_encoder_otimizado.joblib')
joblib.dump(selector, 'feature_selector_otimizado.joblib')

print("\nModelo e objetos auxiliares salvos com sucesso!")
print("\nPara usar este modelo otimizado, execute usar_modelo_otimizado.py")
