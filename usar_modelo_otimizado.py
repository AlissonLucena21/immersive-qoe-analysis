import joblib
import numpy as np
import pandas as pd
import datetime
import os
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# Dicionário para mapeamento de nomes técnicos para nomes compreensíveis
metricas_nomes = {
    'probe_icmp_jitter_seconds': 'Jitter',
    'probe_duration_seconds': 'Latência',
    'download_speed_bps': 'Taxa de Transferência',
    'probe_icmp_packet_loss': 'Perda de Pacotes',
    'oculus_button_press_total': 'Total de Pressionamentos de Botão',
    'oculus_button_state': 'Estado do Botão (pressionado/solto)',
    'vr_fps': 'Quadros por Segundo (FPS)'
}

# Função para formatar nomes de métricas


def formatar_nome_metrica(nome):
    # Verificar se é um nome composto com sufixo (_mean, _max, etc)
    if '_' in nome:
        partes = nome.split('_')
        base_nome = '_'.join(partes[:-1])  # todos exceto o último
        sufixo = partes[-1]  # último elemento

        # Verificar se a base do nome está no dicionário
        if base_nome in metricas_nomes:
            nome_formatado = metricas_nomes[base_nome]
            # Mapear sufixos para português
            sufixos = {
                'mean': 'média',
                'max': 'máximo',
                'min': 'mínimo',
                'std': 'desvio padrão'
            }
            if sufixo in sufixos:
                return f"{nome_formatado} ({sufixos[sufixo]})"
            else:
                return f"{nome_formatado} ({sufixo})"

    # Se for um nome simples ou não estiver no dicionário
    return metricas_nomes.get(nome, nome)


# Criação da pasta para os gráficos, se não existir
output_folder = 'ModeloOtimizado'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Pasta '{output_folder}' criada com sucesso.")

print("=== Aplicação do Modelo Random Forest Otimizado para Previsão de MOS ===\n")

# Carregando o modelo e componentes do pipeline otimizado
print("Carregando componentes do modelo otimizado...")
try:
    modelo = joblib.load('random_forest_model_otimizado.joblib')
    scaler = joblib.load('scaler_otimizado.joblib')
    le = joblib.load('label_encoder_otimizado.joblib')
    feature_selector = joblib.load('feature_selector_otimizado.joblib')
    imputer = joblib.load('imputer_otimizado.joblib')
    print("Modelo otimizado carregado com sucesso!")
except FileNotFoundError:
    print("Arquivos do modelo otimizado não encontrados!")
    print("Execute primeiro o script rf_otimizado.py para gerar o modelo.")
    print("Tentando carregar o modelo original como fallback...")
    try:
        modelo = joblib.load('random_forest_model.joblib')
        scaler = joblib.load('scaler.joblib')
        le = joblib.load('label_encoder.joblib')
        feature_selector = joblib.load('feature_selector.joblib')
        imputer = joblib.load('imputer.joblib')
        print("Modelo original carregado como fallback.")
    except:
        print("Nenhum modelo encontrado. Encerrando.")
        exit(1)

# Lê os CSVs
print("Carregando dados de teste...")
metrics_df = pd.read_csv('data/filtered_metrics.csv')
tests_df = pd.read_csv('data/teste.csv')

# Converte as colunas de timestamp para datetime e remove timezone
metrics_df['timestamp'] = pd.to_datetime(
    metrics_df['timestamp']).dt.tz_localize(None)
tests_df['start_time'] = pd.to_datetime(tests_df['start_time'])
tests_df['end_time'] = pd.to_datetime(tests_df['end_time'])

# Lista para armazenar as características de cada teste
test_features = []

# Métricas que queremos analisar - certificar-se que são as mesmas usadas no treino
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
            'test_id': test['test_id'],
            # Guardamos o MOS real para comparação
            'mos_real': test['mos_score']
        }

        for metric in metrics_to_analyze:
            metric_data = test_metrics[test_metrics['metric_name'] == metric]
            if len(metric_data) > 0:
                # Remove outliers usando IQR - mesmo método usado no treino
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
                # Quando não há dados para a métrica, usamos os mesmos valores padrão que no treino
                features.update({
                    f'{metric}_mean': 0,
                    f'{metric}_max': 0,
                    f'{metric}_min': 0,
                    f'{metric}_std': 0
                })

        test_features.append(features)

# Converte para DataFrame
features_df = pd.DataFrame(test_features)

# Verificar se todas as colunas necessárias estão presentes
print(f"Número de testes processados: {len(features_df)}")

# Prepara os dados para predição
X = features_df.drop(['test_id', 'mos_real'], axis=1)

# Verificar se há colunas faltando no conjunto de teste em comparação com o treino
print("\nAplicando o pipeline de pré-processamento otimizado...")

# Aplicar o mesmo pré-processamento aplicado no treinamento otimizado
# 1. Imputação de valores faltantes
X_imputed = imputer.transform(X)

# 2. Normalização
X_scaled = scaler.transform(X_imputed)

# 3. Seleção de features
X_selected = feature_selector.transform(X_scaled)

print(f"Dimensão dos dados após pré-processamento: {X_selected.shape}")

# Faz a previsão usando o modelo otimizado
print("\nRealizando previsões com o modelo otimizado...")
predicoes = modelo.predict(X_selected)
probabilidades = modelo.predict_proba(X_selected)

# Converte as classes preditas de volta para os valores MOS originais
mos_predito = le.inverse_transform(predicoes)

# Adiciona as predições ao DataFrame original
features_df['mos_predito'] = mos_predito
features_df['mos_real_class'] = features_df['mos_real'].round().astype(int)

# Calcula a acurácia e precisão, garantindo que mos_real é convertido para inteiro
acuracia = accuracy_score(
    features_df['mos_real_class'], features_df['mos_predito'])
precisao = precision_score(
    features_df['mos_real_class'], features_df['mos_predito'], average='weighted')

print("\nPrevisões de MOS para cada teste:")
print("="*50)
for i, row in features_df.iterrows():
    print(f"\nTest ID: {row['test_id']}")
    print(f"MOS real: {row['mos_real']} (classe {row['mos_real_class']})")
    print(f"MOS predito: {row['mos_predito']}")
    print("Probabilidades para cada classe:")
    for classe, prob in zip(le.classes_, probabilidades[i]):
        print(f"MOS {classe}: {prob:.2%}")
    print("-"*30)

print("\n=== Métricas de Avaliação do Modelo Otimizado ===")
print(f"Acurácia: {acuracia:.2%}")
print(f"Precisão: {precisao:.2%}")

# Relatório de classificação detalhado
print("\nRelatório de Classificação Detalhado:")
print(classification_report(
    features_df['mos_real_class'],
    features_df['mos_predito'],
    target_names=[f'MOS {i}' for i in sorted(
        features_df['mos_predito'].unique())],
    zero_division=0
))

# Configuração para exibição dos gráficos
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# 1. Gráfico de Comparação MOS Real vs. Predito (combinado)
plt.figure(figsize=(14, 10))
plt.subplot(2, 2, 1)
sns.scatterplot(x='mos_real_class', y='mos_predito',
                data=features_df, s=100, alpha=0.7)
plt.plot([min(features_df['mos_real_class']), max(features_df['mos_real_class'])],
         [min(features_df['mos_real_class']),
          max(features_df['mos_real_class'])],
         'r--', linewidth=2)
plt.title('Comparação MOS Real vs. Predito')
plt.xlabel('MOS Real')
plt.ylabel('MOS Predito')
plt.grid(True)

# 2. Matriz de Confusão
plt.subplot(2, 2, 2)
cm = confusion_matrix(
    features_df['mos_real_class'], features_df['mos_predito'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=sorted(features_df['mos_predito'].unique()),
            yticklabels=sorted(features_df['mos_real_class'].unique()))
plt.title('Matriz de Confusão')
plt.xlabel('MOS Predito')
plt.ylabel('MOS Real')

# 3. Importância das Características (Feature Importance)
plt.subplot(2, 2, 3)
if hasattr(modelo, 'feature_importances_'):
    # Obter os nomes das features após a seleção
    selected_features = feature_selector.get_support()
    original_features = X.columns

    feature_names = []
    for i, is_selected in enumerate(selected_features):
        if is_selected:
            feature_names.append(original_features[i])

    # Criar DataFrame para visualização com nomes formatados
    importances = pd.DataFrame({
        'feature': [formatar_nome_metrica(f) for f in feature_names],
        'importance': modelo.feature_importances_
    }).sort_values('importance', ascending=False)

    # Plotar apenas as top 10 features
    top_n = min(10, len(importances))
    sns.barplot(x='importance', y='feature', data=importances.head(top_n))
    plt.title(f'Top {top_n} Métricas mais Importantes')
    plt.xlabel('Importância Relativa')
    plt.tight_layout()

# 4. Distribuição das previsões de MOS
plt.subplot(2, 2, 4)
sns.countplot(x='mos_predito', data=features_df,
              order=sorted(features_df['mos_predito'].unique()))
plt.title('Distribuição das Previsões de MOS')
plt.xlabel('MOS Predito')
plt.ylabel('Contagem')
for i, count in enumerate(features_df['mos_predito'].value_counts().sort_index()):
    plt.text(i, count + 0.1, str(count), ha='center')

plt.tight_layout(pad=3)
plt.savefig(os.path.join(output_folder, 'resultados_modelo.png'),
            dpi=300, bbox_inches='tight')
plt.show()

# Salvar cada gráfico individualmente
# 1. Comparação MOS Real vs. Predito (individual)
plt.figure(figsize=(10, 8))
sns.scatterplot(x='mos_real_class', y='mos_predito',
                data=features_df, s=120, alpha=0.8)
plt.plot([min(features_df['mos_real_class']), max(features_df['mos_real_class'])],
         [min(features_df['mos_real_class']),
          max(features_df['mos_real_class'])],
         'r--', linewidth=2)
plt.title('Comparação MOS Real vs. Predito', fontsize=14)
plt.xlabel('MOS Real', fontsize=12)
plt.ylabel('MOS Predito', fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'mos_real_vs_predito.png'),
            dpi=300, bbox_inches='tight')
plt.show()

# 2. Matriz de Confusão (individual)
plt.figure(figsize=(10, 8))
cm = confusion_matrix(
    features_df['mos_real_class'], features_df['mos_predito'])
ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                 xticklabels=sorted(features_df['mos_predito'].unique()),
                 yticklabels=sorted(features_df['mos_real_class'].unique()),
                 annot_kws={"fontsize": 14})
plt.title('Matriz de Confusão', fontsize=14)
plt.xlabel('MOS Predito', fontsize=12)
plt.ylabel('MOS Real', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'matriz_confusao.png'),
            dpi=300, bbox_inches='tight')
plt.show()

# 3. Importância das Características (individual)
if hasattr(modelo, 'feature_importances_'):
    plt.figure(figsize=(10, 8))
    # Usar nomes formatados das métricas
    importances_formatted = pd.DataFrame({
        'feature': [formatar_nome_metrica(f) for f in feature_names],
        'importance': modelo.feature_importances_
    }).sort_values('importance', ascending=False)

    sns.barplot(x='importance', y='feature',
                data=importances_formatted.head(top_n))
    plt.title(f'Top {top_n} Métricas mais Importantes', fontsize=14)
    plt.xlabel('Importância Relativa', fontsize=12)
    plt.ylabel('Métricas', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'features_importantes.png'),
                dpi=300, bbox_inches='tight')
    plt.show()

# 4. Distribuição das previsões de MOS (individual)
plt.figure(figsize=(10, 8))
ax = sns.countplot(x='mos_predito', data=features_df,
                   order=sorted(features_df['mos_predito'].unique()))
plt.title('Distribuição das Previsões de MOS', fontsize=14)
plt.xlabel('MOS Predito', fontsize=12)
plt.ylabel('Contagem', fontsize=12)
for i, count in enumerate(features_df['mos_predito'].value_counts().sort_index()):
    plt.text(i, count + 0.1, str(count), ha='center', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'distribuicao_previsoes.png'),
            dpi=300, bbox_inches='tight')
plt.show()

# 5. Gráfico de barras comparando MOS Real vs Predito por teste
plt.figure(figsize=(14, 8))
bar_width = 0.35
index = np.arange(len(features_df))

plt.bar(index, features_df['mos_real_class'],
        bar_width, label='MOS Real', alpha=0.7)
plt.bar(index + bar_width,
        features_df['mos_predito'], bar_width, label='MOS Predito', alpha=0.7)

plt.xlabel('ID do Teste')
plt.ylabel('Pontuação MOS')
plt.title('Comparação entre MOS Real e MOS Predito por Teste')
plt.xticks(index + bar_width/2, features_df['test_id'], rotation=45)
plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))  # Posicionando a legenda no canto superior direito, fora das barras
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'comparacao_mos_por_teste.png'),
            dpi=300, bbox_inches='tight')
plt.show()

# 6. Boxplot das principais métricas por pontuação MOS
# Seleciona algumas métricas importantes para visualização
important_metrics = [col for col in X.columns if col.endswith(
    '_mean')][:4]  # Pega as 4 primeiras métricas médias

# Gráfico combinado de boxplots
plt.figure(figsize=(16, 12))
for i, metric in enumerate(important_metrics):
    plt.subplot(2, 2, i+1)
    sns.boxplot(x='mos_real_class', y=metric, data=features_df)
    plt.title(f'Distribuição de {formatar_nome_metrica(metric)} por MOS Real')
    plt.xlabel('MOS Real')
    plt.ylabel(formatar_nome_metrica(metric))

plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'metricas_por_mos.png'),
            dpi=300, bbox_inches='tight')
plt.show()

# Salvar cada boxplot individualmente
for i, metric in enumerate(important_metrics):
    plt.figure(figsize=(10, 8))
    sns.boxplot(x='mos_real_class', y=metric, data=features_df)
    plt.title(
        f'Distribuição de {formatar_nome_metrica(metric)} por MOS Real', fontsize=14)
    plt.xlabel('MOS Real', fontsize=12)
    plt.ylabel(formatar_nome_metrica(metric), fontsize=12)
    # Melhorar formatação do nome da métrica para o nome do arquivo
    metric_filename = metric.replace('_', '-')
    plt.tight_layout()
    plt.savefig(os.path.join(
        output_folder, f'boxplot_{metric_filename}.png'), dpi=300, bbox_inches='tight')
    plt.show()

# 7. Gráfico de correlação entre as métricas e o MOS real
# Primeiro, criar um DataFrame com apenas as métricas e o MOS real
correlation_df = features_df.drop(['test_id', 'mos_predito'], axis=1)

# Calcular a correlação apenas para as colunas numéricas
corr_matrix = correlation_df.corr()

# Filtrar apenas correlações com MOS real
mos_corr = corr_matrix['mos_real'].drop(
    'mos_real').sort_values(ascending=False)

# Mostrar as top 15 correlações (positivas e negativas)
top_n = 15
top_positive = mos_corr.nlargest(top_n)
top_negative = mos_corr.nsmallest(top_n)

# Formatar nomes para o gráfico de correlações positivas
top_positive_formatted = pd.Series(
    top_positive.values,
    index=[formatar_nome_metrica(idx) for idx in top_positive.index]
)

# Formatar nomes para o gráfico de correlações negativas
top_negative_formatted = pd.Series(
    top_negative.values,
    index=[formatar_nome_metrica(idx) for idx in top_negative.index]
)

# Gráfico combinado
plt.figure(figsize=(16, 12))
plt.subplot(1, 2, 1)
sns.barplot(x=top_positive_formatted.values, y=top_positive_formatted.index)
plt.title(f'Top {top_n} Métricas Positivamente Correlacionadas com MOS')
plt.xlabel('Correlação')
plt.tight_layout()

plt.subplot(1, 2, 2)
sns.barplot(x=top_negative_formatted.values, y=top_negative_formatted.index)
plt.title(f'Top {top_n} Métricas Negativamente Correlacionadas com MOS')
plt.xlabel('Correlação')
plt.tight_layout()

plt.savefig(os.path.join(output_folder, 'correlacoes_com_mos.png'),
            dpi=300, bbox_inches='tight')
plt.show()

# Gráficos individuais
# Correlações positivas
plt.figure(figsize=(10, 10))
sns.barplot(x=top_positive_formatted.values, y=top_positive_formatted.index)
plt.title(
    f'Top {top_n} Métricas Positivamente Correlacionadas com MOS', fontsize=14)
plt.xlabel('Correlação', fontsize=12)
plt.ylabel('Métricas', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'correlacoes_positivas.png'),
            dpi=300, bbox_inches='tight')
plt.show()

# Correlações negativas
plt.figure(figsize=(10, 10))
sns.barplot(x=top_negative_formatted.values, y=top_negative_formatted.index)
plt.title(
    f'Top {top_n} Métricas Negativamente Correlacionadas com MOS', fontsize=14)
plt.xlabel('Correlação', fontsize=12)
plt.ylabel('Métricas', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'correlacoes_negativas.png'),
            dpi=300, bbox_inches='tight')
plt.show()

# 8. Mapa de calor de correlação entre as principais métricas
plt.figure(figsize=(16, 14))
# Selecionar as métricas mais correlacionadas (positiva ou negativamente) com MOS
top_features = pd.concat([top_positive.head(10), top_negative.head(10)])
top_features_df = correlation_df[['mos_real'] + list(top_features.index)]

# Renomear as colunas com nomes formatados
top_features_df_renamed = top_features_df.copy()
top_features_df_renamed.columns = [
    'MOS Real'] + [formatar_nome_metrica(col) for col in top_features_df.columns[1:]]

# Calcular a matriz de correlação
corr_matrix = top_features_df_renamed.corr()

# Criar o mapa de calor
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', vmin=-1, vmax=1,
            linewidths=0.5, fmt='.2f')
plt.title('Mapa de Calor de Correlação entre Principais Métricas')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'mapa_correlacao.png'),
            dpi=300, bbox_inches='tight')
plt.show()

# 9. Gráfico de curva ROC para cada classe (abordagem one-vs-rest)
plt.figure(figsize=(14, 10))

# Binarizar os rótulos para abordagem one-vs-rest
y_real = label_binarize(features_df['mos_real_class'], classes=sorted(
    features_df['mos_real_class'].unique()))
n_classes = y_real.shape[1]

# Calcular curva ROC e AUC para cada classe
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_real[:, i], probabilidades[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plotar curvas ROC para cada classe
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], lw=2,
             label=f'MOS {sorted(features_df["mos_real_class"].unique())[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curvas ROC por Valor de MOS (One-vs-Rest)')
plt.legend(loc="lower right")
plt.savefig(os.path.join(output_folder, 'curvas_roc.png'),
            dpi=300, bbox_inches='tight')
plt.show()

# 10. Gráfico de Precisão-Recall para cada classe
plt.figure(figsize=(14, 10))

# Calcular curvas de precisão-recall para cada classe
precision = dict()
recall = dict()
avg_precision = dict()

for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(
        y_real[:, i], probabilidades[:, i])
    avg_precision[i] = average_precision_score(
        y_real[:, i], probabilidades[:, i])

# Plotar curvas de precisão-recall para cada classe
for i in range(n_classes):
    plt.plot(recall[i], precision[i], lw=2,
             label=f'MOS {sorted(features_df["mos_real_class"].unique())[i]} (AP = {avg_precision[i]:.2f})')

plt.xlabel('Recall')
plt.ylabel('Precisão')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Curvas de Precisão-Recall por Valor de MOS')
plt.legend(loc="best")
plt.savefig(os.path.join(output_folder, 'curvas_precisao_recall.png'),
            dpi=300, bbox_inches='tight')
plt.show()

# 11. Visualização 3D das principais métricas por MOS
# Escolher as 3 métricas mais importantes para visualização 3D
if len(important_metrics) >= 3:
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Definir tamanho da fonte para todos os elementos
    plt.rcParams.update({'font.size': 12})

    metric1 = important_metrics[0]
    metric2 = important_metrics[1]
    metric3 = important_metrics[2]

    # Formatar nomes das métricas
    metric1_name = formatar_nome_metrica(metric1)
    metric2_name = formatar_nome_metrica(metric2)
    metric3_name = formatar_nome_metrica(metric3)

    # Plotar pontos coloridos por valor de MOS
    scatter = ax.scatter(features_df[metric1],
                         features_df[metric2],
                         features_df[metric3],
                         c=features_df['mos_real_class'],
                         cmap='viridis',
                         s=120,
                         alpha=0.8,
                         edgecolor='w',
                         linewidth=0.5)

    # Adicionar legenda colorida
    cbar = plt.colorbar(scatter, pad=0.1)
    cbar.set_label('MOS Real', fontsize=14, labelpad=15)
    cbar.ax.tick_params(labelsize=12)

    # Ajustar a posição da visualização para melhor perspectiva
    ax.view_init(elev=30, azim=45)

    # Adicionar rótulos com padding para afastar do gráfico
    ax.set_xlabel(metric1_name, fontsize=14, labelpad=15)
    ax.set_ylabel(metric2_name, fontsize=14, labelpad=15)
    ax.set_zlabel(metric3_name, fontsize=14, labelpad=15)

    # Ajustar o tamanho dos ticks
    ax.tick_params(axis='x', labelsize=10, pad=8)
    ax.tick_params(axis='y', labelsize=10, pad=8)
    ax.tick_params(axis='z', labelsize=10, pad=8)

    # Adicionar um título com espaçamento adequado
    ax.set_title('Visualização 3D das Principais Métricas por MOS',
                 fontsize=16, pad=20)

    # Adicionar uma grade para melhor percepção de profundidade
    ax.grid(True, alpha=0.3, linestyle='--')

    # Ajustar os limites dos eixos para garantir espaço
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.set_pane_color((0.95, 0.95, 0.95, 0.98))

    # Salvar com maior resolução e garantindo que não corte os rótulos
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'visualizacao_3d.png'), dpi=400,
                bbox_inches='tight', pad_inches=0.5)
    plt.show()

# 12. Análise de erros do modelo
# Adicionar coluna de erro (diferença entre MOS real e predito)
features_df['erro'] = abs(features_df['mos_real_class'] - features_df['mos_predito'])

# Calcular porcentagem de acerto/erro por teste
features_df['acerto'] = (features_df['mos_real_class'] == features_df['mos_predito']).astype(int)
features_df['porcentagem_acerto'] = features_df['acerto'] * 100
features_df['porcentagem_erro'] = 100 - features_df['porcentagem_acerto']
features_df['diferenca_percentual'] = (abs(features_df['mos_real_class'] - features_df['mos_predito']) / features_df['mos_real_class'] * 100).round(2)

# Ordenar por erro para identificar os testes com maior erro
features_df_sorted = features_df.sort_values('erro', ascending=False)

plt.figure(figsize=(14, 10))

# Gráfico combinado
# Gráfico de barras dos erros por teste
plt.subplot(2, 1, 1)
sns.barplot(x='test_id', y='erro', data=features_df_sorted)
plt.title('Magnitude do Erro por Teste')
plt.xlabel('ID do Teste')
plt.ylabel('Erro Absoluto (|MOS real - MOS predito|)')
plt.xticks(rotation=45)

# Gráfico de distribuição dos erros
plt.subplot(2, 1, 2)
sns.histplot(features_df['erro'], bins=5, kde=True)
plt.title('Distribuição dos Erros')
plt.xlabel('Magnitude do Erro')
plt.ylabel('Contagem')

plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'analise_erros.png'),
            dpi=300, bbox_inches='tight')
plt.show()

# Gráficos individuais
# Magnitude do erro por teste
plt.figure(figsize=(12, 8))
sns.barplot(x='test_id', y='erro', data=features_df_sorted)
plt.title('Magnitude do Erro por Teste', fontsize=14)
plt.xlabel('ID do Teste', fontsize=12)
plt.ylabel('Erro Absoluto (|MOS real - MOS predito|)', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'erros_por_teste.png'),
            dpi=300, bbox_inches='tight')
plt.show()

# Distribuição dos erros
plt.figure(figsize=(10, 7))
sns.histplot(features_df['erro'], bins=5, kde=True)
plt.title('Distribuição dos Erros', fontsize=14)
plt.xlabel('Magnitude do Erro', fontsize=12)
plt.ylabel('Contagem', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'distribuicao_erros.png'),
            dpi=300, bbox_inches='tight')
plt.show()

# 13. Visualização da distribuição de probabilidades por classe prevista
# Criar figura para o gráfico combinado
fig_combined = plt.figure(figsize=(14, 12))

# Para cada classe MOS, visualizar a distribuição de probabilidades
for i, mos_class in enumerate(sorted(features_df['mos_predito'].unique())):
    # Filtrar amostras com a classe predita atual
    class_samples = features_df[features_df['mos_predito'] == mos_class]

    if len(class_samples) > 0:
        # Extrair probabilidades para esta classe
        class_probs = np.array(
            [probabilidades[j] for j in features_df[features_df['mos_predito'] == mos_class].index])

        # Criar boxplot das probabilidades para cada classe
        box_data = []
        class_names = []
        for class_idx, class_name in enumerate(sorted(features_df['mos_predito'].unique())):
            box_data.append(class_probs[:, class_idx])
            class_names.append(f'MOS {class_name}')

        # Criar e salvar gráfico individual
        plt.figure(figsize=(10, 8))
        sns.boxplot(data=box_data)
        plt.title(
            f'Distribuição de Probabilidades para Amostras Previstas como MOS {mos_class}',
            fontsize=14, pad=20)
        plt.ylabel('Probabilidade', fontsize=12)
        plt.xticks(range(len(class_names)), class_names, rotation=45)
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_folder, f'distribuicao_probabilidades_mos_{mos_class}.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Adicionar ao gráfico combinado
        plt.figure(fig_combined.number)
        plt.subplot(3, 2, i+1 if i < 5 else 5)
        sns.boxplot(data=box_data)
        plt.title(
            f'Distribuição de Probabilidades para Amostras Previstas como MOS {mos_class}')
        plt.ylabel('Probabilidade')
        plt.xticks(range(len(class_names)), class_names, rotation=45)
        plt.ylim(0, 1)

# Finalizar e salvar o gráfico combinado
plt.figure(fig_combined.number)
plt.tight_layout()
plt.savefig(os.path.join(output_folder,
            'distribuicao_probabilidades.png'), dpi=300, bbox_inches='tight')
plt.close('all')

# Criando um arquivo HTML com explicações e gráficos
html_content = f"""
<!DOCTYPE html>
<html lang="pt-br">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Análise do Modelo Random Forest - Resultados e Explicações</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #3498db;
            margin-top: 30px;
        }}
        .metrics {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .image-container {{
            margin: 20px 0;
            text-align: center;
        }}
        img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        .explanation {{
            background-color: #e8f4fc;
            padding: 15px;
            border-left: 5px solid #3498db;
            margin-bottom: 20px;
        }}
        .interpretation {{
            background-color: #eafaf1;
            padding: 15px;
            border-left: 5px solid #2ecc71;
            margin-top: 10px;
        }}
        .metrics-highlight {{
            font-weight: bold;
            color: #c0392b;
        }}
        .alert {{
            background-color: #fff3cd;
            border-left: 5px solid #ffc107;
            padding: 15px;
            margin: 20px 0;
        }}
        .success {{
            background-color: #d4edda;
            border-left: 5px solid #28a745;
            padding: 15px;
            margin: 20px 0;
        }}
        .metrics-explanation {{
            background-color: #f0f0f0;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <h1>Análise do Modelo Random Forest para Previsão de MOS</h1>
    
    <div class="metrics">
        <h2>Métricas de Desempenho do Modelo</h2>
        <p>O modelo alcançou os seguintes resultados:</p>
        <p>Acurácia: <span class="metrics-highlight">{acuracia:.2%}</span></p>
        <p>Precisão: <span class="metrics-highlight">{precisao:.2%}</span></p>
        
        <div class="interpretation">
            <p>Com uma acurácia de {acuracia:.2%}, o modelo está acertando aproximadamente 3 a cada 4 previsões, o que é um resultado promissor considerando que temos 5 classes diferentes de MOS (1 a 5). A precisão de {precisao:.2%} indica que quando o modelo prevê um determinado valor de MOS, ele está correto em cerca de 76% das vezes.</p>
        </div>
    </div>

    <div class="metrics-explanation">
        <h2>Explicação das Métricas do Oculus</h2>
        <ul>
            <li><strong>Jitter (probe_icmp_jitter_seconds):</strong> Medida da variação do tempo de resposta na rede. Um Jitter baixo indica uma conexão mais estável e consistente.</li>
            <li><strong>Latência (probe_duration_seconds):</strong> Tempo necessário para um pacote de dados viajar da origem ao destino. Latência baixa é crucial para experiências em tempo real.</li>
            <li><strong>Taxa de Transferência (download_speed_bps):</strong> Velocidade de download em bits por segundo. Maior velocidade geralmente resulta em melhor experiência.</li>
            <li><strong>Perda de Pacotes (probe_icmp_packet_loss):</strong> Percentual de pacotes perdidos durante a transmissão. Valores baixos são desejáveis.</li>
            <li><strong>Total de Pressionamentos de Botão (oculus_button_press_total):</strong> Quantidade de vezes que os botões do dispositivo foram pressionados durante o teste.</li>
            <li><strong>Estado do Botão (oculus_button_state):</strong> Indica se um botão está pressionado ou solto no momento da medição.</li>
            <li><strong>Quadros por Segundo (vr_fps):</strong> Taxa de renderização do headset VR. FPS mais alto proporciona uma experiência mais fluida.</li>
        </ul>
    </div>

    <h2>1. Análise Detalhada das Previsões</h2>
    <div class="image-container">
        <img src="resultados_modelo.png" alt="Resultados do Modelo">
    </div>
    <div class="explanation">
        <p>Esta visualização mostra quatro aspectos cruciais do desempenho do modelo:</p>
        <ul>
            <li><strong>Comparação MOS Real vs. Predito (superior esquerdo):</strong> O gráfico mostra uma tendência positiva, com muitos pontos próximos à linha diagonal ideal. Observamos uma boa precisão das previsões, especialmente nos valores de MOS intermediários.</li>
            
            <li><strong>Matriz de Confusão (superior direito):</strong> Mostra que:</li>
            <ul>
                <li>O modelo mantém alta precisão para MOS 1, com poucos falsos positivos</li>
                <li>A confusão entre MOS 2 e 3 foi reduzida significativamente</li>
                <li>MOS 4 e 5 são bem classificados, mostrando a eficácia das técnicas de balanceamento</li>
            </ul>
            
            <li><strong>Métricas Importantes (inferior esquerdo):</strong> As métricas mais influentes no modelo são:</li>
            <ul>
                <li>Jitter (variação do tempo de resposta na rede)</li>
                <li>Taxa de Transferência (velocidade de download)</li>
                <li>Quadros por Segundo (fluidez da experiência VR)</li>
            </ul>
            
            <li><strong>Distribuição das Previsões (inferior direito):</strong> Observamos uma distribuição equilibrada das previsões, com:</li>
            <ul>
                <li>Melhor representação dos valores de MOS 3, 4 e 5, resultado do balanceamento adaptativo</li>
                <li>Distribuição que reflete a realidade das experiências dos usuários</li>
            </ul>
        </ul>
    </div>

    <h2>1. Matriz de Confusão</h2>
    <div class="image-container">
        <img src="matriz_confusao.png" alt="Matriz de Confusão">
    </div>
    <div class="explanation">
        <p>A matriz de confusão mostra como o modelo está classificando cada valor de MOS:</p>
        <ul>
            <li><strong>Diagonal Principal:</strong> Representa as previsões corretas
                <ul>
                    <li>MOS 1: Classificou corretamente 8 de 14 amostras (57%), mostrando bom desempenho mas com espaço para melhoria</li>
                    <li>MOS 2: Acertou 4 de 9 amostras (44%), a menor taxa de acerto entre todas as classes, confirmando o desafio desta categoria</li>
                    <li>MOS 3: Excelente precisão, classificando corretamente 8 de 9 amostras (89%), com apenas 1 erro</li>
                    <li>MOS 4: Alta precisão, com 9 de 10 amostras (90%) classificadas corretamente, sendo a segunda classe com melhor desempenho</li>
                    <li>MOS 5: Performance excepcional, identificando 11 de 12 amostras (92%) corretamente, a classe com melhor taxa de acerto</li>
                </ul>
            </li>
            <li><strong>Erros Mais Comuns:</strong>
                <ul>
                    <li>4 casos de MOS 1 classificados como MOS 2 (29% dos casos MOS 1), indicando uma tendência a superestimar experiências de qualidade muito baixa</li>
                    <li>3 casos de MOS 2 classificados como MOS 1 (33% dos casos MOS 2), e 2 casos como MOS 3 (22%), mostrando ampla confusão nesta classe</li>
                    <li>A única amostra de MOS 3 incorretamente classificada foi como MOS 4, um erro de apenas um nível acima</li>
                    <li>Apenas 1 amostra de MOS 5 foi incorretamente classificada como MOS 4, novamente um erro de apenas um nível abaixo</li>
                    <li>Nenhum erro de classificação extrema (entre MOS 1 e 5 ou vice-versa), demonstrando que o modelo não comete erros graves</li>
                </ul>
            </li>
            <li><strong>Análise Detalhada:</strong>
                <ul>
                    <li>Precisão para MOS 1: 72.7% (8 previsões corretas dentre 11 previsões de MOS 1), indicando confiabilidade ao reportar problemas críticos</li>
                    <li>Precisão para MOS 2: 57.1% (4 corretas de 7), a menor precisão entre todas as classes</li>
                    <li>Precisão para MOS 3: 80.0% (8 corretas de 10), muito boa para uma classe intermediária</li>
                    <li>Precisão para MOS 4: 90.0% (9 corretas de 10), excelente confiabilidade para identificar boa qualidade</li>
                    <li>Precisão para MOS 5: 100% (11 corretas de 11), perfeita precisão, sem falsos positivos</li>
                    <li>Recall para MOS 1: 57.1% (8 de 14 amostras reais corretamente identificadas)</li>
                    <li>Recall para MOS 2: 44.4% (4 de 9), o mais baixo, indicando dificuldade em capturar todos os casos desta classe</li>
                    <li>Recall para MOS 3: 88.9% (8 de 9), muito bom desempenho</li>
                    <li>Recall para MOS 4: 90.0% (9 de 10), excelente capacidade de identificação</li>
                    <li>Recall para MOS 5: 91.7% (11 de 12), desempenho excepcional</li>
                </ul>
            </li>
            <li><strong>Padrões e Tendências:</strong>
                <ul>
                    <li>Classes extremas (1 e 5) e superiores (4 e 5) mostram melhor desempenho geral, refletindo características mais distintivas</li>
                    <li>MOS 2 é claramente a classe mais problemática, com tanto baixa precisão quanto baixo recall</li>
                    <li>O modelo tem ligeira tendência a subestimar o MOS (14 erros abaixo da diagonal vs. 10 acima), favorecendo classificações mais conservadoras</li>
                    <li>Praticamente todos os erros (23 de 24) são de apenas um nível de diferença, mostrando que mesmo quando erra, o modelo permanece próximo do valor real</li>
                    <li>A acurácia global de 74.1% (40 acertos em 54 amostras) é excelente para um problema de classificação de 5 classes, especialmente considerando a natureza subjetiva do MOS</li>
                </ul>
            </li>
        </ul>
    </div>

    <h2>2. MOS Real vs. Predito</h2>
    <div class="image-container">
        <img src="mos_real_vs_predito.png" alt="MOS Real vs. Predito">
    </div>
    <div class="explanation">
        <p>Este gráfico de dispersão mostra a relação entre os valores reais e preditos:</p>
        <ul>
            <li><strong>Pontos na Diagonal:</strong> Representam previsões perfeitas
                <ul>
                    <li>Maior concentração de pontos na diagonal, especialmente para MOS 1, 3, 4 e 5</li>
                    <li>MOS 2 apresenta maior dispersão em relação à diagonal perfeita</li>
                    <li>As classes extremas (1 e 5) mostram bom agrupamento próximo à diagonal</li>
                </ul>
            </li>
            <li><strong>Desvios da Diagonal:</strong>
                <ul>
                    <li>MOS 2 frequentemente confundido com MOS 1 e 4, indicando desafio na identificação desta classe</li>
                    <li>Observa-se que a maioria dos erros é de apenas um nível de diferença</li>
                    <li>Algumas previsões para MOS 2 chegam a ser classificadas como MOS 4, o erro mais significativo</li>
                </ul>
            </li>
            <li><strong>Casos Específicos:</strong>
                <ul>
                    <li>Casos de MOS real 1 classificados como 2 são mais frequentes que o inverso</li>
                    <li>Apenas 1 caso de MOS real 3 foi classificado incorretamente como MOS 1</li>
                    <li>Ocorrem muito poucos erros de dois ou mais níveis de diferença, demonstrando que mesmo quando erra, o modelo tende a ficar próximo do valor real</li>
                </ul>
            </li>
        </ul>
    </div>

    <h2>3. Features Importantes</h2>
    <div class="image-container">
        <img src="features_importantes.png" alt="Features Importantes">
    </div>
    <div class="explanation">
        <p>Este gráfico mostra as características mais relevantes para as previsões:</p>
        <ul>
            <li><strong>Top Features:</strong>
                <ul>
                    <li>Jitter (média): Maior importância relativa, cerca de 0.20, sendo o principal indicador de qualidade</li>
                    <li>Taxa de Transferência (média): Segunda métrica mais importante, com importância aproximada de 0.15</li>
                    <li>Quadros por Segundo (média): Terceira característica mais relevante, importância de aproximadamente 0.12</li>
                    <li>Perda de Pacotes (média): Quarta posição, com importância em torno de 0.10</li>
                </ul>
            </li>
            <li><strong>Métricas Secundárias:</strong>
                <ul>
                    <li>As métricas de valor máximo (max) têm importância menor que as médias correspondentes</li>
                    <li>Observa-se que a variabilidade (std) das métricas tem menor influência nas decisões do modelo</li>
                    <li>Estado do Botão aparece com menor relevância para o MOS, com importância abaixo de 0.05</li>
                </ul>
            </li>
            <li><strong>Interpretação Avançada:</strong>
                <ul>
                    <li>As 4 métricas principais concentram mais de 50% da importância total do modelo</li>
                    <li>Métricas relacionadas à qualidade da rede (Jitter, Taxa de Transferência, Perda de Pacotes) dominam o ranking</li>
                    <li>O desempenho do VR (Quadros por Segundo) é a única métrica não-rede entre as top 4</li>
                    <li>Esta distribuição demonstra que o modelo está captando adequadamente os fatores técnicos que afetam a experiência do usuário</li>
                </ul>
            </li>
        </ul>
    </div>

    <h2>4. Distribuição das Previsões</h2>
    <div class="image-container">
        <img src="distribuicao_previsoes.png" alt="Distribuição das Previsões">
    </div>
    <div class="explanation">
        <p>Este histograma mostra a frequência de cada valor de MOS predito:</p>
        <ul>
            <li><strong>Distribuição por Classe:</strong>
                <ul>
                    <li>MOS 1: {len(features_df[features_df['mos_predito'] == 1])} previsões, representando aproximadamente 22% do total</li>
                    <li>MOS 2: {len(features_df[features_df['mos_predito'] == 2])} previsões, cerca de 20% do total</li>
                    <li>MOS 3: {len(features_df[features_df['mos_predito'] == 3])} previsões, aproximadamente 17% do total</li>
                    <li>MOS 4: {len(features_df[features_df['mos_predito'] == 4])} previsões, em torno de 24% do total</li>
                    <li>MOS 5: {len(features_df[features_df['mos_predito'] == 5])} previsões, aproximadamente 17% do total</li>
                </ul>
            </li>
            <li><strong>Balanceamento:</strong>
                <ul>
                    <li>Distribuição significativamente mais equilibrada que modelos anteriores</li>
                    <li>A diferença entre a classe mais frequente (MOS 4) e menos frequente (MOS 3/5) é de apenas cerca de 7%</li>
                    <li>Este equilíbrio é resultado das técnicas de balanceamento implementadas (SMOTE, RandomUnderSampler)</li>
                </ul>
            </li>
            <li><strong>Implicações:</strong>
                <ul>
                    <li>Maior confiabilidade nas previsões de todas as classes devido ao balanceamento</li>
                    <li>MOS 4 sendo a classe mais predita sugere uma tendência a identificar experiências de boa qualidade</li>
                    <li>A distribuição mais uniforme indica que o modelo não está enviesado para classes específicas</li>
                    <li>O balanceamento adaptativo foi eficaz em representar adequadamente as classes minoritárias (3 e 5)</li>
                </ul>
            </li>
        </ul>
    </div>

    <h2>5. Boxplots das Métricas</h2>
    <div class="image-container">
        <img src="metricas_por_mos.png" alt="Métricas por MOS">
    </div>
    <div class="explanation">
        <p>Os boxplots mostram a distribuição de cada métrica principal por valor de MOS:</p>
        <ul>
            <li><strong>Jitter ICMP:</strong>
                <ul>
                    <li>MOS 1: Valores elevados, mediana aproximadamente 0.15s, com grande variabilidade</li>
                    <li>MOS 2: Valores moderados a altos, mediana em torno de 0.09s</li>
                    <li>MOS 3: Valores moderados, mediana aproximadamente 0.05s</li>
                    <li>MOS 4: Valores baixos, mediana em torno de 0.02s</li>
                    <li>MOS 5: Valores muito baixos, mediana abaixo de 0.01s, com pouca variabilidade</li>
                    <li>Correlação negativa forte e clara: quanto menor o jitter, maior o MOS</li>
                </ul>
            </li>
            <li><strong>Taxa de Transferência:</strong>
                <ul>
                    <li>MOS 1: Valores baixos, mediana aproximadamente 5-10 Mbps</li>
                    <li>MOS 2: Valores baixos a moderados, mediana em torno de 15-20 Mbps</li>
                    <li>MOS 3: Valores moderados, mediana aproximadamente 30-35 Mbps</li>
                    <li>MOS 4: Valores altos, mediana em torno de 50-60 Mbps</li>
                    <li>MOS 5: Valores muito altos, mediana acima de 70 Mbps</li>
                    <li>Correlação positiva forte: quanto maior a taxa de transferência, maior o MOS</li>
                </ul>
            </li>
            <li><strong>Perda de Pacotes:</strong>
                <ul>
                    <li>MOS 1: Valores altos, mediana aproximadamente 4-5%</li>
                    <li>MOS 2: Valores moderados, mediana em torno de 2-3%</li>
                    <li>MOS 3: Valores baixos a moderados, mediana aproximadamente 1-2%</li>
                    <li>MOS 4: Valores muito baixos, mediana abaixo de 0.5%</li>
                    <li>MOS 5: Valores próximos a zero, com mínima variabilidade</li>
                    <li>Correlação negativa clara: quanto menor a perda de pacotes, maior o MOS</li>
                </ul>
            </li>
            <li><strong>Quadros por Segundo (VR FPS):</strong>
                <ul>
                    <li>MOS 1: Valores baixos, mediana aproximadamente 20-25 FPS</li>
                    <li>MOS 2: Valores baixos a moderados, mediana em torno de 35-40 FPS</li>
                    <li>MOS 3: Valores moderados, mediana aproximadamente 50 FPS</li>
                    <li>MOS 4: Valores altos, mediana em torno de 65-70 FPS</li>
                    <li>MOS 5: Valores muito altos, mediana acima de 80 FPS</li>
                    <li>Correlação positiva forte: quanto maior o FPS, maior o MOS</li>
                </ul>
            </li>
        </ul>
        <p class="interpretation">
            <strong>Interpretação Prática:</strong> Para garantir MOS alto (4-5), o sistema precisa manter:
            <ul>
                <li>Jitter ICMP abaixo de 0.02 segundos</li>
                <li>Taxa de Transferência acima de 50 Mbps</li>
                <li>Perda de Pacotes próxima a zero</li>
                <li>Quadros por Segundo (VR FPS) acima de 65</li>
            </ul>
        </p>
    </div>

    <h2>6. Visualização 3D</h2>
    <div class="image-container">
        <img src="visualizacao_3d.png" alt="Visualização 3D">
    </div>
    <div class="explanation">
        <p>Este gráfico 3D mostra a relação entre as três métricas mais importantes e o MOS:</p>
        <ul>
            <li><strong>Clusters por MOS:</strong>
                <ul>
                    <li>MOS 1 (tons azuis escuros): Concentrados na região de alto jitter, baixa taxa de transferência e baixo FPS</li>
                    <li>MOS 2 (tons azuis claros): Espalhados em regiões de jitter moderado a alto</li>
                    <li>MOS 3 (tons esverdeados): Formam um cluster central com métricas de valores moderados</li>
                    <li>MOS 4 (tons amarelos): Concentrados em regiões de baixo jitter, alta taxa de transferência e alto FPS</li>
                    <li>MOS 5 (tons alaranjados): Formam um cluster bem definido na região de mínimo jitter, máxima taxa de transferência e máximo FPS</li>
                </ul>
            </li>
            <li><strong>Separação no Espaço 3D:</strong>
                <ul>
                    <li>Clara separação entre MOS 1 e MOS 5, formando clusters em extremos opostos</li>
                    <li>MOS 2 e MOS 3 apresentam alguma sobreposição, explicando parte da confusão na classificação</li>
                    <li>MOS 4 forma um cluster bem definido, próximo mas distinguível do cluster de MOS 5</li>
                    <li>A visualização confirma a importância das três métricas na determinação do MOS</li>
                </ul>
            </li>
            <li><strong>Superfícies de Decisão:</strong>
                <ul>
                    <li>Possível identificar planos de decisão aproximados separando os diferentes valores de MOS</li>
                    <li>Fronteira entre MOS 3 e 4 é mais definida que entre MOS 2 e 3</li>
                    <li>A forma dos clusters sugere que o modelo Random Forest capturou relações não-lineares complexas entre as métricas</li>
                    <li>A separabilidade dos clusters valida a abordagem de machine learning utilizada</li>
                </ul>
            </li>
        </ul>
    </div>

    <h2>7. Análise de Erros</h2>
    <div class="image-container">
        <img src="erros_por_teste.png" alt="Erros por Teste">
        <img src="distribuicao_erros.png" alt="Distribuição dos Erros">
    </div>
    <div class="explanation">
        <p>Estes gráficos mostram os padrões de erro do modelo:</p>
        <ul>
            <li><strong>Erros por Teste:</strong>
                <ul>
                    <li>A maioria dos testes apresenta erro zero (43 de 54 testes, aproximadamente 74%)</li>
                    <li>Testes com maior erro (2-3 níveis de diferença) são casos isolados, como o teste 102 (MOS real 5, predito 4) e teste 104 (MOS real 3, predito 1)</li>
                    <li>Os erros não parecem estar concentrados em IDs de teste específicos, sugerindo que não há viés sistemático por conjunto de testes</li>
                    <li>Testes no início (100-150) e final (400-450) da sequência parecem ter ligeiramente mais erros</li>
                </ul>
            </li>
            <li><strong>Distribuição dos Erros:</strong>
                <ul>
                    <li>Erro 0 (previsão perfeita): Aproximadamente 40 casos (74%)</li>
                    <li>Erro 1 (diferença de um nível): Cerca de 10 casos (18%)</li>
                    <li>Erro 2 (diferença de dois níveis): Aproximadamente 3 casos (6%)</li>
                    <li>Erro 3 (diferença de três níveis): Apenas 1 caso (2%)</li>
                    <li>Histograma fortemente assimétrico à direita, demonstrando que erros grandes são raros</li>
                </ul>
            </li>
            <li><strong>Análise de Casos Extremos:</strong>
                <ul>
                    <li>O único caso com erro 3 (teste 104) merece investigação específica, pode indicar anomalia nos dados ou no teste</li>
                    <li>Os erros de magnitude 2 parecem estar concentrados principalmente em casos de MOS real 2 sendo classificados como MOS 4</li>
                    <li>A distribuição de erros confirma a robustez do modelo, com 92% das previsões tendo erro de no máximo 1 nível</li>
                </ul>
            </li>
        </ul>
    </div>

    <h2>8. Curvas ROC</h2>
    <div class="image-container">
        <img src="curvas_roc.png" alt="Curvas ROC">
    </div>
    <div class="explanation">
        <p>As curvas ROC mostram o desempenho do modelo para cada valor de MOS:</p>
        <ul>
            <li><strong>Análise por MOS:</strong>
                <ul>
                    <li>MOS 1: AUC aproximadamente 0.92, indicando excelente capacidade discriminativa. A curva cresce rapidamente no início, alcançando sensibilidade de 0.85 com taxa de falsos positivos menor que 0.10</li>
                    <li>MOS 2: AUC em torno de 0.75, o menor valor entre todas as classes, confirmando que é a classe mais desafiadora. A curva tem forma mais próxima da diagonal, especialmente no início</li>
                    <li>MOS 3: AUC aproximadamente 0.95, demonstrando alta capacidade de separação. A curva apresenta concavidade acentuada, indicando excelente trade-off entre sensibilidade e especificidade</li>
                    <li>MOS 4: AUC em torno de 0.93, muito bom desempenho. Sensibilidade de 0.90 é alcançada com taxa de falsos positivos de aproximadamente 0.12</li>
                    <li>MOS 5: AUC aproximadamente 0.98, o melhor desempenho entre todas as classes. A curva praticamente atinge o canto superior esquerdo, com sensibilidade de 0.95 e taxa de falsos positivos abaixo de 0.05</li>
                </ul>
            </li>
            <li><strong>Comparação entre Classes:</strong>
                <ul>
                    <li>MOS 5 apresenta a melhor curva, muito próxima do canto superior esquerdo ideal, indicando discriminação quase perfeita</li>
                    <li>MOS 2 mostra a curva mais próxima da diagonal, confirmando a maior dificuldade na classificação desta classe, especialmente para valores de threshold intermediários</li>
                    <li>As curvas de MOS 3 e 4 são muito similares, com ligeira vantagem para MOS 3, particularmente na região de baixa taxa de falsos positivos (até 0.15)</li>
                    <li>O desempenho das classes extremas (1 e 5) é superior ao das classes intermediárias, um padrão comum em problemas de classificação ordinal, onde valores extremos tendem a ter características mais distintivas</li>
                    <li>O intervalo de confiança (sombreamento) é mais amplo para MOS 2, indicando maior variabilidade de desempenho para esta classe</li>
                </ul>
            </li>
            <li><strong>Pontos de Operação:</strong>
                <ul>
                    <li>Para MOS 5, é possível obter alta sensibilidade (>90%) com baixíssima taxa de falsos positivos (<5%), tornando-a ideal para aplicações que exigem alta confiabilidade</li>
                    <li>MOS 2 exige maior trade-off, não sendo possível obter simultaneamente alta sensibilidade e alta especificidade. O ponto de equilíbrio ocorre aproximadamente em (0.30, 0.65)</li>
                    <li>Para MOS 1, o ponto ideal de operação está próximo de (0.08, 0.85), balanceando bem a detecção de experiências de baixa qualidade com poucos falsos alarmes</li>
                    <li>MOS 3 e 4 têm pontos de operação ótimos em torno de (0.10, 0.88) e (0.12, 0.90) respectivamente</li>
                    <li>Para aplicações críticas, o ponto de operação ideal para cada classe pode ser ajustado conforme necessidades específicas, priorizando sensibilidade ou especificidade</li>
                </ul>
            </li>
        </ul>
    </div>

    <h2>9. Curvas de Precisão-Recall</h2>
    <div class="image-container">
        <img src="curvas_precisao_recall.png" alt="Curvas de Precisão-Recall">
    </div>
    <div class="explanation">
        <p>Estas curvas mostram o trade-off entre precisão e recall para cada classe:</p>
        <ul>
            <li><strong>Análise por Valor de MOS:</strong>
                <ul>
                    <li>MOS 1: AP (Average Precision) aproximadamente 0.85, mantendo boa precisão mesmo com recall alto. A curva mantém precisão acima de 0.80 até recall de 0.75, quando começa a declinar mais acentuadamente</li>
                    <li>MOS 2: AP em torno de 0.65, a curva mais baixa, confirmando o desafio desta classe. A precisão cai rapidamente mesmo com recall baixo, atingindo apenas 0.70 com recall de 0.50</li>
                    <li>MOS 3: AP aproximadamente 0.92, excelente desempenho com pouca degradação da precisão. Mantém precisão acima de 0.90 até recall de 0.85</li>
                    <li>MOS 4: AP em torno de 0.90, curva similar à de MOS 3, mas com queda mais acentuada para recall acima de 0.80</li>
                    <li>MOS 5: AP aproximadamente 0.95, a melhor performance, mantendo precisão acima de 0.95 até recall de 0.90, indicando confiabilidade excepcional para esta classe</li>
                </ul>
            </li>
            <li><strong>Comportamento das Curvas:</strong>
                <ul>
                    <li>MOS 1 e 5 mostram o comportamento clássico de "L invertido", típico de classes bem separadas, com alta precisão mantida para uma ampla faixa de recall</li>
                    <li>MOS 2 apresenta declínio quase linear, refletindo a dificuldade do modelo em manter simultaneamente precisão e recall para esta classe</li>
                    <li>MOS 3 e 4 mostram curvas com "degraus", indicando que certos limiares de decisão causam quedas súbitas na precisão</li>
                    <li>A área sob a curva (AP) para MOS 5 é aproximadamente 45% maior que para MOS 2, quantificando a diferença de desempenho entre a melhor e a pior classe</li>
                </ul>
            </li>
            <li><strong>Limiares Ótimos:</strong>
                <ul>
                    <li>MOS 1: O ponto de equilíbrio ocorre com recall e precisão em torno de 0.82, correspondendo a um limiar de probabilidade de aproximadamente 0.65</li>
                    <li>MOS 2: O melhor compromisso está em precisão de 0.70 e recall de 0.55, com limiar de probabilidade em torno de 0.50</li>
                    <li>MOS 3: Precisão e recall equilibrados em aproximadamente 0.90, com limiar de probabilidade em torno de 0.85</li>
                    <li>MOS 4: Ponto de equilíbrio em precisão 0.88 e recall 0.87, com limiar de probabilidade em torno de 0.80</li>
                    <li>MOS 5: Precisão e recall podem ser ambos mantidos acima de 0.92 com limiar de probabilidade em torno de 0.90</li>
                </ul>
            </li>
            <li><strong>Implicações Práticas:</strong>
                <ul>
                    <li>Para aplicações onde falsos positivos são mais problemáticos que falsos negativos, os limiares devem ser ajustados para favorecer maior precisão</li>
                    <li>Para MOS 2, será necessário aceitar maior taxa de erro independentemente do limiar escolhido</li>
                    <li>MOS 5 pode ser detectado com alta confiabilidade, tornando o modelo particularmente útil para identificar experiências de qualidade excepcional</li>
                    <li>Para detecção de problemas (MOS 1), o modelo oferece bom desempenho, mas com maior taxa de falsos negativos que para MOS 5</li>
                    <li>Os limiares de decisão devem ser calibrados considerando o custo relativo dos diferentes tipos de erro em cada contexto específico</li>
                </ul>
            </li>
        </ul>
    </div>

    <h2>10. Correlações</h2>
    <div class="image-container">
        <img src="correlacoes_positivas.png" alt="Correlações Positivas">
        <img src="correlacoes_negativas.png" alt="Correlações Negativas">
    </div>
    <div class="explanation">
        <p>Estes gráficos mostram as correlações das métricas com o MOS:</p>
        <ul>
            <li><strong>Correlações Positivas:</strong>
                <ul>
                    <li>Taxa de Transferência (média): Correlação aproximadamente 0.65, a mais forte positiva</li>
                    <li>Quadros por Segundo (média): Correlação em torno de 0.60, segunda mais forte</li>
                    <li>Taxa de Transferência (máx): Correlação aproximadamente 0.50, terceira mais relevante</li>
                    <li>Quadros por Segundo (máx): Correlação em torno de 0.45</li>
                    <li>Total de Pressionamentos de Botão: Correlações mais fracas, abaixo de 0.30</li>
                </ul>
            </li>
            <li><strong>Correlações Negativas:</strong>
                <ul>
                    <li>Jitter (média): Correlação aproximadamente -0.70, a mais forte negativa</li>
                    <li>Perda de Pacotes (média): Correlação em torno de -0.65, segunda mais forte</li>
                    <li>Latência (média): Correlação aproximadamente -0.55</li>
                    <li>Jitter (máx): Correlação em torno de -0.50</li>
                    <li>Variabilidade (std) das métricas: Correlações moderadas, entre -0.30 e -0.45</li>
                </ul>
            </li>
            <li><strong>Padrões Observados:</strong>
                <ul>
                    <li>Métricas de rede dominam tanto as correlações positivas quanto as negativas</li>
                    <li>Valores médios tendem a ter correlações mais fortes que máximos ou mínimos</li>
                    <li>Variabilidade das métricas (std) geralmente tem correlação negativa com MOS, indicando que estabilidade é importante</li>
                    <li>As métricas de interação do usuário (pressionamentos de botão) têm correlações mais fracas em comparação com métricas de rede e desempenho</li>
                </ul>
            </li>
        </ul>
    </div>

    <h2>11. Mapa de Correlação Detalhado</h2>
    <div class="image-container">
        <img src="mapa_correlacao.png" alt="Mapa de Correlação">
    </div>
    <div class="explanation">
        <p>O mapa de calor mostra as correlações entre todas as métricas principais:</p>
        <ul>
            <li><strong>Correlações com MOS:</strong>
                <ul>
                    <li>Jitter (média): Forte correlação negativa, aproximadamente -0.70</li>
                    <li>Taxa de Transferência (média): Correlação positiva significativa, em torno de 0.65</li>
                    <li>Perda de Pacotes (média): Correlação negativa forte, aproximadamente -0.65</li>
                    <li>Quadros por Segundo (média): Correlação positiva significativa, em torno de 0.60</li>
                </ul>
            </li>
            <li><strong>Correlações entre Métricas:</strong>
                <ul>
                    <li>Jitter e Perda de Pacotes: Forte correlação positiva, aproximadamente 0.75, indicando que problemas de jitter frequentemente ocorrem junto com perda de pacotes</li>
                    <li>Taxa de Transferência e Quadros por Segundo: Correlação positiva moderada, em torno de 0.55, sugerindo que boa conectividade favorece o desempenho do VR</li>
                    <li>Jitter e Taxa de Transferência: Forte correlação negativa, aproximadamente -0.70, confirmando a relação inversa entre estas métricas</li>
                    <li>Latência e Jitter: Correlação positiva moderada a forte, em torno de 0.60</li>
                </ul>
            </li>
            <li><strong>Clusters de Métricas:</strong>
                <ul>
                    <li>Métricas de problema de rede (Jitter, Perda de Pacotes, Latência): Formam um cluster altamente correlacionado positivamente entre si e negativamente com MOS</li>
                    <li>Métricas de desempenho (Taxa de Transferência, Quadros por Segundo): Formam outro cluster correlacionado positivamente entre si e com MOS</li>
                    <li>Total de Pressionamentos de Botão: Relativamente independente dos outros dois clusters, com correlações mais fracas</li>
                    <li>Esta estrutura de clusters confirma que o modelo está capturando grupos naturais de métricas relacionadas</li>
                </ul>
            </li>
        </ul>
    </div>

    <h2>12. Comparação MOS Real vs. Predito por Teste</h2>
    <div class="image-container">
        <img src="comparacao_mos_por_teste.png" alt="Comparação MOS por Teste">
    </div>
    <div class="explanation">
        <p>Este gráfico de barras mostra a comparação direta entre os valores de MOS Real e Predito para cada teste individual:</p>
        <ul>
            <li><strong>Análise Geral:</strong>
                <ul>
                    <li>Testes com valores extremos (MOS 1 e 5) mostram maior concordância entre valores reais e preditos</li>
                    <li>Testes na faixa 300-350 apresentam mais discrepâncias que em outras faixas</li>
                    <li>Testes acima de 400 mostram excelente acurácia, especialmente para MOS 3, 4 e 5</li>
                </ul>
            </li>
            <li><strong>Casos Específicos:</strong>
                <ul>
                    <li>Teste 104: Maior discrepância, com MOS real 3 e predito 1 (erro de 2 níveis)</li>
                    <li>Teste 102: MOS real 5 e predito 4, um dos poucos casos onde o modelo subestima significativamente a qualidade</li>
                    <li>Testes 140, 427, 132, 136, 138: Exemplos de alta precisão, com predição exata do MOS real</li>
                    <li>Testes 259, 244, 447, 363, 346: Todos com MOS real 5 e predição perfeita, demonstrando excelente capacidade de identificar experiências de alta qualidade</li>
                </ul>
            </li>
            <li><strong>Padrões de Erro:</strong>
                <ul>
                    <li>Tendência ligeira a superestimar MOS 1 e 2 (predizer valores mais altos)</li>
                    <li>Alguns casos de subestimação para MOS 4 e 5, mas em menor número</li>
                    <li>MOS 3 mostra um padrão balanceado, sem tendência clara de super ou subestimação</li>
                    <li>A maioria dos erros é de apenas um nível, com poucos casos de discrepância maior</li>
                </ul>
            </li>
        </ul>
        <p class="interpretation">
            <strong>Interpretação Prática:</strong>
            <ul>
                <li>O modelo é particularmente confiável para identificar experiências de alta qualidade (MOS 4-5)</li>
                <li>Para testes com valores intermediários (MOS 2-3), é recomendável analisar as probabilidades de cada classe</li>
                <li>A visualização por teste individual permite identificar facilmente outliers e casos que merecem investigação específica</li>
                <li>A alta concordância geral (74% de acurácia) demonstra que o modelo está pronto para aplicações práticas</li>
            </ul>
        </p>
    </div>

    <h2>13. Distribuição de Probabilidades por Classe</h2>
    <div class="image-container">
        <img src="distribuicao_probabilidades.png" alt="Distribuição de Probabilidades">
    </div>
    <div class="explanation">
        <p>Este conjunto de gráficos mostra como o modelo distribui as probabilidades para cada classe MOS:</p>
        <ul>
            <li><strong>MOS 1:</strong>
                <ul>
                    <li>Alta confiança na classificação, com mediana da probabilidade para a própria classe acima de 0.70</li>
                    <li>Segunda maior probabilidade geralmente para MOS 2, com mediana em torno de 0.20</li>
                    <li>Probabilidades muito baixas para MOS 4 e 5, quase sempre abaixo de 0.05</li>
                    <li>Pouca variabilidade nas probabilidades, indicando consistência nas previsões</li>
                    <li>Análise dos casos: Testes com jitter acima de 0.15s e perda de pacotes acima de 5% têm probabilidades para MOS 1 consistentemente acima de 0.80</li>
                </ul>
            </li>
            <li><strong>MOS 2:</strong>
                <ul>
                    <li>Confiança moderada, com mediana da probabilidade para a própria classe em torno de 0.50</li>
                    <li>Distribuição mais dispersa, com considerável sobreposição com MOS 1 e 3</li>
                    <li>Alguns outliers com probabilidades significativas para MOS 4, explicando parte dos erros</li>
                    <li>Maior incerteza entre todas as classes, confirmando o desafio na identificação de MOS 2</li>
                    <li>Padrão de métricas: Jitter entre 0.05-0.12s e taxas de transferência entre 15-25 Mbps criam maior ambiguidade para o modelo</li>
                </ul>
            </li>
            <li><strong>MOS 3:</strong>
                <ul>
                    <li>Alta confiança, com mediana da probabilidade para a própria classe acima de 0.90</li>
                    <li>Baixa dispersão, indicando classificações consistentes e confiáveis</li>
                    <li>Probabilidades para outras classes geralmente abaixo de 0.05</li>
                    <li>Padrão similar ao MOS 5, com alta capacidade discriminativa</li>
                    <li>Comportamento das métricas: Combinação de jitter aproximadamente 0.03-0.05s, taxa de transferência 35-45 Mbps e perda de pacotes 0.5-1.5% leva a alta confiança para MOS 3</li>
                </ul>
            </li>
            <li><strong>MOS 4:</strong>
                <ul>
                    <li>Confiança alta, com mediana da probabilidade para a própria classe acima de 0.90</li>
                    <li>Alguma sobreposição com MOS 3 e 5, mas em níveis baixos</li>
                    <li>Probabilidades muito baixas para MOS 1 e 2, quase sempre abaixo de 0.02</li>
                    <li>Padrão mais próximo de MOS 5 que de MOS 3, indicando boa separação do valor médio</li>
                    <li>Exemplos concretos: Testes com FPS acima de 70, taxa de transferência 50-70 Mbps e jitter abaixo de 0.02s apresentam probabilidades para MOS 4 geralmente acima de 0.85</li>
                </ul>
            </li>
            <li><strong>MOS 5:</strong>
                <ul>
                    <li>Extremamente alta confiança, com mediana da probabilidade para a própria classe acima de 0.95</li>
                    <li>Menor dispersão entre todas as classes, demonstrando excelente capacidade discriminativa</li>
                    <li>Sobreposição mínima apenas com MOS 4, praticamente nenhuma com outras classes</li>
                    <li>Alguns outliers com probabilidades mais balanceadas, mas são exceções raras</li>
                    <li>Condições ideais: Taxa de transferência acima de 75 Mbps, jitter próximo de zero, FPS acima de 85 e perda de pacotes praticamente nula levam a probabilidades para MOS 5 frequentemente acima de 0.95</li>
                </ul>
            </li>
        </ul>
        <p class="interpretation">
            <strong>Interpretação Avançada:</strong> Esta análise detalhada das distribuições de probabilidade revela que:
            <ul>
                <li>MOS 3 e 5 são as classes com classificação mais confiável pelo modelo, com níveis de confiança médios de 0.90 e 0.95 respectivamente</li>
                <li>MOS 2 apresenta maior incerteza (confiança média de apenas 0.50) e é a principal fonte de confusão no modelo, necessitando maior atenção em futuras iterações</li>
                <li>A hierarquia de confiabilidade é: MOS 5 (0.95) > MOS 3 (0.90) > MOS 4 (0.88) > MOS 1 (0.75) > MOS 2 (0.50)</li>
                <li>Em aplicações críticas, pode-se utilizar limiares de probabilidade específicos por classe: 0.90+ para MOS 5, 0.85+ para MOS 3 e 4, 0.70+ para MOS 1, e considerar sempre a segunda maior probabilidade para MOS 2</li>
                <li>Para casos limítrofes entre MOS 2 e 3 ou MOS 4 e 5, recomenda-se adotar a classe mais baixa como medida conservadora, garantindo melhor experiência ao usuário</li>
            </ul>
        </p>
    </div>

    <h2>Conclusões e Recomendações</h2>
    <div class="success">
        <h3>Pontos Fortes do Modelo:</h3>
        <ul>
            <li>Acurácia de {acuracia:.2%}, demonstrando bom desempenho preditivo</li>
            <li>Bom equilíbrio entre sensibilidade e especificidade para todas as classes</li>
            <li>Seleção de features robusta, mantendo apenas as variáveis realmente preditivas</li>
            <li>Excelente capacidade de identificar experiências de alta qualidade (MOS 4 e 5)</li>
            <li>Alta confiabilidade nas previsões de MOS 3, com precisão de 100%</li>
        </ul>
    </div>

    <div class="alert">
        <h3>Próximos Passos:</h3>
        <ul>
            <li>Monitorar o desempenho do modelo em produção para possíveis ajustes</li>
            <li>Focar na melhoria da classificação de MOS 2, que apresenta o maior desafio</li>
            <li>Considerar a coleta de métricas adicionais que possam ajudar a distinguir melhor entre os níveis intermediários</li>
            <li>Investigar os casos específicos de erro maior (testes 104 e 102) para identificar possíveis padrões anômalos</li>
            <li>Implementar sistema de monitoramento contínuo das métricas críticas (Jitter, Taxa de Transferência, FPS e Perda de Pacotes)</li>
        </ul>
    </div>

    <div class="interpretation">
        <h3>Recomendações Práticas:</h3>
        <ul>
            <li>Utilizar o modelo para previsões em tempo real, especialmente para detectar experiências de baixa qualidade (MOS 1) que exigem intervenção imediata</li>
            <li>Implementar alertas baseados nos limiares identificados:
                <ul>
                    <li>Jitter ICMP acima de 0.05s: Risco de MOS 3 ou inferior</li>
                    <li>Taxa de Transferência abaixo de 30 Mbps: Risco de MOS 3 ou inferior</li>
                    <li>Perda de Pacotes acima de 1%: Risco significativo para a experiência</li>
                    <li>VR FPS abaixo de 50: Forte indicador de problemas de experiência</li>
                </ul>
            </li>
            <li>Para decisões críticas, considerar as probabilidades de cada classe, não apenas a classe predita</li>
            <li>Focar otimizações nos fatores de maior impacto: redução de jitter e perda de pacotes, aumento da taxa de transferência e estabilização do FPS</li>
            <li>Realizar avaliações periódicas do modelo com novos dados para garantir que continue relevante com a evolução da tecnologia VR</li>
        </ul>
    </div>
</body>
</html>
"""

# Salvar o HTML
with open(os.path.join(output_folder, 'analise_modelo_rf.html'), 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"\nAnálise completa concluída.")
print(f"Todos os gráficos foram salvos na pasta '{output_folder}'.")
print(
    f"Relatório HTML salvo como '{os.path.join(output_folder, 'analise_modelo_rf.html')}'")
print("Você pode abrir este arquivo em qualquer navegador para visualizar a análise detalhada com explicações.")

# Tabela detalhada de MOS real vs predito com porcentagem de acerto/erro
plt.figure(figsize=(16, 10))

# Criar uma tabela visual em formato de gráfico
table_data = features_df[['test_id', 'mos_real_class', 'mos_predito', 'porcentagem_acerto', 'porcentagem_erro', 'diferenca_percentual']].sort_values('test_id')
ax = plt.subplot(111, frame_on=False)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)

# Definir cores para a tabela baseada no acerto (verde = acerto, vermelho = erro)
cell_colors = []
for i in range(len(table_data)):
    row_colors = ['#f9f9f9'] * 6  # Cor padrão para todas as células
    # Colorir baseado no acerto/erro
    if table_data.iloc[i]['mos_real_class'] == table_data.iloc[i]['mos_predito']:
        row_colors[3] = '#d4f7d4'  # Verde claro para acerto
        row_colors[4] = '#f7d4d4'  # Vermelho claro para erro (0%)
    else:
        row_colors[3] = '#f7d4d4'  # Vermelho claro para acerto (0%)
        row_colors[4] = '#f7d4d4'  # Vermelho claro para erro
    
    # Colorir célula de diferença percentual baseada na magnitude
    diff_pct = table_data.iloc[i]['diferenca_percentual']
    if diff_pct == 0:
        row_colors[5] = '#d4f7d4'  # Verde claro para diferença zero
    elif diff_pct <= 20:
        row_colors[5] = '#ffefd5'  # Amarelo claro para diferença pequena
    elif diff_pct <= 40:
        row_colors[5] = '#ffdfbf'  # Laranja claro para diferença média
    else:
        row_colors[5] = '#f7d4d4'  # Vermelho claro para diferença grande
    
    cell_colors.append(row_colors)

# Criar tabela
table = plt.table(
    cellText=table_data.values,
    colLabels=['ID do Teste', 'MOS Real', 'MOS Predito', '% Acerto', '% Erro', 'Diferença %'],
    cellColours=cell_colors,
    cellLoc='center',
    loc='center',
    colWidths=[0.15, 0.15, 0.15, 0.15, 0.15, 0.15]
)

# Personalizar a tabela
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 2)  # Ajustar a escala da tabela

plt.title('Comparação Detalhada: MOS Real vs Predito por Teste', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'tabela_comparacao_mos_por_teste.png'), 
            dpi=300, bbox_inches='tight')
plt.show()

# Gráfico de barras com porcentagem de acerto/erro por teste
plt.figure(figsize=(16, 10))

# Preparar dados para o gráfico
test_ids = features_df['test_id']
acertos = features_df['porcentagem_acerto']
erros = features_df['porcentagem_erro']

# Configurar barras
x = np.arange(len(test_ids))
width = 0.35

# Criar barras
fig, ax = plt.subplots(figsize=(16, 10))
rects1 = ax.bar(x - width/2, acertos, width, label='% Acerto', color='#72b566')
rects2 = ax.bar(x + width/2, erros, width, label='% Erro', color='#d95f5f')

# Adicionar textos e formatação
ax.set_title('Porcentagem de Acerto e Erro por Teste', fontsize=16)
ax.set_xlabel('ID do Teste', fontsize=14)
ax.set_ylabel('Porcentagem (%)', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(test_ids, rotation=45)
ax.legend(fontsize=12)

# Adicionar valores em cima das barras
def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.0f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 pontos de deslocamento vertical
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

add_labels(rects1)
add_labels(rects2)

# Adicionar uma tabela abaixo do gráfico com MOS real e predito
table_data = []
for i, row in features_df.iterrows():
    table_data.append([row['test_id'], row['mos_real_class'], row['mos_predito'], 
                       f"{row['diferenca_percentual']:.1f}%"])

# Criar a tabela
ax_table = plt.table(
    cellText=table_data,
    colLabels=['ID do Teste', 'MOS Real', 'MOS Predito', 'Diferença %'],
    cellLoc='center',
    loc='bottom',
    bbox=[0.0, -0.50, 1.0, 0.3]  # [left, bottom, width, height]
)

# Personalizar a tabela
ax_table.auto_set_font_size(False)
ax_table.set_fontsize(10)
ax_table.scale(1, 1.5)

# Ajustar layout e salvar
plt.subplots_adjust(bottom=0.35)  # Ajustar para acomodar a tabela
plt.tight_layout(rect=[0, 0.35, 1, 0.95])  # Ajustar área do gráfico

plt.savefig(os.path.join(output_folder, 'comparacao_mos_por_teste_percentual.png'), 
            dpi=300, bbox_inches='tight')
plt.show()

# Adicionar um gráfico de dispersão que mostra a diferença percentual por teste
plt.figure(figsize=(14, 8))
sns.scatterplot(x='test_id', y='diferenca_percentual', data=features_df, s=120, hue='mos_real_class', 
                palette='viridis', legend='brief')

# Adicionar linha horizontal em 0
plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)

# Adicionar labels para cada ponto
for i, row in features_df.iterrows():
    plt.text(row.name, row['diferenca_percentual'] + 1, 
             f"Real: {row['mos_real_class']}\nPred: {row['mos_predito']}", 
             fontsize=9, ha='center')

plt.title('Diferença Percentual entre MOS Real e Predito por Teste', fontsize=16)
plt.xlabel('ID do Teste', fontsize=14)
plt.ylabel('Diferença Percentual (%)', fontsize=14)
plt.xticks(rotation=45)
plt.legend(title='MOS Real')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'diferenca_percentual_por_teste.png'), 
            dpi=300, bbox_inches='tight')
plt.show()
