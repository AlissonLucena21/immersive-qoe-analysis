import pandas as pd
from sklearn.model_selection import train_test_split

# Lê o arquivo original
df = pd.read_csv('test_metadatas.csv')

# Separa em treino e teste mantendo a proporção de classes (estratificado)
train_df, test_df = train_test_split(
    df,
    test_size=0.2,  # 20% para teste (80% para treino)
    random_state=42,  # Para garantir reprodutibilidade
    stratify=df['mos_score']  # Mantém a mesma proporção de classes
)

# Salva os arquivos
train_df.to_csv('treinamento.csv', index=False)
test_df.to_csv('teste.csv', index=False)

print(f"Arquivo original dividido em:")
print(f"treinamento.csv: {len(train_df)} linhas")
print(f"teste.csv: {len(test_df)} linhas")