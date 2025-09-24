# Immersive QoE Analysis

## Um arcabouço para a coleta de métricas de desempenho para a avaliação da qualidade de experiência em ambientes imersivos virtuais e conectados

Este repositório contém o código-fonte, scripts e documentação relacionados ao estudo desenvolvido para **avaliação e previsão da Qualidade de Experiência (QoE)** em **ambientes virtuais imersivos e conectados**, utilizando parâmetros técnicos de rede e de aplicação integrados a dados subjetivos de usuários.  

O projeto inclui a implementação de um **arcabouço baseado em aprendizado de máquina (Random Forest)** para correlacionar métricas objetivas e subjetivas, com foco em monitoramento e análise em tempo real da QoE.

---

### 📌 Objetivos do Projeto
- Investigar a relação entre parâmetros técnicos de rede/aplicação e a experiência percebida pelos usuários.  
- Desenvolver e validar um modelo de inferência de QoE usando **Random Forest**.  
- Integrar dados **objetivos** (rede e aplicação) e **subjetivos** (percepção dos usuários).

---

### 🗂 Estrutura do Repositório


---

### ⚙️ Tecnologias Utilizadas
- **Python 3.10+**
- Bibliotecas principais:
  - `scikit-learn` (Random Forest, métricas)
  - `pandas` e `numpy` (manipulação de dados)
  - `matplotlib` e `seaborn` (visualizações)
  - `scipy` (análises estatísticas)
- Unity
- CoppeliaSim
- Docker
- MetaQuest2
- Prometheus
- Blockbox Exporter
- Speedtest Exporter
- Network Emulator for Windows Toolkit

---

### 🚀 Como Executar
1. Clone este repositório:
   
   ```BASH
   git clone https://github.com/seu-usuario/qoe-immersive-model.git
   cd qoe-immersive-model

2. Instale as dependências.

3. Execute os scripts de treinamento e avaliação:

    ```BASH
    python split_dataset.py
    python random_forest_otimizado.py
    python usar_modelo_otimizado.py

### 📊 Resultados

- Acurácia do modelo: 74,07%

- Precisão do modelo: 76,12%

- Evidências apontam para a relevância da integração de parâmetros objetivos e subjetivos na predição de QoE.

### 📚 Referência Acadêmica

Se este repositório for utilizado em trabalhos acadêmicos, por favor cite:

LUCENA, A. A. de. Immersive QoE Analysis: um arcabouço para a coleta de métricas de desempenho para a avaliação da qualidade de experiência em ambientes imersivos virtuais e conectados. GitHub, 2025. Disponível em: https://github.com/AlissonLucena21/immersive-qoe-analysis.git.

### 👨‍💻 Autor

**Alisson Alves de Lucena** – Aluno do Programa de Pós-Graduação em Ciência da Computação (PPGCC) - Universidade Federal de Campina Grande, Campus Campina Grande - PB.
