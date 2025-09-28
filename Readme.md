# Previsão de Cancelamento de Corridas de Uber por Motoristas

Este projeto de *Machine Learning* tem como objetivo desenvolver um modelo preditivo para identificar a probabilidade de um motorista de Uber cancelar uma corrida **antes que a solicitação seja enviada a ele**. A motivação principal é reduzir o tempo de espera e a frustração dos usuários, otimizando a alocação de corridas para motoristas com maior probabilidade de aceitá-las e completá-las.

**Autor:** Eric Koji Nakirimoto
**Data:** 27/08/2025

## 📋 Sobre o Projeto

O modelo foi treinado com o dataset [Uber Ride Analytics Dataset 2024](https://www.kaggle.com/datasets/yashdevladdha/uber-ride-analytics-dashboard/data), que contém 148.770 registros de reservas de corridas, incluindo informações sobre status da reserva, veículos, locais, avaliações e cancelamentos.

O foco principal é a variável `Cancelled Rides by Driver` (Corridas Canceladas pelo Motorista), buscando entender os padrões que levam a esse evento e criar um sistema capaz de prevê-lo.

## 🚀 Metodologia

O desenvolvimento do projeto seguiu as seguintes etapas:

1.  **Análise Exploratória de Dados (EDA):**

      * Estudo inicial das estatísticas e da distribuição dos dados.
      * Análise detalhada das variáveis, com foco em locais de partida/destino, tipos de veículo e motivos de cancelamento.

2.  **Engenharia de Atributos:**

      * Para enriquecer o modelo, foram criados novos atributos a partir dos dados existentes. O mais impactante foi o cálculo da **taxa de cancelamento histórica por local de partida e de destino** (`cancellation_rate_percent_pickup` e `cancellation_rate_percent_drop`).

3.  **Pré-processamento e Limpeza:**

      * Remoção de dados irrelevantes (ex: cancelamentos por motivo de "cliente doente", que não é previsível).
      * Tratamento de valores nulos.
      * Conversão de dados categóricos (como `Vehicle Type`) em formato numérico através de *One-Hot Encoding*.
      * Transformação de `Date` e `Time` em timestamps numéricos.

4.  **Balanceamento dos Dados:**

      * O dataset original era altamente desbalanceado (\~15% de cancelamentos por motorista). Para evitar que o modelo ficasse viciado em prever apenas a classe majoritária (não cancelamento), foi aplicada a técnica de **undersampling**, criando um dataset final com uma proporção de 50/50 entre as classes.

5.  **Seleção e Treinamento de Modelos:**

      * Diversos algoritmos de classificação foram testados, incluindo:
          * Regressão Logística
          * KNN (K-Nearest Neighbors)
          * CART (Árvore de Decisão)
          * SVM (Support Vector Machine)
          * Random Forest
          * AdaBoost
          * Redes Neurais (com diferentes arquiteturas).
      * Após análises, os atributos `Date` e `Time` foram removidos, pois poderiam introduzir ruído e não generalizar bem para anos futuros (ex: feriados). A remoção melhorou a performance dos modelos.

## 🏆 Modelo Final Escolhido

O modelo que apresentou o melhor equilíbrio entre as métricas de avaliação, especialmente `F1-Score` e `Recall`, foi uma **Rede Neural** com a seguinte arquitetura e configuração:

  * **Estrutura:** 3 camadas (1 de entrada, 1 oculta com 16 neurônios, 1 de saída).
  * **Funções de Ativação:** `ReLU` na camada oculta e `Sigmoid` na camada de saída.
  * **Otimizador:** `Adam`.
  * **Função de Perda:** `binary_crossentropy`.
  * **Dados de Treino:** Normalizados usando `MinMaxScaler`.

Este modelo se destacou por sua alta capacidade de identificar corretamente os cancelamentos (`Recall`) sem sacrificar a precisão geral (`F1-Score`), além de ter um tempo de treinamento mais rápido em comparação com outros modelos de alta performance como SVM.

### Métricas de Performance (Resultados no Conjunto de Teste)

  * **Acurácia:** \~65%
  * **F1-Score (para cancelamentos):** 0.74
  * **Recall (para cancelamentos):** 0.97 (identifica 97% das corridas que seriam canceladas)
  * **Precision (para cancelamentos):** 0.59

> **Observação:** O alto `Recall` é o resultado mais importante para o objetivo do projeto, pois a prioridade é evitar que uma corrida com alta chance de cancelamento seja designada a um motorista.

## 🛠️ Como Executar o Projeto

### Pré-requisitos

Certifique-se de ter as seguintes bibliotecas Python instaladas:

```
pandas
numpy
scikit-learn
tensorflow
keras
joblib
matplotlib
seaborn
```

Você pode instalar todas com o comando:

```bash
pip install pandas numpy scikit-learn tensorflow keras joblib matplotlib seaborn
```

### Executando o Notebook

1.  Clone este repositório.
2.  Abra o arquivo `Projeto_Eric_MVP_ML_&_Analytics.ipynb` em um ambiente como Jupyter Notebook ou Google Colab.
3.  Execute as células em sequência para reproduzir todo o processo de análise e treinamento.

## 🤖 Como Usar o Modelo Treinado

O modelo treinado (`modelo_neural.keras`) e o normalizador (`scaler.pkl`) estão salvos no repositório. Para fazer uma nova previsão, siga o exemplo abaixo:

```python
import pandas as pd
import joblib
import tensorflow as tf

# Carregar o normalizador e o modelo
scaler = joblib.load('scaler.pkl')
model = tf.keras.models.load_model('modelo_neural.keras')

# Criar um DataFrame com os dados da nova corrida
# (os nomes das colunas devem ser os mesmos usados no treinamento)
dados_novos = pd.DataFrame({
    'Avg VTAT': [15.0],                      # Tempo médio para o motorista chegar (minutos)
    'Type_Bike': [0],
    'Type_Go Mini': [1],                     # Exemplo: a corrida é do tipo 'Go Mini'
    'Type_Go Sedan': [0],
    'Type_Premier Sedan': [0],
    'Type_Uber XL': [0],
    'Type_eBike': [0],
    'cancellation_rate_percent_drop': [18.5], # Taxa de cancelamento histórica do local de destino
    'cancellation_rate_percent_pickup': [22.0] # Taxa de cancelamento histórica do local de partida
})

# Normalizar os dados
dados_novos_scaled = scaler.transform(dados_novos)

# Fazer a previsão
probabilidade_cancelar = model.predict(dados_novos_scaled)
previsao_classe = (probabilidade_cancelar > 0.5).astype(int)

# Exibir o resultado
print(f"Probabilidade de Cancelamento: {probabilidade_cancelar[0][0]:.2%}")
if previsao_classe[0][0] == 1:
    print("Resultado: ALTA chance de cancelamento. Recomenda-se procurar outro motorista.")
else:
    print("Resultado: BAIXA chance de cancelamento. A corrida pode ser enviada.")

```

## 🔮 Próximos Passos

  * **Pipeline de Produção:** Desenvolver um pipeline automatizado para re-treinar o modelo periodicamente com novos dados.
  * **Análise por Região:** Avaliar se os atributos e o modelo se comportam de maneira diferente em outras cidades ou países, considerando fatores culturais.
  * **Otimização de Hiperparâmetros:** Utilizar técnicas como GridSearchCV ou RandomizedSearchCV para encontrar a combinação ideal de hiperparâmetros para a rede neural.
