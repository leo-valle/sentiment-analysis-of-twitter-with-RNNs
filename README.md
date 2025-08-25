# Análise de Sentimentos e Emoções em Tweets com Redes Neuronais Recorrentes

## Visão Geral do Projeto

Este projeto explora o campo do Processamento de Linguagem Natural (PLN) para classificar sentimentos (positivo, negativo, neutro) e emoções em tweets relacionados com a marca Dell. Foram implementadas e comparadas quatro arquiteturas diferentes de Redes Neuronais Recorrentes (RNNs) para avaliar a sua eficácia nesta tarefa: **LSTM**, **GRU**, **SimpleRNN** e **LSTM Bidirecional**.

O objetivo é demonstrar um pipeline completo de PLN, desde a limpeza e pré-processamento dos dados até ao treino e avaliação de modelos de deep learning.

## Dataset

O dataset utilizado é o `sentiment-emotion-labelled_Dell_tweets.csv`, que contém uma coleção de tweets sobre a Dell, cada um rotulado com um sentimento e uma emoção.

## Metodologia

O fluxo de trabalho do projeto foi estruturado nas seguintes etapas:

1.  **Carregamento e Limpeza dos Dados:**
    * Os dados foram carregados a partir do ficheiro CSV.
    * Foi aplicada uma função de pré-processamento para limpar o texto de cada tweet, que incluiu:
        * Conversão para minúsculas.
        * Remoção de URLs, menções (@), hashtags (#) e caracteres não alfabéticos.
        * Remoção de *stopwords* (palavras comuns como "a", "o", "de") em inglês.

2.  **Balanceamento do Dataset:**
    * A análise inicial mostrou que as classes de sentimento estavam desbalanceadas (ex: muito mais tweets negativos do que neutros).
    * Para evitar que o modelo ficasse enviesado para a classe maioritária, foi criado um conjunto de treino balanceado, amostrando aleatoriamente **7.000 exemplos** de cada uma das três classes de sentimento (positivo, negativo e neutro).

3.  **Divisão dos Dados:**
    * O conjunto balanceado de 21.000 tweets foi dividido em **80% para treino** (16.800 tweets) e **20% para validação** (4.200 tweets).
    * O conjunto de teste final foi formado pela junção do conjunto de validação com os dados restantes do dataset original que não foram usados para o treino, garantindo uma avaliação robusta em dados não vistos e desbalanceados.

4.  **Tokenização e Padding:**
    * O texto foi convertido em sequências numéricas através de um `Tokenizer`, considerando um vocabulário máximo de **1.000 palavras** mais frequentes.
    * As sequências foram padronizadas para um comprimento máximo de **144 tokens** (o limite de caracteres de um tweet) através de *padding*, para que todas as entradas tivessem o mesmo tamanho.

5.  **Codificação dos Rótulos:**
    * Os rótulos de sentimento (negativo, neutro, positivo) foram convertidos para um formato numérico e, em seguida, para o formato *one-hot encoding*, adequado para a função de perda `categorical_crossentropy`.

## Arquiteturas dos Modelos

Foram treinados e avaliados quatro modelos de RNN, todos com uma arquitetura base semelhante, variando apenas a camada recorrente:

1.  **LSTM (Long Short-Term Memory):** Eficaz para aprender dependências de longo prazo.
2.  **GRU (Gated Recurrent Unit):** Uma versão mais simples e computacionalmente mais eficiente que a LSTM.
3.  **SimpleRNN:** A arquitetura de RNN mais básica.
4.  **Bidirectional LSTM:** Processa a sequência de texto em ambas as direções (da esquerda para a direita e da direita para a esquerda), capturando um contexto mais rico.

Todos os modelos utilizaram uma camada de `Embedding` inicial e uma camada `Dense` final com ativação `softmax` para a classificação. Foram também aplicadas técnicas de regularização como `Dropout` para evitar o sobreajuste (*overfitting*).

## Resultados

Após o treino por 15 épocas, os modelos apresentaram as seguintes acurácias no conjunto de teste:

| Modelo              | Acurácia no Teste |
| ------------------- | :---------------: |
| **LSTM Bidirecional** |    **73.24%** |
| LSTM                |      72.02%       |
| GRU                 |      72.00%       |
| SimpleRNN           |      67.18%       |

A **LSTM Bidirecional** obteve o melhor desempenho, o que sugere que a análise do contexto em ambas as direções do texto foi benéfica para esta tarefa.

## Como Executar o Projeto

1.  **Pré-requisitos:** Certifique-se de ter o Python 3 e as seguintes bibliotecas instaladas:
    ```
    pandas
    numpy
    matplotlib
    seaborn
    nltk
    tensorflow
    scikit-learn
    ```
    Pode instalá-las com o comando:
    `pip install pandas numpy matplotlib seaborn nltk tensorflow scikit-learn`

2.  **Estrutura de Dados:**
    * Coloque o ficheiro do notebook (`ProjetoFinal_ET287_v2.ipynb`) e o dataset (`sentiment-emotion-labelled_Dell_tweets.csv`) na mesma pasta.

3.  **Execução:**
    * Abra o notebook num ambiente como Jupyter Notebook ou Google Colab.
    * Execute as células em ordem para reproduzir a análise e os resultados.

## Tecnologias Utilizadas

* **Linguagem:** Python 3
* **Ambiente:** Jupyter Notebook / Google Colab
* **Bibliotecas Principais:**
    * TensorFlow (com Keras) para a construção dos modelos de deep learning.
    * Scikit-learn para pré-processamento e métricas de avaliação.
    * Pandas e NumPy para manipulação de dados.
    * NLTK para processamento de texto (remoção de stopwords).
    * Matplotlib e Seaborn para a visualização dos resultados.
