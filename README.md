# Projeto IC_MachineLearning_DGA 🧪🤖

Bem-vindo ao nosso laboratório virtual de análise de Gases Dissolvidos (DGA) e Machine Learning aplicado a transformadores de potência! Este é um espaço onde a ciência encontra a tecnologia para melhorar a confiabilidade e a eficiência dos transformadores.

## Sobre o Projeto 📚

Nossa Iniciação Científica (IC) é uma jornada emocionante em direção ao futuro da manutenção preditiva. Estamos explorando como os dados de DGA podem ser transformados em informações valiosas usando algoritmos de Machine Learning.

## O que estamos fazendo? 🚀

- Desenvolvendo modelos de Machine Learning inteligentes para detectar e diagnosticar falhas em transformadores com base em análises de gases dissolvidos.
- Investigando as peculiaridades dos métodos de DGA, incluindo Roger, Doernemburg, Gas Chave e Duval, para entender quando cada um brilha.
- Criando visualizações incríveis para transformar dados complexos em insights claros.
- Colaborando com mentes brilhantes para construir um futuro mais eficiente e confiável para a energia elétrica.

## Como Você Pode Contribuir 🤝

- Dê uma olhada em nosso código e notebooks.
- Compartilhe suas ideias, sugestões e problemas na seção "Issues".
- Ajude-nos a melhorar a documentação e os relatórios técnicos.
- Junte-se às discussões e compartilhe seu conhecimento.

## Estrutura do Repositório 📂

- 📁 **Código-fonte**: Nossa base de experimentação e modelagem.
- 📁 **Dados**: Conjuntos de dados (se aplicável) e informações sobre a coleta.
- 📁 **Documentação**: Relatórios, artigos e documentos técnicos.
- 📁 **Recursos**: Referências e materiais úteis.
- 📁 **Issues**: O lugar para conversar e colaborar.

## Processo 1 - Pré-processamento de Dados

Nesta etapa, realizamos o pré-processamento dos dados para garantir que eles estejam prontos para análise e modelagem. Alguns dos principais passos incluem:

- Importação de bibliotecas Python essenciais para manipulação de dados.
- Tratamento de valores faltantes utilizando métodos como média, mediana e moda.
- Conversão da coluna "Data" para o formato de data apropriado.
- Eliminação de registros duplicados.
- Mapeamento das categorias da coluna "Defeito" para números.
- Padronização dos valores flutuantes dos gases, substituindo vírgulas por pontos.

## Processo 2 - Análise Exploratória de Dados (AED)

A Análise Exploratória de Dados é uma parte crucial deste projeto, pois nos permite entender profundamente as características dos dados relacionados à concentração de gases, tipos de defeitos e fontes. Algumas das análises realizadas incluem:

- Criação de histogramas para visualizar a distribuição dos gases dissolvidos.
- Cálculo de estatísticas descritivas, como média, desvio padrão e quartis.
- Contagem de ocorrências de cada tipo de defeito.
- Exploração da correlação entre os gases.
- Criação de gráficos de dispersão para investigar relações entre variáveis.
- Análise temporal dos dados para compreender as mudanças ao longo do tempo.

## Processo 3 - Modelo de Classificação Utilizando o Classificador OPF (Optimum-Path Forest)

O Classificador OPF é aplicado aos dados de DGA para a tarefa de classificação de tipos de defeitos. Este é um dos pontos centrais do projeto, e o código é detalhado nesta seção:

- Preparação dos dados, incluindo padronização e codificação das classes.
- Loop de execução para treinar e avaliar o classificador OPF várias vezes.
- Avaliação do desempenho do classificador, incluindo cálculo de acurácia, F1 Score e matriz de confusão.
- Apresentação organizada dos resultados, incluindo médias e desvios padrão das métricas de desempenho.

## Processo 4 - Comparação com Outros Modelos de Machine Learning

Nesta etapa, comparamos o desempenho do Classificador OPF com outros modelos populares de Machine Learning, como K-Nearest Neighbors (KNN), Decision Tree, Random Forest, Support Vector Machine (SVM), Naive Bayes e Logistic Regression. Avaliamos esses modelos em termos de Acurácia e F1 Score.

## Processo 5 - Modelo de Previsão

Nesta seção, apresentamos um modelo de previsão que pode ser usado para fazer previsões com base em novos dados de DGA. Os detalhes do modelo e seu treinamento estão disponíveis aqui.

## Bibliotecas Utilizadas

Durante todo o projeto, utilizamos as seguintes bibliotecas Python:

- Pandas: Para manipulação de dados.
- NumPy: Para operações numéricas.
- Matplotlib e Seaborn: Para criação de gráficos e visualizações.
- Plotly Express: Para gráficos interativos.
- Tabulate: Para tabulação dos resultados.

Para mais informações técnicas, consulte a documentação completa ( Artigo e Notebook ) e o código-fonte disponíveis neste repositório.

## Contato

Se você tiver dúvidas, sugestões ou estiver interessado em colaborar neste projeto, entre em contato conosco pelos seguintes meios:

- Email: vinnyciussouza@outlook.com
- LinkedIn: [Vinicius Santos](https://www.linkedin.com/in/vinicius-souza-santoss/)
- Twitter: [@ViniciusKhan](https://twitter.com/viniciuskhan)

Obrigado por visitar nosso repositório e contribuir para avanços na análise de gases dissolvidos em transformadores de potência!


Vamos transformar dados em energia! 💡⚙️

