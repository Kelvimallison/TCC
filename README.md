# TCC
# Previsão de Vendas de Smartphones no Varejo
### Aprendizado de Máquina aplicado à gestão de compras e estoque

## Descrição do Projeto
Este repositório contém o código desenvolvido no Trabalho de Conclusão de Curso (TCC) do MBA em Data Science e Analytics da USP/Esalq, com o título **"Previsão de vendas no setor de varejo de telefonia"**.

O trabalho utilizou o histórico de vendas de smartphones das marcas Apple, Samsung e Motorola de uma rede varejista de Minas Gerais, entre 2022 e 2025, para **prever o volume mensal de vendas de cada modelo de aparelho** e **identificar os modelos em declínio de vendas**, apoiando as decisões de compra e de reposição de estoque.


## Objetivo
Desenvolver e comparar modelos de aprendizado de máquina para:

- prever o volume mensal de vendas de cada modelo de smartphone;
- identificar os modelos em declínio de vendas;
- considerar a sazonalidade e a fase do ciclo de vida de cada aparelho;
- apoiar as decisões de compra e de gestão de estoque.

---

## Técnicas e Tecnologias Utilizadas

### Linguagem e Bibliotecas
- Python
- Pandas e NumPy
- Scikit-learn
- XGBoost
- Matplotlib e Seaborn

### Modelos de Aprendizado de Máquina
**Previsão do volume de vendas (regressão)**
- Regressão Linear (baseline)
- Random Forest
- XGBoost (configuração padrão e ajustado com GridSearchCV)

**Identificação de declínio de vendas (classificação)**
- Regressão Logística
- Random Forest
- XGBoost

### Métricas de Avaliação
- Regressão: MAE, RMSE, R² e SMAPE
- Classificação: acurácia balanceada, precisão, revocação, F1-score, coeficiente Kappa e matriz de confusão

---

## Base de Dados
Os dados de vendas pertencem a uma empresa real e são protegidos por acordo de confidencialidade, por isso **não são publicados neste repositório**. Dados pessoais de clientes foram removidos durante o tratamento, em conformidade com a LGPD.



## Autor
**Kelvim Allison Silva Almeida**
MBA em Data Science e Analytics – USP/Esalq

Orientador: Prof. Henrique Raymundo Gioia
