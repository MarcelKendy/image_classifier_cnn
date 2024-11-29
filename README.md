---

# Projeto de Classificação de Imagens

Este projeto utiliza modelos CNN (Convolutional Neural Network) de redes neurais para classificar imagens em diferentes categorias. Ele foi desenvolvido em Python, utilizando frameworks como torch, scikit-learn, matplotlib e numpy.

Seu objetivo é a implementação de um sistema de treinamento de modelos de redes neurais utilizando PyTorch, com foco na classificação de imagens, para criar uma aplicação flexível e eficiente para treinar diferentes modelos de aprendizado profundo em conjuntos de dados de imagens, utilizando recursos como parada antecipada (early stopping) e controle de tempo máximo de treinamento. 

O sistema é capaz de treinar os modelos, avaliar o desempenho no conjunto de validação e salvar o modelo com a melhor performance. Durante o treinamento, o sistema calcula as perdas de treinamento e validação para monitorar o progresso do modelo. O critério de perda utilizado é a função cross_entropy_loss, que é adequada para problemas de classificação com múltiplas classes. A implementação também permite personalizar o número de épocas e sua "patience", o otimizador, a função de perda entre outros parâmetros.

O projeto é compatível com uma variedade de modelos populares, e inclui 5 deles:

ResNet-18: Um modelo eficiente para tarefas de classificação de imagens, com foco em redes residuais.
VGG-16: Um modelo simples e poderoso, baseado em camadas convolucionais profundas.
InceptionV3: Um modelo mais avançado que utiliza uma arquitetura de múltiplos caminhos e camadas convolucionais de diferentes tamanhos.
AlexNet: Um dos modelos pioneiros em redes neurais convolucionais profundas, com excelente desempenho em várias tarefas de classificação de imagens.
DenseNet-121: Um modelo com conexões densas entre camadas, que facilita o fluxo de informações e melhora a precisão.
Esse projeto demonstra como diferentes modelos podem ser treinados e avaliados, permitindo comparar suas performances e escolher o melhor para uma aplicação específica.

Foi desenvolvido inteiramente por mim, Marcel Kendy Rabelo Matsumoto, de matrícula 5200 em São Gotardo/MG para a disciplina SIN 393 lecionada por João Fernando Mari [joaofmari.github.io](https://joaofmari.github.io/) do curso de Sistemas de Informação - UFV/CRP 

Livre para qualquer tipo de uso.

## Pré-requisitos

1. **Python 3.6 ou superior**: Certifique-se de que o Python está instalado na sua máquina. [Baixe aqui](https://www.python.org/downloads/).

2. **Instalação das Bibliotecas Necessárias**: Execute os comandos abaixo para instalar as bibliotecas usadas no projeto:

```bash
pip install numpy torch torchvision scikit-learn matplotlib
```

## Estrutura do Projeto

- `image_dataset/`: Pasta contendo subpastas com as imagens organizadas por categoria.
- `classifier.py`: Script principal que executa o projeto de classificação de imagens.

Certifique-se de que as imagens estão organizadas na pasta `image_dataset` em subpastas, com cada subpasta representando uma categoria.

## Como Executar

Para rodar o projeto, utilize o comando:

```bash
python classifier.py
```

Após a execução, os resultados serão exibidos no console e salvos em um arquivo `results.txt` na pasta do projeto. Esse arquivo conterá as métricas e relatório de classificação.

---
