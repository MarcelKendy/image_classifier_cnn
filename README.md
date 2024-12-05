# Projeto de Classificação de Imagens com CNNs

Este projeto utiliza modelos CNN (Convolutional Neural Network) de redes neurais para classificar imagens em diferentes categorias. Ele foi desenvolvido em Python, utilizando frameworks como torch, scikit-learn, matplotlib.

Seu objetivo é a implementação de um sistema de treinamento de modelos de redes neurais utilizando PyTorch, com foco na classificação de imagens, para criar uma aplicação flexível e eficiente para treinar diferentes modelos de aprendizado profundo em conjuntos de dados de imagens encontrados com ajuda do kaggle.com, utilizando recursos como parada antecipada (early stopping), controle de tempo máximo de treinamento, data augmentation e otimização de hiperparâmetros. 

O dataset escolhido foi o [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset), que contém 4 classes de imagens representando diferentes tipos de tumores ou a ausência dele: glioma, meningioma, sem tumor e pituitary. A escolha foi feita porque a detecção e classificação rápida de tumores cerebrais é uma área de pesquisa importante no campo de imagens médicas, e ajudam a selecionar o método de tratamento mais apropriado para salvar vidas.

O sistema é capaz de treinar os modelos, avaliar o desempenho no conjunto de validação e salvar o modelo com a melhor performance. Durante o treinamento, o sistema calcula as perdas de treinamento e validação para monitorar o progresso do modelo. O critério de perda utilizado é a função cross_entropy_loss, que é adequada para problemas de classificação com múltiplas classes. A implementação também permite personalizar o número de épocas, a tolerância sob épocas não produtivas, o otimizador, a função de perda entre outros parâmetros.

O projeto é compatível com uma variedade de modelos populares, e já inclui 5 deles:

- **ResNet-18**: Um modelo eficiente para tarefas de classificação de imagens, com foco em redes residuais.
- **VGG-16**: Um modelo simples e poderoso, baseado em camadas convolucionais profundas.
- **InceptionV3**: Um modelo mais avançado que utiliza uma arquitetura de múltiplos caminhos e camadas convolucionais de diferentes tamanhos.
- **AlexNet**: Um dos modelos pioneiros em redes neurais convolucionais profundas, com excelente desempenho em várias tarefas de classificação de imagens.
- **DenseNet-121**: Um modelo com conexões densas entre camadas, que facilita o fluxo de informações e melhora a precisão.

Esse projeto demonstra como diferentes modelos podem ser treinados e avaliados, permitindo comparar suas performances e escolher o melhor para uma aplicação específica.

Foi desenvolvido inteiramente por mim, Marcel Kendy Rabelo Matsumoto, de matrícula 5200 e residente em São Gotardo/MG para a disciplina SIN 393 (Introdução à visão computacional), lecionada por João Fernando Mari [joaofmari.github.io](https://joaofmari.github.io/) do curso de Sistemas de Informação - UFV/CRP.

Livre para qualquer tipo de uso.

## Pré-requisitos

1. **Python 3.6 ou superior**: Certifique-se de que o Python está instalado na sua máquina. [Baixe aqui](https://www.python.org/downloads/).

2. **Instalação das Bibliotecas Necessárias**: Execute os comandos abaixo para instalar as bibliotecas usadas no projeto:

```bash
pip install torch torchvision scikit-learn matplotlib seaborn
```

3. **Executando o projeto**: Dentro do diretório do projeto, execute o comando:

```bash
python classifier.py
```
