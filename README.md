---

# Projeto de Classificação de Imagens

Este projeto utiliza modelos CNN (Convolutional Neural Network) de redes neurais para classificar imagens em diferentes categorias. Ele foi desenvolvido em Python e mais informações serão listadas abaixo.
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
