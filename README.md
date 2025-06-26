# Smart CCD

**Smart CCD** é um software desenvolvido para análise de cromatografia por densitometria (CCD), com funcionalidades para captura de imagens, corte, normalização e quantificação utilizando técnicas de visão computacional e aprendizado de máquina (CNN).

## Funcionalidades

- Captura de imagem via webcam
- Corte e visualização da região de interesse
- Cálculo de área normalizada
- Armazenamento de dados em Excel
- Treinamento de modelo de rede neural convolucional (InceptionV3)
- Interface gráfica desenvolvida com [Flet](https://flet.dev/)

## Como usar (Windows)

1. Instale o Python 3.10 ou superior.
2. Instale os requisitos com:

```bash
pip install -r requirements.txt
```

3. Execute o atalho `Iniciar_SmartCCD.bat` na raiz do repositório para iniciar o app com terminal.

## Requisitos

- Python ≥ 3.10
- OpenCV
- TensorFlow
- Pandas
- Pillow
- Flet
- Matplotlib

## Autores

- **Daniella L. Vale** – Desenvolvimento completo do software e testes
- **Rodolfo S. Barboza** – Idealização das funcionalidades e validação dos requisitos

## Apoio

Este projeto contou com apoio de ferramentas de IA generativa como **ChatGPT (OpenAI)** e **DeepSeek** para estruturação e aceleração do desenvolvimento de funções e lógica de engenharia de software.
