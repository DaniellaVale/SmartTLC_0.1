"""
Application name: SmartTLC
Version: 0.1

Software for Thin Layer Chromatography (TLC) analysis with quantification
and normalization functionalities for chromatographic spots using computer vision and machine learning.

Authors:
    - Daniella L. Vale (Software development)
    - Rodolfo S. Barboza (Concept and functional suggestions)

Description:
    This software was developed to assist in TLC image analysis,
    allowing sample capture via webcam, cropping, storage, spot normalization,
    and training of convolutional neural network (CNN) models for concentration quantification.

License:
    MIT License (modified for non-commercial use)
"""

import flet as ft
import cv2
import base64
from PIL import Image
from io import BytesIO
import threading
import time
import os
import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Variáveis globais
ultima_imagem = None
imagem_cortada = None
tabela = pd.DataFrame()
tabela.to_excel("dadosCCD.xlsx", index=False)
arquivo_da_imagem = None
percentuais = []
grafico_container = None
camera_thread = None
camera_running = False

"""Reset all page elements to default state"""
def resetar_estado_da_pagina(page):
    page.overlay.clear()
    page.appbar = None
    page.drawer = None
    page.floating_action_button = None
    page.bottom_appbar = None
    page.dialog = None
    page.snack_bar = None
    page.controls.clear()
    page.update()

"""Navigate to another page with proper cleanup"""
def ir_para_pagina(page, funcao_pagina):
    global camera_thread, camera_running

    if camera_running:
        camera_running = False
        if camera_thread and camera_thread.is_alive():
            camera_thread.join(timeout=0.5)

    resetar_estado_da_pagina(page)
    funcao_pagina(page)
    page.update()

"""Main home page of the application"""
def pagina_inicial(page):
    page.title = "TLC Analysis"
    page.clean()
    page.bgcolor = ft.colors.BLUE_50
    
    titulo = ft.Text(
        value="TLC Analysis", 
        text_align=ft.TextAlign.CENTER,
        color=ft.colors.BLUE_GREY_700,
        size=40
    )
    
    botao_inicio = ft.ElevatedButton(
        text="Start samples",
        color=ft.colors.WHITE,
        bgcolor=ft.colors.BLUE_700,
        icon=ft.icons.INSIGHTS,
        icon_color=ft.colors.WHITE,
        on_click=lambda e: ir_para_pagina(page, amostras),
        width=400,
        height=50
    )
    
    botao_arquivo = ft.ElevatedButton(
        text="Arquivos",
        color=ft.colors.WHITE,
        bgcolor=ft.colors.BLUE_700,
        icon=ft.icons.FILE_OPEN,
        icon_color=ft.colors.WHITE,
        width=400,
        height=50
    )
    
    page.add(
        ft.Column(
            [
                ft.Container(titulo, padding=20, alignment=ft.alignment.center),
                ft.Container(botao_inicio, padding=10, alignment=ft.alignment.center)
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            expand=True
        )
    )
    page.update()

"""Page for capturing and processing TLC samples"""
def amostras(page):
    titulo=ft.Container(content=ft.Text(value="TLC Image Capture", 
                                      text_align=ft.alignment.bottom_center,
                                      color=ft.colors.BLUE_GREY_700,
                                      size=40),
                                      left=20,
                                      height=100,
                                      right=100,
                                      top=ft.alignment.bottom_center,
                                      alignment=ft.alignment.bottom_center)
    
    botao_pag_inicial = ft.ElevatedButton(text="⬅Back to home",
                                        on_click=lambda e: ir_para_pagina(page, pagina_inicial),
                                        right=100,
                                        top=560)
    
    botao_CNN = ft.ElevatedButton(text="📊Image Analysis by Regression",
                                on_click=lambda e: ir_para_pagina(page, CNN),
                                right=80,
                                top=440)
    
    botao_normalizacaoCCD = ft.ElevatedButton(text="🔬TLC Normalization",
                                            on_click=lambda e: ir_para_pagina(page, normalizacao),
                                            right=100,
                                            top=500)
    
    aviso_cortar = ft.Text("A new window will appear. After selecting the area, press Enter",
                          size=12, color=ft.colors.BLUE_GREY_600, italic=True)
    
    container_aviso = ft.Container(aviso_cortar, top=480, right=750)
    mensagem_salvo = ft.Text("", color=ft.colors.GREEN, size=14)
    container_mensagem = ft.Container(mensagem_salvo, top=630, right=370)
"""Display live camera feed"""
    def aovivo():
        global ultima_imagem
        captura = cv2.VideoCapture(1)
        if not captura.isOpened():
            print('Error: Camera could not be opened.')
            return

        captura.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        captura.set(cv2.CAP_PROP_FOCUS, 30)
        captura.set(cv2.CAP_PROP_AUTO_WB, 0)
        captura.set(cv2.CAP_PROP_WB_TEMPERATURE, 4000)
        captura.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        captura.set(cv2.CAP_PROP_EXPOSURE, -6)
        captura.set(cv2.CAP_PROP_BRIGHTNESS, 30)
        captura.set(cv2.CAP_PROP_CONTRAST, 60)
        captura.set(cv2.CAP_PROP_SATURATION, 10)

        while True:
            funcionando, frame = captura.read()
            pildaimagem = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            local_de_armazenar = BytesIO()
            pildaimagem.save(local_de_armazenar, format="JPEG")
            imagemdacamera=base64.b64encode(local_de_armazenar.getvalue()).decode("utf-8")
            imagem_camera.src_base64 = imagemdacamera
            ultima_imagem=frame
            page.update()

    imagem_camera = ft.Image(width=300,height=400, right=800, top=100)
    
    corfundo2= ft.Container(expand=True,
                           gradient=ft.LinearGradient(begin=ft.alignment.top_left,
                                                     end=ft.alignment.bottom_right,
                                                     colors=[ft.colors.BLUE_100,ft.colors.BLUE_900],
                                                     stops=[0,1]))
    
    nome_amostra1=ft.TextField(label="nome da amostra")
    nome_amostra=ft.Container(nome_amostra1,top=500, right=300)
    concentracao_amostra1= ft.TextField(label="Concentração da amostra")
    concentracao_amostra = ft.Container(concentracao_amostra1, top=550, right=300)
"""Crop the selected area from the image"""
    def cortar_imagem(e):
        global ultima_imagem, imagem_cortada
        cortar=cv2.selectROI("seleção da imagem", ultima_imagem, showCrosshair=True)
        x,y,l,a = cortar
        cortada = ultima_imagem[int(y): int(y+a), int(x): int(x+l)]
        bolenanodaimagem, memoria = cv2.imencode('.png', cortada)
        cortada_base64 = base64.b64encode(memoria).decode('utf-8')
        imagemcortada_foto.src_base64=cortada_base64
        imagem_cortada=cortada
        cv2.destroyAllWindows()

    botao_cortar=ft.ElevatedButton(text="✂Crop image", on_click=cortar_imagem, top=440, right=920)
    imagemcortada_foto= ft.Image(cortar_imagem, width=300,height=400, right=300, top=100)
"""Save the processed image and sample data"""
    def salvarimagem(e):
        global imagem_cortada
        nome= nome_amostra1.value.strip() 
        concentracao = concentracao_amostra1.value.strip()
        if imagem_cortada is not None:
            pil_da_imagem = Image.fromarray(cv2.cvtColor(imagem_cortada, cv2.COLOR_BGR2RGB))
            pil_da_imagem.save(f"{nome}, C= {concentracao}.jpg")
        dados_amostra = pd.DataFrame({"Nome":[nome], "concentração":[concentracao]})
        dados_anteriores = pd.read_excel("dadosCCD.xlsx")
        juntadar_tabelas=pd.concat([dados_anteriores,dados_amostra])
        juntadar_tabelas.to_excel("dadosCCD.xlsx", index=False)
        page.snack_bar = ft.SnackBar(content=ft.Text("✅ Sample saved successfully!"), open=True)
        page.update()

    botao_capturar= ft.ElevatedButton(text="💾Save image",on_click=salvarimagem, top=600, right=370)
    
    estilo= ft.Stack(expand=True, 
                    controls=[titulo, imagem_camera, botao_capturar, 
                             nome_amostra,concentracao_amostra, botao_cortar,container_aviso,
                             imagemcortada_foto, botao_pag_inicial, botao_CNN, botao_normalizacaoCCD, container_mensagem])

    page.add(estilo)
    threading.Thread(target=aovivo, daemon=True).start()
"""Extract sample name and concentration from filename"""
def extrair_informacao_do_nome(nome_arquivo):
    informacao = re.search(r"(.+), C= ([\d.]+)\.jpg", nome_arquivo)
    if informacao:
        nome, concentracao = informacao.groups()
        return nome, float(concentracao)
    return None, None
"""Load image data for regression analysis"""
def carregar_dados_regressao(pasta_imagens):
    concentracoes = []
    intensidades = []
    
    for nome_arquivo in os.listdir(pasta_imagens):
        if nome_arquivo.endswith(".jpg"):
            caminho = os.path.join(pasta_imagens, nome_arquivo)
            img = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img_invertida = 255 - img
                intensidade = np.sum(img_invertida)
                
                nome, conc = extrair_informacao_do_nome(nome_arquivo)
                if conc is not None:
                    concentracoes.append(conc)
                    intensidades.append(intensidade)
    
    return np.array(concentracoes), np.array(intensidades)
"""Page for performing regression analysis on TLC images"""
def CNN(page):
    page.clean()
    page.title = "Polynomial Regression Analysis"
    page.scroll = ft.ScrollMode.ALWAYS
    
    caminho_pasta = ""
    caminho_arquivo_teste = ""
    grafico_container = ft.Column()
    resultados_container = ft.Column()
    coef = None
    modelo = None
    
    comando_inicial = ft.Text("Select folder with images for analysis", 
                            size=20, weight=ft.FontWeight.BOLD)
    
    pasta_imagens = ft.Text("📂 No folder selected", size=14, italic=True)
    arquivo_teste = ft.Text("📂 No test file selected", size=14, italic=True)
    status_text = ft.Text("🔄 Waiting for action...", size=14)
    resultado_predicao = ft.Text("", size=16, weight=ft.FontWeight.BOLD, color=ft.colors.BLUE_800)

    """Handle folder selection"""
    def selecionar_pasta(e: ft.FilePickerResultEvent):
        nonlocal caminho_pasta
        if e.path:
            caminho_pasta = e.path
            pasta_imagens.value = f"📂 Selected folder: {caminho_pasta}"
            page.update()
    """Handle test file selection"""
    def selecionar_arquivo_teste(e: ft.FilePickerResultEvent):
        nonlocal caminho_arquivo_teste
        if e.files:
            caminho_arquivo_teste = e.files[0].path
            arquivo_teste.value = f"📂 Selected file: {os.path.basename(caminho_arquivo_teste)}"
            page.update()
    
    pegar_pasta = ft.FilePicker(on_result=selecionar_pasta)
    pegar_arquivo = ft.FilePicker(on_result=selecionar_arquivo_teste)
    page.overlay.extend([pegar_pasta, pegar_arquivo])

    """Calculate intensity from an image"""
    def calcular_intensidade(caminho_imagem):
        img = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img_invertida = 255 - img
            return np.sum(img_invertida)
        return None
    """Predict concentration from test image"""
    def prever_concentracao(e):
        nonlocal coef, modelo
        
        if coef is None or modelo is None:
            resultado_predicao.value = "⚠️ First train the model with calibration images!"
            page.update()
            return
            
        if not caminho_arquivo_teste:
            resultado_predicao.value = "⚠️ No test file selected!"
            page.update()
            return
            
        try:
            intensidade = calcular_intensidade(caminho_arquivo_teste)
            if intensidade is None:
                resultado_predicao.value = "❌ Error processing test image!"
                page.update()
                return
                
            # Resolver a equação quadrática para encontrar C dado I
            # coef[0]*C² + coef[1]*C + (coef[2] - intensidade) = 0
            a, b, c = coef[0], coef[1], coef[2] - intensidade
            
            discriminante = b**2 - 4*a*c
            if discriminante < 0:
                resultado_predicao.value = "❌ Intensity outside calibration range!"
                page.update()
                return
                
            concentracao = (-b + np.sqrt(discriminante)) / (2*a)
            
            resultado_predicao.value = f"Predicted concentration: {concentracao:.3f} mg/mL"
            page.update()
            
        except Exception as err:
            resultado_predicao.value = f"❌ Erro: {str(err)}"
            page.update()
    """Create calibration curve graph"""
    def criar_grafico(x, y, coef):
        plt.figure(figsize=(8, 5))
        plt.scatter(x, y, color="blue", label="Experimental data")
        
        x_ordenado = np.linspace(min(x), max(x), 500)
        y_modelo = np.poly1d(coef)(x_ordenado)
        
        plt.plot(x_ordenado, y_modelo, color="red", label="Polynomial model (degree 2)")
        plt.xlabel("Concentration (mg/mL)")
        plt.ylabel("Total inverted intensity")
        plt.title("Calibration curve")
        plt.legend()
        plt.grid(True)
        
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=100)
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
    """Perform the regression analysis"""
    def iniciar_analise(e):
        nonlocal grafico_container, resultados_container, coef, modelo
    
        if not caminho_pasta:
            status_text.value = "⚠️ Error: No folder selected!"
            page.update()
            return
    
        status_text.value = "📥 Loading data..."
        grafico_container.controls = []
        resultados_container.controls = []
        page.update()
    
        try:
            # Load data: x=concentrations, y=intensities
            x_real, y = carregar_dados_regressao(caminho_pasta)
        
            if len(x_real) == 0:
                status_text.value = "⚠️ Error: No valid images found!"
                page.update()
                return
        
            status_text.value = "⚙️ Calculating regression..."
            page.update()
        
            # Polynomial regression: y(intensity) = f(x(concentration))
            coef = np.polyfit(x_real, y, 2)
            modelo = np.poly1d(coef)
        
            # Predict intensities for known concentrations
            y_pred = modelo(x_real)
        
            # Calculate predicted concentrations (solving quadratic equation)
            x_pred = []
            for intensidade in y:
                a, b, c = coef[0], coef[1], coef[2] - intensidade
                discriminante = b**2 - 4*a*c
                if discriminante >= 0:
                    concentracao = (-b + np.sqrt(discriminante)) / (2*a)
                    x_pred.append(concentracao)
                else:
                    x_pred.append(np.nan)  #  Marker for out-of-range values
        
            # Calculate MAE for concentrations
            x_pred_validas = [x for x in x_pred if not np.isnan(x)]
            x_real_validas = [x_real[i] for i in range(len(x_pred)) if not np.isnan(x_pred[i])]
        
            if len(x_pred_validas) > 0:
                mae_conc = mean_absolute_error(x_real_validas, x_pred_validas)
            else:
                mae_conc = float('nan')
        
            # Calculate R² score
            mae_int = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)
        
            # Create UI elements
            equacao = ft.Container(
                content=ft.Text(
                    f"Modelo: Intensidade = {coef[0]:.4e}·C² + {coef[1]:.4e}·C + {coef[2]:.4f}",
                    size=16,
                    weight=ft.FontWeight.BOLD
                ),
                padding=10
            )
        
            metricas = ft.Container(
                content=ft.Column([
                    ft.Text("📊 Metrics:", weight=ft.FontWeight.BOLD),
                    ft.Text(f"• MAE (concentration): {mae_conc:.4f} mg/mL"),
                    ft.Text(f"• R²: {r2:.4f}")
                ]),
                padding=10
            )
        
            tabela = ft.DataTable(
                columns=[
                    ft.DataColumn(ft.Text("Sample")),
                    ft.DataColumn(ft.Text("Real Conc. (mg/mL)")),
                    ft.DataColumn(ft.Text("Predicted Conc. (mg/mL)")),
                    ft.DataColumn(ft.Text("Difference"))
                ],
                rows=[
                    ft.DataRow(
                        cells=[
                            ft.DataCell(ft.Text(f"{i+1}")),
                            ft.DataCell(ft.Text(f"{x_real[i]:.4f}")),
                            ft.DataCell(ft.Text(f"{x_pred[i]:.4f}" if not np.isnan(x_pred[i]) else "N/A")),
                            ft.DataCell(ft.Text(f"{abs(x_real[i]-x_pred[i]):.4f}" if not np.isnan(x_pred[i]) else "N/A"))
                        ]
                    ) for i in range(len(x_real))
                ],
                width=700
            )
        
            img_base64 = criar_grafico(x_real, y, coef)
            grafico = ft.Image(
                src_base64=img_base64,
                width=600,
                height=400,
                fit=ft.ImageFit.CONTAIN
            )
        
            resultados_container.controls = [
                ft.Card(
                    content=ft.Container(
                        content=ft.Column([
                            equacao,
                            metricas,
                            ft.Text("📋 Results:", weight=ft.FontWeight.BOLD),
                            tabela
                        ]),
                        padding=20
                    ),
                    width=800
                )
            ]
        
            grafico_container.controls = [
                ft.Card(
                    content=ft.Container(
                        content=grafico,
                        padding=20
                    )
                )
            ]
        
            status_text.value = "✅ Analysis completed successfully!"
        
        except Exception as err:
            status_text.value = f"❌ Erro: {str(err)}"
            resultados_container.controls = [
                ft.Text(f"Detalhes do erro: {str(err)}", color="red")
            ]
    
        page.update()
    
    page.add(
        ft.Column([
            comando_inicial,
            ft.Row([
                ft.ElevatedButton("Select Folder", 
                                 on_click=lambda e: pegar_pasta.get_directory_path()),
                pasta_imagens
            ], alignment=ft.MainAxisAlignment.CENTER),
            
            ft.ElevatedButton("Start Analysis", 
                             on_click=iniciar_analise,
                             icon=ft.icons.ANALYTICS),
            
            ft.Divider(height=20),
            
            ft.Text("Test new sample:", size=16, weight=ft.FontWeight.BOLD),
            ft.Row([
                ft.ElevatedButton("Select Test Image",
                                 on_click=lambda e: pegar_arquivo.pick_files(allowed_extensions=["jpg", "jpeg", "png"])),
                arquivo_teste
            ], alignment=ft.MainAxisAlignment.CENTER),
            
            ft.ElevatedButton("Predict Concentration",
                            on_click=prever_concentracao,
                            icon=ft.icons.CALCULATE),
            
            resultado_predicao,
            
            status_text,
            resultados_container,
            grafico_container,
            ft.ElevatedButton("⬅ Back to home", 
                             on_click=lambda e: ir_para_pagina(e.page, pagina_inicial))
        ], 
        scroll=ft.ScrollMode.ALWAYS,
        spacing=20)
    )
    page.update()

# página para fazer normalização da CCD

# Global variables
#creates an empty variable to store the image bytes in the future
arquivo_da_imagem = None
#creates an empty list where the percentages will be stored
percentuais = []
#function to load the image, uses the and for execution, image preview icon and status text
"""Load image from file"""
def carregar_imagem(e, visualizacao_da_imagem, estatus_do_texto):
    #image file global variable, image bytes
    global arquivo_da_imagem
    # if no file is selected
    if not e.files:
        #exibir a mensagem
        estatus_do_texto.value = "⚠️ No image loaded."
        #make text status update
        estatus_do_texto.update()
        return
    # put the selected file in the file variable
    arquivo = e.files[0]

    

    try:
        # rb eh modo arquivo binário
        # ler o arquivo aberto como abrir
        with open(arquivo.path, "rb") as abrir:
            #substituir o arquivo da imagem pela leitura o arquivo aberto (abrir)
            arquivo_da_imagem = abrir.read()
        #fazer a conversão do pil da imagem
        pil_da_imagem = Image.open(BytesIO(arquivo_da_imagem))
        
        #converter o arquivo da imagem em base 64 para ser lido pelo app
        img_base64 = base64.b64encode(arquivo_da_imagem).decode("utf-8")
        #substituir o src base 64 da imagem projetada pelo app (visualizacao_da_imagem) pelo arquivo de base 64
        visualizacao_da_imagem.src_base64 = img_base64
        #fazer o update da vizualização da imagem no app
        visualizacao_da_imagem.update()
        # no estatus (frase de aplicação) colocar o valor da frase abaixo
        estatus_do_texto.value = "✅ Image loaded successfully!"
        #excessao caso o codigo anterior de carregar imagem não funcione
    except Exception as err:
        # substituir a frase de estatutos pela frase abaixo
        estatus_do_texto.value = f"❌ Error loading image: {err}"
    # fazer o update do texto
    estatus_do_texto.update()

"""Process image to detect spots and calculate normalization"""
def processar_imagem():
    global arquivo_da_imagem, percentuais

    if arquivo_da_imagem is None:
        return None, "⚠️ No image loaded!"

    try:
        # Carregar imagem (mantido igual ao original)
        pil_da_imagem = Image.open(BytesIO(arquivo_da_imagem)).convert("RGB")
        imagem = np.array(pil_da_imagem)
        imagem_bgr = cv2.cvtColor(imagem, cv2.COLOR_RGB2BGR)

        # Pré-processamento (mantido igual)
        imagem_bgr = cv2.GaussianBlur(imagem_bgr, (9, 9), 0)
        imagem_hsv = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2HSV)
        
        # Segmentação de cor (mantido igual)
        cor_baixa = np.array([80, 30, 30])
        cor_alta = np.array([140, 255, 255])
        mascara_cor = cv2.inRange(imagem_hsv, cor_baixa, cor_alta)

        # Limpeza morfológica (mantido igual)
        kernel = np.ones((9, 9), np.uint8)
        mascara_limpa = cv2.morphologyEx(mascara_cor, cv2.MORPH_CLOSE, kernel)
        mascara_limpa = cv2.dilate(mascara_limpa, kernel, iterations=1)

        # Detecção de contornos (mantido igual)
        contornos, _ = cv2.findContours(mascara_limpa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        intensidades = []
        centros_x = []

        imagem_cinza = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2GRAY)

        for contorno in contornos:
            area = cv2.contourArea(contorno)
            if area > 150:
                mascara_mancha = np.zeros_like(imagem_cinza)
                cv2.drawContours(mascara_mancha, [contorno], -1, 255, -1)

                x, y, w, h = cv2.boundingRect(contorno)
                roi = imagem_cinza[y:y+h, x:x+w]
                roi_invertido = 255 - roi
                sinal_total = np.sum(roi_invertido * (mascara_mancha[y:y+h, x:x+w] // 255))

                intensidades.append(sinal_total)

                M = cv2.moments(contorno)
                centro_x = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
                centros_x.append(centro_x)

                cv2.drawContours(imagem_bgr, [contorno], -1, (0, 0, 255), 2)

        percentuais = []

        if intensidades:
            intensidades_ordenadas = [i for _, i in sorted(zip(centros_x, intensidades))]
            concentracoes_conhecidas = [25, 12.5, 6.25, 3.13, 1.57, 0.785]

            if len(intensidades_ordenadas) == len(concentracoes_conhecidas):
                coeficientes = np.polyfit(intensidades_ordenadas, concentracoes_conhecidas, 2)
                modelo = np.poly1d(coeficientes)
                concentracoes_estimadas = modelo(intensidades_ordenadas)
                total = sum(concentracoes_estimadas)
                percentuais = [(c / total) * 100 for c in concentracoes_estimadas]
            else:
                total = sum(intensidades_ordenadas)
                percentuais = [(i / total) * 100 for i in intensidades_ordenadas]

        _, buffer = cv2.imencode(".png", imagem_bgr)
        imagem_base64 = base64.b64encode(buffer).decode("utf-8")
        return imagem_base64, "✅ Processing completed!"

    except Exception as err:
        return None, f"❌ Processing error: {err}"


"""Display detected contours on image"""
def exibir_contornos(e, vizualidacao_da_imagem, estatus_do_texto):
    def processar():
        src, msg = processar_imagem()
        if src:
            vizualidacao_da_imagem.src_base64 = src
            vizualidacao_da_imagem.update()
        estatus_do_texto.value = msg
        estatus_do_texto.update()
    threading.Thread(target=processar, daemon=True).start()

grafico_container = None

"""Display spot distribution graph"""
def exibir_grafico(e):
    global percentuais, grafico_container

    if not percentuais:
        e.page.snack_bar = ft.SnackBar(ft.Text("⚠️ No spots detected!"), open=True)
        e.page.update()
        return

    try:
        # Generate matplotlib graph
        fig, ax = plt.subplots()
        barras = ax.bar(range(1, len(percentuais) + 1), percentuais, color='blue')
        ax.set_xlabel("Spot")
        ax.set_ylabel("Normalized Area (%")
        ax.set_title("Spot Distribution")
        ax.set_xticks(range(1, len(percentuais) + 1))
        for barra in barras:
            altura = barra.get_height()
            ax.text(barra.get_x() + barra.get_width() / 2, altura + 0.5, f"{altura:.2f}%", ha='center', va='bottom', fontsize=8)

        # Convert to base64 image
        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        plt.close(fig)

        # Update graph container
        grafico_container.controls.clear()
        grafico_container.controls.append(ft.Image(src_base64=img_str, width=400, height=300))
        grafico_container.update()

    except Exception as err:
        e.page.snack_bar = ft.SnackBar(ft.Text(f"❌ Erro ao gerar gráfico: {err}"), open=True)
        e.page.update()



"""Page for TLC spot normalization analysis"""
def normalizacao(page):
    page.title = "Chromatography Analysis"
    page.scroll = ft.ScrollMode.ALWAYS

    estatus_do_texto = ft.Text("🔄 Waiting for action...")
    visualizacao_da_imagem = ft.Image(width=300, height=300, fit=ft.ImageFit.CONTAIN)

    file_picker = ft.FilePicker(on_result=lambda e: carregar_imagem(e, visualizacao_da_imagem, estatus_do_texto))
    page.overlay.append(file_picker)

    botao_carregar = ft.ElevatedButton("📂 Load Image", on_click=lambda _: file_picker.pick_files(allowed_extensions=["png", "jpg", "jpeg"]))
    botao_contornos = ft.ElevatedButton("🔍 Show Contours", on_click=lambda e: exibir_contornos(e, visualizacao_da_imagem, estatus_do_texto))
    global grafico_container
    grafico_container = ft.Column([])

    botao_grafico = ft.ElevatedButton("📊 Show Graph", on_click=exibir_grafico)

    # Adiciona o botão e o gráfico ao layout da tela
    page.add(botao_grafico)
    page.add(grafico_container)


    botao_pag_inicial = ft.ElevatedButton(text="Back to home",
                                          on_click=lambda e: ir_para_pagina(e.page, pagina_inicial),
                                          )

    page.add(
        ft.Column([
            botao_carregar,
            visualizacao_da_imagem,
            botao_contornos,
            botao_grafico,
            estatus_do_texto,
            botao_pag_inicial
        ], alignment=ft.MainAxisAlignment.CENTER)
    )

    
def main(page: ft.Page):
    pagina_inicial(page)  # Carrega a página inicial por padrão

ft.app(target=main)
