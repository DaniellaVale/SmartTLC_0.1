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

# Global variables
last_image = None
cropped_image = None
data_table = pd.DataFrame()
data_table.to_excel("TLC_data.xlsx", index=False)
image_file = None
percentages = []
graph_container = None
camera_thread = None
camera_running = False

def reset_page_state(page):
    page.overlay.clear()
    page.appbar = None
    page.drawer = None
    page.floating_action_button = None
    page.bottom_appbar = None
    page.dialog = None
    page.snack_bar = None
    page.controls.clear()
    page.update()

def go_to_page(page, page_function):
    global camera_thread, camera_running

    if camera_running:
        camera_running = False
        if camera_thread and camera_thread.is_alive():
            camera_thread.join(timeout=0.5)

    reset_page_state(page)
    page_function(page)
    page.update()

def home_page(page):
    page.title = "Smart TLC"
    page.clean()
    page.bgcolor = ft.colors.BLUE_50
    
    title = ft.Text(
        value="TLC Analysis with Machine Learning", 
        text_align=ft.TextAlign.CENTER,
        color=ft.colors.BLUE_GREY_700,
        size=40
    )
    
    start_button = ft.ElevatedButton(
        text="Start samples",
        color=ft.colors.WHITE,
        bgcolor=ft.colors.BLUE_700,
        icon=ft.icons.INSIGHTS,
        icon_color=ft.colors.WHITE,
        on_click=lambda e: go_to_page(page, samples),
        width=400,
        height=50
    )
    
    files_button = ft.ElevatedButton(
        text="Files",
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
                ft.Container(title, padding=20, alignment=ft.alignment.center),
                ft.Container(start_button, padding=10, alignment=ft.alignment.center)
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            expand=True
        )
    )
    page.update()

def samples(page):
    title=ft.Container(content=ft.Text(value="TLC Image Capture", 
                                      text_align=ft.alignment.bottom_center,
                                      color=ft.colors.BLUE_GREY_700,
                                      size=40),
                                      left=20,
                                      height=100,
                                      right=100,
                                      top=ft.alignment.bottom_center,
                                      alignment=ft.alignment.bottom_center)
    
    home_button = ft.ElevatedButton(text="⬅Back to home",
                                        on_click=lambda e: go_to_page(page, home_page),
                                        right=100,
                                        top=560)
    
    CNN_button = ft.ElevatedButton(text="📊Image Analysis by Regression",
                                on_click=lambda e: go_to_page(page, CNN),
                                right=80,
                                top=440)
    
    normalization_button = ft.ElevatedButton(text="🔬TLC Normalization",
                                            on_click=lambda e: go_to_page(page, normalization),
                                            right=100,
                                            top=500)
    
    crop_notice = ft.Text("A new window will appear. After selecting the area, press Enter",
                          size=12, color=ft.colors.BLUE_GREY_600, italic=True)
    
    notice_container = ft.Container(crop_notice, top=480, right=750)
    save_message = ft.Text("", color=ft.colors.GREEN, size=14)
    message_container = ft.Container(save_message, top=630, right=370)

    def live_feed():
        global last_image
        capture = cv2.VideoCapture(1)
        if not capture.isOpened():
            print('Error: camera could not be opened.')
            return

        capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        capture.set(cv2.CAP_PROP_FOCUS, 30)
        capture.set(cv2.CAP_PROP_AUTO_WB, 0)
        capture.set(cv2.CAP_PROP_WB_TEMPERATURE, 4000)
        capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        capture.set(cv2.CAP_PROP_EXPOSURE, -6)
        capture.set(cv2.CAP_PROP_BRIGHTNESS, 30)
        capture.set(cv2.CAP_PROP_CONTRAST, 60)
        capture.set(cv2.CAP_PROP_SATURATION, 10)

        while True:
            working, frame = capture.read()
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            storage = BytesIO()
            pil_image.save(storage, format="JPEG")
            camera_image=base64.b64encode(storage.getvalue()).decode("utf-8")
            camera_display.src_base64 = camera_image
            last_image=frame
            page.update()

    camera_display = ft.Image(width=300,height=400, right=800, top=100)
    
    background= ft.Container(expand=True,
                           gradient=ft.LinearGradient(begin=ft.alignment.top_left,
                                                     end=ft.alignment.bottom_right,
                                                     colors=[ft.colors.BLUE_100,ft.colors.BLUE_900],
                                                     stops=[0,1]))
    
    sample_name_field=ft.TextField(label="Sample name")
    sample_name=ft.Container(sample_name_field,top=500, right=300)
    concentration_field= ft.TextField(label="Sample concentration")
    sample_concentration = ft.Container(concentration_field, top=550, right=300)

    def crop_image(e):
        global last_image, cropped_image
        crop=cv2.selectROI("Image selection", last_image, showCrosshair=True)
        x,y,w,h = crop
        cropped = last_image[int(y): int(y+h), int(x): int(x+w)]
        encoded_image, buffer = cv2.imencode('.png', cropped)
        cropped_base64 = base64.b64encode(buffer).decode('utf-8')
        cropped_display.src_base64=cropped_base64
        cropped_image=cropped
        cv2.destroyAllWindows()

    crop_button=ft.ElevatedButton(text="✂Crop image", on_click=crop_image, top=440, right=920)
    cropped_display= ft.Image(crop_image, width=300,height=400, right=300, top=100)

    def save_image(e):
        global cropped_image
        name= sample_name_field.value.strip() 
        concentration = concentration_field.value.strip()
        if cropped_image is not None:
            pil_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
            pil_image.save(f"{name}, C= {concentration}.jpg")
        sample_data = pd.DataFrame({"Name":[name], "concentration":[concentration]})
        previous_data = pd.read_excel("TLC_data.xlsx")
        combined_tables=pd.concat([previous_data,sample_data])
        combined_tables.to_excel("TLC_data.xlsx", index=False)
        page.snack_bar = ft.SnackBar(content=ft.Text("✅ Sample saved successfully!"), open=True)
        page.update()

    save_button= ft.ElevatedButton(text="💾Save image",on_click=save_image, top=600, right=370)
    
    layout= ft.Stack(expand=True, 
                    controls=[title, camera_display, save_button, 
                             sample_name,sample_concentration, crop_button,notice_container,
                             cropped_display, home_button, CNN_button, normalization_button, message_container])

    page.add(layout)
    threading.Thread(target=live_feed, daemon=True).start()

def extract_info_from_filename(filename):
    info = re.search(r"(.+), C= ([\d.]+)\.jpg", filename)
    if info:
        name, concentration = info.groups()
        return name, float(concentration)
    return None, None

def load_regression_data(image_folder):
    concentrations = []
    intensities = []
    
    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg"):
            path = os.path.join(image_folder, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                inverted_img = 255 - img
                intensity = np.sum(inverted_img)
                
                name, conc = extract_info_from_filename(filename)
                if conc is not None:
                    concentrations.append(conc)
                    intensities.append(intensity)
    
    return np.array(concentrations), np.array(intensities)

def CNN(page):
    page.clean()
    page.title = "Polynomial Regression Analysis"
    page.scroll = ft.ScrollMode.ALWAYS
    
    folder_path = ""
    test_file_path = ""
    graph_container = ft.Column()
    results_container = ft.Column()
    coefficients = None
    model = None
    
    initial_text = ft.Text("Select folder with images for analysis", 
                            size=20, weight=ft.FontWeight.BOLD)
    
    folder_text = ft.Text("📂 No folder selected", size=14, italic=True)
    file_text = ft.Text("📂 No test file selected", size=14, italic=True)
    status_text = ft.Text("🔄 Waiting for action...", size=14)
    prediction_result = ft.Text("", size=16, weight=ft.FontWeight.BOLD, color=ft.colors.BLUE_800)
    
    def select_folder(e: ft.FilePickerResultEvent):
        nonlocal folder_path
        if e.path:
            folder_path = e.path
            folder_text.value = f"📂 Selected folder: {folder_path}"
            page.update()
    
    def select_test_file(e: ft.FilePickerResultEvent):
        nonlocal test_file_path
        if e.files:
            test_file_path = e.files[0].path
            file_text.value = f"📂 Selected file: {os.path.basename(test_file_path)}"
            page.update()
    
    folder_picker = ft.FilePicker(on_result=select_folder)
    file_picker = ft.FilePicker(on_result=select_test_file)
    page.overlay.extend([folder_picker, file_picker])
    
    def calculate_intensity(image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            inverted_img = 255 - img
            return np.sum(inverted_img)
        return None
    
    def predict_concentration(e):
        nonlocal coefficients, model
        
        if coefficients is None or model is None:
            prediction_result.value = "⚠️ First train the model with calibration images!"
            page.update()
            return
            
        if not test_file_path:
            prediction_result.value = "⚠️ No test file selected!"
            page.update()
            return
            
        try:
            intensity = calculate_intensity(test_file_path)
            if intensity is None:
                prediction_result.value = "❌ Error processing test image!"
                page.update()
                return
                
            # Solve quadratic equation to find C given I
            # coefficients[0]*C² + coefficients[1]*C + (coefficients[2] - intensity) = 0
            a, b, c = coefficients[0], coefficients[1], coefficients[2] - intensity
            
            discriminant = b**2 - 4*a*c
            if discriminant < 0:
                prediction_result.value = "❌ Intensity outside calibration range!"
                page.update()
                return
                
            concentration = (-b + np.sqrt(discriminant)) / (2*a)
            
            prediction_result.value = f"Predicted concentration: {concentration:.3f} mg/mL"
            page.update()
            
        except Exception as err:
            prediction_result.value = f"❌ Error: {str(err)}"
            page.update()
    
    def create_graph(x, y, coef):
        plt.figure(figsize=(8, 5))
        plt.scatter(x, y, color="blue", label="Experimental data")
        
        x_sorted = np.linspace(min(x), max(x), 500)
        y_model = np.poly1d(coef)(x_sorted)
        
        plt.plot(x_sorted, y_model, color="red", label="Polynomial model (degree 2)")
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
    
    def start_analysis(e):
        nonlocal graph_container, results_container, coefficients, model
    
        if not folder_path:
            status_text.value = "⚠️ Error: No folder selected!"
            page.update()
            return
    
        status_text.value = "📥 Loading data..."
        graph_container.controls = []
        results_container.controls = []
        page.update()
    
        try:
            # Load data correctly: x=concentrations, y=intensities
            x_real, y = load_regression_data(folder_path)
        
            if len(x_real) == 0:
                status_text.value = "⚠️ Error: No valid images found!"
                page.update()
                return
        
            status_text.value = "⚙️ Calculating regression..."
            page.update()
        
            # Correct polynomial regression: y(intensity) = f(x(concentration))
            coefficients = np.polyfit(x_real, y, 2)
            model = np.poly1d(coefficients)
        
            # Predict intensities for known concentrations
            y_pred = model(x_real)
        
            # Calculate predicted concentrations (solving quadratic equation)
            x_pred = []
            for intensity in y:
                a, b, c = coefficients[0], coefficients[1], coefficients[2] - intensity
                discriminant = b**2 - 4*a*c
                if discriminant >= 0:
                    concentration = (-b + np.sqrt(discriminant)) / (2*a)
                    x_pred.append(concentration)
                else:
                    x_pred.append(np.nan)  # Marker for out-of-range values
        
            # Calculate MAE for concentrations
            x_pred_valid = [x for x in x_pred if not np.isnan(x)]
            x_real_valid = [x_real[i] for i in range(len(x_pred)) if not np.isnan(x_pred[i])]
        
            if len(x_pred_valid) > 0:
                mae_conc = mean_absolute_error(x_real_valid, x_pred_valid)
            else:
                mae_conc = float('nan')
        
            # Metrics for intensities (optional - can be removed if not needed)
            r2 = r2_score(y, y_pred)
        
            # Create UI elements
            equation = ft.Container(
                content=ft.Text(
                    f"Model: Intensity = {coefficients[0]:.4e}·C² + {coefficients[1]:.4e}·C + {coefficients[2]:.4f}",
                    size=16,
                    weight=ft.FontWeight.BOLD
                ),
                padding=10
            )
        
            metrics = ft.Container(
                content=ft.Column([
                    ft.Text("📊 Metrics:", weight=ft.FontWeight.BOLD),
                    ft.Text(f"• MAE (concentration): {mae_conc:.4f} mg/mL"),
                    ft.Text(f"• R²: {r2:.4f}")
                ]),
                padding=10
            )
        
            table = ft.DataTable(
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
        
            img_base64 = create_graph(x_real, y, coefficients)
            graph = ft.Image(
                src_base64=img_base64,
                width=600,
                height=400,
                fit=ft.ImageFit.CONTAIN
            )
        
            results_container.controls = [
                ft.Card(
                    content=ft.Container(
                        content=ft.Column([
                            equation,
                            metrics,
                            ft.Text("📋 Results:", weight=ft.FontWeight.BOLD),
                            table
                        ]),
                        padding=20
                    ),
                    width=800
                )
            ]
        
            graph_container.controls = [
                ft.Card(
                    content=ft.Container(
                        content=graph,
                        padding=20
                    )
                )
            ]
        
            status_text.value = "✅ Analysis completed successfully!"
        
        except Exception as err:
            status_text.value = f"❌ Error: {str(err)}"
            results_container.controls = [
                ft.Text(f"Error details: {str(err)}", color="red")
            ]
    
        page.update()
    
    page.add(
        ft.Column([
            initial_text,
            ft.Row([
                ft.ElevatedButton("Select Folder", 
                                 on_click=lambda e: folder_picker.get_directory_path()),
                folder_text
            ], alignment=ft.MainAxisAlignment.CENTER),
            
            ft.ElevatedButton("Start Analysis", 
                             on_click=start_analysis,
                             icon=ft.icons.ANALYTICS),
            
            ft.Divider(height=20),
            
            ft.Text("Test new sample:", size=16, weight=ft.FontWeight.BOLD),
            ft.Row([
                ft.ElevatedButton("Select Test Image",
                                 on_click=lambda e: file_picker.pick_files(allowed_extensions=["jpg", "jpeg", "png"])),
                file_text
            ], alignment=ft.MainAxisAlignment.CENTER),
            
            ft.ElevatedButton("Predict Concentration",
                            on_click=predict_concentration,
                            icon=ft.icons.CALCULATE),
            
            prediction_result,
            
            status_text,
            results_container,
            graph_container,
            ft.ElevatedButton("⬅ Back to home", 
                             on_click=lambda e: go_to_page(e.page, home_page))
        ], 
        scroll=ft.ScrollMode.ALWAYS,
        spacing=20)
    )
    page.update()

def normalization(page):
    page.title = "Chromatography Analysis - Flet"
    page.scroll = ft.ScrollMode.ALWAYS

    status_text = ft.Text("🔄 Waiting for action...")
    image_display = ft.Image(width=300, height=300, fit=ft.ImageFit.CONTAIN)

    file_picker = ft.FilePicker(on_result=lambda e: load_image(e, image_display, status_text))
    page.overlay.append(file_picker)

    load_button = ft.ElevatedButton("📂 Load Image", on_click=lambda _: file_picker.pick_files(allowed_extensions=["png", "jpg", "jpeg"]))
    contours_button = ft.ElevatedButton("🔍 Show Contours", on_click=lambda e: show_contours(e, image_display, status_text))
    global graph_container
    graph_container = ft.Column([])

    graph_button = ft.ElevatedButton("📊 Show Graph", on_click=show_graph)
    home_button = ft.ElevatedButton(text="Back to home",
                                        on_click=lambda e: go_to_page(e.page, home_page))

    page.add(
        ft.Column([
            load_button,
            image_display,
            contours_button,
            graph_button,
            status_text,
            home_button
        ], alignment=ft.MainAxisAlignment.CENTER)
    )

def main(page: ft.Page):
    home_page(page)

ft.app(target=main)