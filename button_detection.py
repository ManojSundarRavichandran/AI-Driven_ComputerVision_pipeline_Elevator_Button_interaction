import os
import re
import torch
import cv2
import pytesseract
import pandas as pd
import numpy as np
from pytesseract import Output
from PIL import Image, ImageOps
from pressed_non_pressed_detection import load_model, predict

# Load the pressed/non-pressed model
weights_path = 'Pressed_NonPressed_model/best_model_weights.pth'
pressed_model = load_model(weights_path)

# Set the TESSDATA_PREFIX environment variable
os.environ['TESSDATA_PREFIX'] = '/usr/share/tesseract-ocr/4.00/tessdata/'

pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

tessdata_prefix = os.environ.get('TESSDATA_PREFIX')
if not tessdata_prefix or not os.path.exists(tessdata_prefix):
    raise EnvironmentError(f"TESSDATA_PREFIX is not set correctly or the directory does not exist: {tessdata_prefix}")

# Load YOLOv5 model
yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp3/weights/best.pt')

def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply thresholding
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply morphological operations to reduce noise
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    return morph



# Function to perform OCR on the bounding boxes
def ocr_on_bounding_boxes(image_path, floor_number):
    # Perform object detection
    results = yolo_model(image_path)
    df = pd.DataFrame(results.pandas().xyxy[0])

    # Load image
    image = cv2.imread(image_path)

    # Check if image is loaded
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    for index, row in df.iterrows():
        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])

        # Crop the ROI
        roi = image[ymin:ymax, xmin:xmax]

        # Convert ROI to PIL image
        roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

        # Save the ROI image
        roi_pil.save(f'roi_{index}.png')

        # Preprocess the ROI image
        processed_image = preprocess_image(roi)

        # Convert processed image to PIL format
        pil_image = Image.fromarray(processed_image)

        inverted_image = ImageOps.invert(pil_image)

        pil_image.save(f'pil_image_{index}.png')
        inverted_image.save(f'inverted_image_{index}.png')

        # Perform OCR
        text = pytesseract.image_to_data(inverted_image, lang='eng', output_type=Output.DICT, config='--psm 10')

        non_empty_text = [t for t in text['text'] if t.strip() != '']
        non_empty_text = [t.replace('C', '').replace('c', '') for t in non_empty_text]
        processed_text = [t.replace('(', '', 1).replace(')', '', 1) if t.startswith('(') else t for t in non_empty_text]

        # print(processed_text)



        for text in processed_text:
            numbers = re.findall(r'\d+', text)
            alphabets = re.findall(r'[a-zA-Z]+', text)

            # print(alphabets)

            if numbers or alphabets:
                
                if floor_number in numbers + alphabets:
                    
                    prediction = predict(roi_pil, pressed_model)
                    # print(prediction)
                    # print(prediction)
                    if prediction > 0.5:
                        print(f"Button {floor_number} found on position {xmin, ymin, xmax, ymax}")
                        print('Button is already pressed')
                        # return xmin, ymin, xmax, ymax
                        # return None
                    else:
                        print(f"Button {floor_number} found on position {xmin, ymin, xmax, ymax}")
                        print('Pressing the button')

                        return xmin, ymin, xmax, ymax

                    break

    return None
                   
                

    # Print the detected text for each button
    # print(f"{processed_text} found on position {xmin, ymin, xmax, ymax}")
