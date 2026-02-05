from button_detection import ocr_on_bounding_boxes

# Path to the image you want to analyze
image_path = '/media/asmany/Drive_D/Intelligent_Robotics/test_image/2.jpeg'  # Replace with the path to your image

print('Which floor do you want to go?')
floor_number = input()

# Perform OCR on the bounding boxes in the image
coordinates = ocr_on_bounding_boxes(image_path, floor_number)


