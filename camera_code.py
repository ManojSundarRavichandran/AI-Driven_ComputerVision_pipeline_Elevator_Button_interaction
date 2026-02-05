import cv2
import matplotlib.pyplot as plt

def take_photos():
    # Camera initialization
    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return
    
    # Take a single photo automatically
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
    else:
        photo_filename = 'cameraphoto.jpg'

        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        print(frame.shape)

        cv2.imwrite(photo_filename, frame)
        print(f"Photo saved as {photo_filename}")
    
    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

    # Display the photo using matplotlib
    img = cv2.imread('cameraphoto.jpg')
    if img is not None:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Captured Photo')
        plt.axis('off')
        plt.show()




