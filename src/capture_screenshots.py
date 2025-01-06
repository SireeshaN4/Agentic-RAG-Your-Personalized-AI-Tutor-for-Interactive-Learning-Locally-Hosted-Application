import cv2
import os
from fpdf import FPDF

def capture_screenshots(output_dir, num_screenshots=5, interval=5):
    """Capture screenshots from the webcam and save them as images."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cap = cv2.VideoCapture(0)  # Open the webcam (0 for default camera)
    if not cap.isOpened():
        print("Error: Could not open video device.")
        return
    
    print("Press 's' to capture a screenshot or 'q' to quit.")
    screenshot_count = 0
    
    while screenshot_count < num_screenshots:
        ret, frame = cap.read()  # Capture a frame
        if not ret:
            print("Error: Could not read frame.")
            break
        
        cv2.imshow('Video', frame)  # Display the live webcam feed
        
        key = cv2.waitKey(interval * 1000) & 0xFF  # Wait for a key press for the interval duration
        if key == ord('s'):  # If the 's' key is pressed, save the screenshot
            screenshot_path = os.path.join(output_dir, f'screenshot_{screenshot_count + 1}.png')
            cv2.imwrite(screenshot_path, frame)
            print(f"Screenshot saved: {screenshot_path}")
            screenshot_count += 1
        elif key == ord('q'):  # If the 'q' key is pressed, quit the loop
            break
    
    cap.release()  # Release the webcam
    cv2.destroyAllWindows()  # Close OpenCV windows

def create_pdf_from_images(image_dir, output_pdf):
    """Combine images into a single PDF."""
    pdf = FPDF()
    for image_file in sorted(os.listdir(image_dir)):  # Sort files alphabetically
        if image_file.endswith('.png'):
            image_path = os.path.join(image_dir, image_file)
            pdf.add_page()  # Add a new page for each image
            pdf.image(image_path, x=10, y=10, w=190)  # Add the image to the PDF
    
    pdf.output(output_pdf)  # Save the PDF
    print(f"PDF created: {output_pdf}")

if __name__ == "__main__":
    output_dir = "screenshots"  # Directory to save screenshots
    output_pdf = "screenshots.pdf"  # Output PDF file
    
    capture_screenshots(output_dir, num_screenshots=5, interval=5)  # Capture 5 screenshots
    create_pdf_from_images(output_dir, output_pdf)  # Convert screenshots to PDF
