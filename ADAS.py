import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog
from tkinter import ttk
from tkinter import ttk
import tkinter
from PIL import Image, ImageTk
import pyttsx3

engine = pyttsx3.init()

# Initialize the main GUI window
root = Tk()
root.title("Video Player with Object and Lane Detection")
root.geometry("800x600")
root.configure(background='#333333')
root.resizable(False, False)

# Load the background image
bg_image = Image.open("D:\\SLTC\\SLTC 5th sem\\TCC1\\Group02_Vision based driver assits system\\car.png")  # Replace with your image file
bg_photo = ImageTk.PhotoImage(bg_image)

# Create a background label
bg_label = Label(root, image=bg_photo)
bg_label.place(relwidth=1, relheight=1)  # Cover the entire window

# Initialize video capture object and flags for the current screen
cap = None
current_screen = "main"

# Initialize cascade classifiers (use your XML files)
stop_sign_cascade = cv2.CascadeClassifier('D:\\SLTC\\SLTC 5th sem\\TCC1\\Group02_Vision based driver assits system\\Cascades\\Stopsign_HAAR_19Stages.xml')
pedestrian_cascade = cv2.CascadeClassifier('D:\\SLTC\\SLTC 5th sem\\TCC1\\Group02_Vision based driver assits system\\Cascades\\pedestrian.xml')
traffic_light_cascade = cv2.CascadeClassifier('D:\\SLTC\\SLTC 5th sem\\TCC1\\Group02_Vision based driver assits system\\Cascades\\TrafficLight_HAAR_16Stages.xml')
bumpy_cascade = cv2.CascadeClassifier('D:\\SLTC\\SLTC 5th sem\\TCC1\\Group02_Vision based driver assits system\\Cascades\\bumper.xml')
# Add more classifiers as needed

# Add a topic label
topic_label = tkinter.Label(root, text="Real-Time Video Streaming and Object & Lane Detection", font=('calibri', 18, 'bold'), background='#ADD8E6')
topic_label.pack(side=tkinter.TOP, pady=20)

# Create a function to open a video file
def AddVideo():
    global cap
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4")])
    if file_path:
        # Release any previous video capture
        if cap:
            cap.release()
        cap = cv2.VideoCapture(file_path)
        switch_screen("video")

def OpenWebcam():
    global cap
    # Release any previous video capture
    if cap:
        cap.release()
    cap = cv2.VideoCapture(0)  # 0 represents the default camera (usually the built-in webcam)
    switch_screen("video")

# Function to detect stop signs and draw rectangles
def detect_objects(frame):
    # Perform object detection (example: stop signs)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    stop_signs = stop_sign_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    pedestrian = pedestrian_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    traffic_lights = traffic_light_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    bumpy = bumpy_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around detected objects
    for (x, y, w, h) in stop_signs:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red rectangle for stop signs
        text_y = y + h + 20  # Adjust the y-coordinate to be below the rectangle
        cv2.putText(frame, "Stop Sign", (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        engine.say("stop")
        engine.runAndWait()
    
    for (x, y, w, h) in pedestrian:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red rectangle for stop signs
        text_y = y + h + 20  # Adjust the y-coordinate to be below the rectangle
        cv2.putText(frame, "Pedestrian", (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
    for (x, y, w, h) in traffic_lights:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red rectangle for stop signs
        text_y = y + h + 20  # Adjust the y-coordinate to be below the rectangle
        cv2.putText(frame, "Traffic Lights", (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
    
    for (x, y, w, h) in bumpy:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red rectangle for stop signs
        text_y = y + h + 20  # Adjust the y-coordinate to be below the rectangle
        cv2.putText(frame, "Bumpy", (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Add more object detection code for other classifiers as needed

    return frame

# Function to detect main lanes 
def detect_main_lanes(frame):
    # (Insert the lane detection code here)
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Apply Canny edge detection with adjusted thresholds
    edges = cv2.Canny(blurred, 50, 150)

    # Define a region of interest
    height, width = frame.shape[:2]
    roi_vertices = [
        (0, height),
        (width // 2, height // 2),
        (width, height),
    ]
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, [np.array(roi_vertices, np.int32)], 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Use Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(
        masked_edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=50
    )

    if lines is not None:
        # Sort the lines by their length
        lines = sorted(lines, key=lambda line: np.linalg.norm(line[0][0:2] - line[0][2:4]), reverse=True)

        # Initialize lists to store left and right lane lines
        left_lines = []
        right_lines = []

        # Separate lines into left and right based on slope
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1)
            if slope < -0.2:
                left_lines.append(line)
            elif slope > 0.2:
                right_lines.append(line)

        # Draw the two longest lines on each side (main lanes)
        for lines in [left_lines, right_lines]:
            if lines:
                longest_line = max(lines, key=lambda line: np.linalg.norm(line[0][0:2] - line[0][2:4]))
                x1, y1, x2, y2 = longest_line[0]
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

    return frame

# Function to switch screens
def switch_screen(screen):
    global current_screen
    current_screen = screen
    if screen == "main":
        add_button.pack()
        open_webcam_button.pack()
        stop_button.pack_forget()
        label.pack_forget()
    elif screen == "video":
        add_button.pack_forget()
        open_webcam_button.pack_forget()
        stop_button.pack()
        label.pack()
        PlayVideo()

# Function to play the video
def PlayVideo():
    def update():
        ret, frame = cap.read()
        if ret:
            frame_with_objects = detect_objects(frame)
            frame_with_objects = detect_main_lanes(frame_with_objects)  # Perform lane detection
            frame_with_objects = cv2.cvtColor(frame_with_objects, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(image=Image.fromarray(frame_with_objects))
            label.config(image=photo)
            label.image = photo
            label.after(10, update)
        else:
            cap.release()
            switch_screen("main")  # Return to the main screen when video ends or webcam is stopped

    update()

# Function to stop video playback
def StopVideo():
    global cap
    if cap:
        cap.release()
        switch_screen("main")  # Return to the main screen when stopped

# Create a uniform button style
button_style = {"font": ("calibri", 14), "background": "#3498db", "foreground": "#FFFFFF", "width": 20, "height": 2}

# Create labels for buttons
add_label = Label(root, text="Open Video File", font=("calibri", 12), bg="#333333", fg="#FFFFFF")
webcam_label = Label(root, text="Use Webcam", font=("calibri", 12), bg="#333333", fg="#FFFFFF")

# Create buttons with custom styles
button_style = {"font": ("calibri", 12), "background": "#800000", "foreground": "#FFFFFF", "width": 18, "height": 2}
add_button = Button(root, text="Add Video", command=AddVideo, **button_style)
open_webcam_button = Button(root, text="Open Webcam", command=OpenWebcam, **button_style)
add_button.pack(side=LEFT, padx=10, pady=20)
open_webcam_button.pack(side=LEFT, padx=10, pady=20)

stop_button = Button(root, text="Stop", command=StopVideo, **button_style)
stop_button.pack(side=LEFT, padx=10, pady=20)

# Create a label to display the video
label = Label(root)
label.pack()

# Initially show the main screen
switch_screen("main")

# Execute Tkinter
root.mainloop()
