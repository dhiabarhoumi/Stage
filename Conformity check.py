from ultralytics import YOLO
import cv2
import pandas as pd
from datetime import datetime

# Load the trained YOLO model
model = YOLO("C:/Users/dhiab/OneDrive/Bureau/Stage/Training/best.pt")

# Create a video capture object for webcam
video_capture = cv2.VideoCapture(0)

# Create a dataframe to store box information
box_data = pd.DataFrame(columns=['Class', 'Conformity', 'Timestamp'])

# Define the required number of goussets and cornieres for conformity
required_goussets = 4
required_cornieres = 2

# Loop through frames from the webcam
while True:
    # Read the current frame
    ret, frame = video_capture.read()
    
    # Perform object detection using the YOLO model
    results = model(frame)
    print(results)
    


    # Check if any detections are present
    if len(results) == 0:
        continue
    
    for detection in results:
        # Get the detected objects and their class labels
        bboxes = detection.boxes.numpy()
        confidences = detection.probs.numpy()
        class_labels = detection.keypoints.numpy().astype(int)
        
        # Initialize counters for goussets and cornieres
        num_goussets = 0
        num_cornieres = 0
        
        # Iterate over the detected objects
        for bbox, confidence, class_label in zip(bboxes, confidences, class_labels):
            if confidence > 0.5:  # Adjust the confidence threshold as needed
                label = results.names[int(class_label)]
                
                if label == 'gousset':
                    num_goussets += 1
                elif label == 'corniere':
                    num_cornieres += 1
        
        # Determine the conformity based on the number of goussets and cornieres
        if num_goussets == required_goussets and num_cornieres == required_cornieres:
            conformity = 'Conform'
        else:
            conformity = 'Non-Conform'
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Add the classification result to the dataframe
        box_data = box_data.append({'Class': 'SBM Box', 'Conformity': conformity, 'Timestamp': timestamp},
                                   ignore_index=True)
    
    # Display the frame with bounding boxes and class labels
    cv2.imshow("Detection Results", frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the windows
video_capture.release()
cv2.destroyAllWindows()

# Generate reports based on the box_data dataframe
# Report 1: Quantity Tested
quantity_tested = box_data.shape[0]
print("Quantity Tested: ", quantity_tested)

# Report 2: Conformity Ratios
conformity_ratios = box_data['Conformity'].value_counts(normalize=True)
print("Conformity Ratios:")
print(conformity_ratios)

# Report 3: Generate Excel report
box_data.to_excel("report.xlsx", index=False)
print("Excel report generated: report.xlsx")
#adapt this code to yolov8



















