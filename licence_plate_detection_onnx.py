import cv2
import numpy as np
import onnxruntime as ort

# 1. Load the Model
model_path = "/home/medprime/Music/ANPR/rfdetr_alpr_int8.onnx"
session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape[2:] 

# 2. Setup Video Capture and Writer
video_path = "/home/medprime/Music/ANPR/anpr-demo-video.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties for saving
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define codec and create VideoWriter object
output_path = "/home/medprime/Music/ANPR/output_detected.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

def preprocess(frame, target_size):
    img = cv2.resize(frame, target_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 3. Processing Loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    input_tensor = preprocess(frame, input_shape)
    outputs = session.run(None, {input_name: input_tensor})
    
    # Swapped logic: based on your log, outputs[0] is scores, outputs[1] is boxes
    raw_logits = outputs[0][0] # Shape (300, 1)
    raw_boxes = outputs[1][0]  # Shape (300, 4)

    for i in range(len(raw_logits)):
        # Convert Logit to Confidence %
        score = sigmoid(raw_logits[i][0])
        
        if score > 0.45:  # Confidence Threshold
            box = raw_boxes[i]
            # RF-DETR center-based normalized format [cx, cy, w, h]
            cx, cy, bw, bh = box
            
            x1 = int((cx - bw/2) * frame_width)
            y1 = int((cy - bh/2) * frame_height)
            x2 = int((cx + bw/2) * frame_width)
            y2 = int((cy + bh/2) * frame_height)

            # Draw
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Plate: {score:.2f}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Save the frame to the output video
    out.write(frame)
    
    cv2.imshow("RF-DETR License Plate Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Video saved to: {output_path}")