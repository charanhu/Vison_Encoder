import cv2
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load the processor and model
processor = BlipProcessor.from_pretrained("/Users/charan/VSCode/GITHUB/Vison_Encoder/models/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("/Users/charan/VSCode/GITHUB/Vison_Encoder/models/blip-image-captioning-base")

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initialize video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB and PIL Image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

    # Prepare inputs
    inputs = processor(pil_image, return_tensors="pt").to(device)

    # Generate caption
    with torch.no_grad():
        generated_ids = model.generate(**inputs)
        caption = processor.decode(generated_ids[0], skip_special_tokens=True)

    # Print caption
    print("Caption:", caption)

    # Display the frame
    cv2.imshow('Webcam', frame)

    # Exit loop on 'q' press
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
