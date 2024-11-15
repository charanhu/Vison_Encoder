import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

# Specify the local directory where the model is saved
local_dir = "/Users/charan/VSCode/GITHUB/Vison_Encoder/models/git-large-vatex"

# Load the processor and model from the local directory
processor = AutoProcessor.from_pretrained(local_dir)
model = AutoModelForCausalLM.from_pretrained(local_dir)

# Set the device (use 'mps' for Apple Silicon, 'cpu' otherwise)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model.to(device)

# Print the number of temporal embeddings
print("Number of temporal embeddings:", model.config.num_image_with_embedding)

# Initialize video capture from the default webcam
cap = cv2.VideoCapture(0)

# Frame buffer to hold a sequence of frames
frame_buffer = []
buffer_size = 1  # Set buffer_size to 6

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB and then to PIL Image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

    # Append the frame to the buffer
    frame_buffer.append(pil_image)

    # Display the frame
    cv2.imshow('Webcam', frame)

    # If the buffer is full, process the frames
    if len(frame_buffer) == buffer_size:
        # Prepare the inputs for the model
        pixel_values = processor(images=frame_buffer, return_tensors="pt").pixel_values.to(device)

        # Generate the caption
        with torch.no_grad():
            generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
            generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Print the generated caption
        print("Description:", generated_caption)

        # Clear the buffer
        frame_buffer = []

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
