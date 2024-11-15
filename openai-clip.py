import os
import cv2
import torch
import torch.nn as nn
from PIL import Image
from transformers import (
    CLIPProcessor,
    CLIPModel,
    AutoTokenizer,
    AutoModelForCausalLM
)

# ==========================
# 1. Environment and Device Setup
# ==========================

# Set Transformers to work in offline mode if necessary
# os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Device configuration: Use GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================
# 2. Model Loading
# ==========================

# Paths to local models (update these paths as per your setup)
# If working offline, ensure the models are downloaded and specify the local paths
# Otherwise, you can use Hugging Face model identifiers directly

# CLIP Model and Processor
clip_model_name = "/Users/charan/VSCode/GITHUB/Vison_Encoder/models/openai_clip"  # Use local path if offline
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
clip_model = CLIPModel.from_pretrained(clip_model_name)
clip_model.to(device)
clip_model.eval()  # Set to evaluation mode
print("CLIP model loaded successfully.")

# Phi2 Model and Tokenizer
phi2_model_name = "anilbhatt1/phi2-proj-offset-peft-model"  # Replace with your model path if different
tokenizer = AutoTokenizer.from_pretrained(phi2_model_name, use_fast=False)
phi2_model = AutoModelForCausalLM.from_pretrained(phi2_model_name)
phi2_model.to(device)
phi2_model.eval()  # Set to evaluation mode
print("Phi2 model loaded successfully.")

# Optional: If you have a separate projection model, load it here
# For example, if your projection model is a simple linear layer saved as a .pth file
# projection_model_path = "/path/to/projection_model.pth"
# projection_model = torch.load(projection_model_path)
# projection_model.to(device)
# projection_model.eval()
# print("Projection model loaded successfully.")

# ==========================
# 3. Define Projection Layer (If Needed)
# ==========================

# If your projection model is a separate linear layer, define it here
# Update input and output dimensions based on your models

# Example projection model (adjust dimensions as per your setup)
# class ProjectionModel(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(ProjectionModel, self).__init__()
#         self.linear = nn.Linear(input_dim, output_dim)
#
#     def forward(self, x):
#         return self.linear(x)
#
# # Initialize projection model
# input_dim = 512  # Example: CLIP's embedding dimension for ViT-B/32 is 512
# output_dim = 2560  # Example: Phi2's embedding dimension
# projection_model = ProjectionModel(input_dim, output_dim)
# # Load trained weights
# projection_model.load_state_dict(torch.load(projection_model_path))
# projection_model.to(device)
# projection_model.eval()
# print("Projection model initialized.")

# ==========================
# 4. Webcam Initialization
# ==========================

# Initialize video capture from the default webcam (index 0)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot access the webcam.")
    exit()

print("Webcam initialized successfully.")

# ==========================
# 5. Caption Generation Function
# ==========================

def generate_caption(image_pil, clip_processor, clip_model, phi2_model, tokenizer, device):
    """
    Generates a descriptive caption for the given PIL image using CLIP and phi2 models.
    """
    # Preprocess the image for CLIP
    inputs = clip_processor(images=image_pil, return_tensors="pt").to(device)

    with torch.no_grad():
        # Obtain image embeddings from CLIP
        clip_outputs = clip_model.get_image_features(**inputs)  # Shape: [1, 512]
        clip_embeddings = clip_outputs / clip_outputs.norm(p=2, dim=-1, keepdim=True)  # Normalize

    # If using a separate projection model, project the embeddings
    # projected_embeddings = projection_model(clip_embeddings)  # Shape: [1, 2560]

    # Prepare the prompt for the language model
    prompt = "Describe the image:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)  # Shape: [1, seq_length]

    # Generate caption using phi2 model
    # Depending on your phi2 model's architecture, you might need to adjust the inputs
    # Here, we assume the phi2 model can generate text based on a text prompt
    with torch.no_grad():
        output_sequences = phi2_model.generate(
            input_ids=input_ids,
            # If your phi2 model accepts additional inputs like image embeddings, include them here
            # For example:
            # encoder_hidden_states=projected_embeddings.unsqueeze(1),
            max_new_tokens=50,  # Adjust based on desired caption length
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True
        )

    # Decode the generated tokens to text
    generated_caption = tokenizer.decode(output_sequences[0], skip_special_tokens=True)

    return generated_caption

# ==========================
# 6. Main Loop for Real-Time Captioning
# ==========================

print("Starting real-time image captioning. Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Convert the frame from BGR (OpenCV format) to RGB (PIL format)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

    try:
        # Generate caption for the current frame
        caption = generate_caption(pil_image, clip_processor, clip_model, phi2_model, tokenizer, device)
    except Exception as e:
        print(f"Error during caption generation: {e}")
        caption = "Error generating caption."

    # Overlay the generated caption on the frame
    # Position: (10, 30) - adjust as needed
    # Font: FONT_HERSHEY_SIMPLEX
    # Scale: 0.7
    # Color: Green (0, 255, 0)
    # Thickness: 2
    cv2.putText(
        frame,
        f"Caption: {caption}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    # Display the frame with the caption
    cv2.imshow("Webcam - Press 'q' to Exit", frame)

    # Exit condition: Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Exiting...")
        break

# ==========================
# 7. Resource Cleanup
# ==========================

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
print("Resources released and program terminated.")
