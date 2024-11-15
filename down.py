# from transformers import AutoProcessor, AutoModelForCausalLM

# # Specify the local directory where you want to save the model
# local_dir = "/Users/charan/VSCode/GITHUB/Vison_Encoder/models/blip-image-captioning-base"

# # Download and save the processor
# processor = AutoProcessor.from_pretrained("microsoft/git-large-vatex")
# processor.save_pretrained(local_dir)

# # Download and save the model
# model = AutoModelForCausalLM.from_pretrained("microsoft/git-large-vatex")
# model.save_pretrained(local_dir)


# import cv2
# import torch
# from PIL import Image
# from transformers import BlipProcessor, BlipForConditionalGeneration

# # Load the processor and model
# processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# processor.save_pretrained(local_dir)
# model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
# model.save_pretrained(local_dir)



# from transformers import CLIPProcessor, CLIPModel

# # Specify the local directory where you want to save the model
# local_model_dir = "/Users/charan/VSCode/GITHUB/Vison_Encoder/models/openai_clip"

# # Download and save the processor
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
# processor.save_pretrained(local_model_dir)

# # Download and save the model
# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# model.save_pretrained(local_model_dir)

from transformers import AutoTokenizer, AutoModelForCausalLM

# Define the model name
model_name = "anilbhatt1/phi2-proj-offset-peft-model"

# Define the local directory where the model will be saved
local_model_dir = "/Users/charan/VSCode/GITHUB/Vison_Encoder/models/phi2-proj-offset-peft-model"  # Update the path as needed

# Create the local directory if it doesn't exist
# import os
# os.makedirs(local_model_dir, exist_ok=True)

# Load and save the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
tokenizer.save_pretrained(local_model_dir)

# Load and save the model
model = AutoModelForCausalLM.from_pretrained(model_name)
model.save_pretrained(local_model_dir)

print(f"Model and tokenizer have been saved to {local_model_dir}")

# def load_models():
#     # Load the processor and model
def load_models():
    # Load the tokenizer and model from the local directory
    tokenizer = AutoTokenizer.from_pretrained(local_model_dir, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(local_model_dir
