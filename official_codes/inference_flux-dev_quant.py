import torch
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig, FluxTransformer2DModel, FluxPipeline
from transformers import BitsAndBytesConfig as BitsAndBytesConfig, T5EncoderModel
import time

quant_config = BitsAndBytesConfig(load_in_8bit=True)

text_encoder_8bit = T5EncoderModel.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    subfolder="text_encoder_2",
    quantization_config=quant_config,
    torch_dtype=torch.float16,
)

quant_config = DiffusersBitsAndBytesConfig(load_in_8bit=True)
transformer_8bit = FluxTransformer2DModel.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    subfolder="transformer",
    quantization_config=quant_config,
    torch_dtype=torch.float16,
)

pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    text_encoder_2=text_encoder_8bit,
    transformer=transformer_8bit,
    torch_dtype=torch.float16,
    device_map="balanced",
)

prompt = "a tiny astronaut hatching from an egg on the moon"
start_time = time.time()
image = pipeline(prompt, 
                 guidance_scale=3.5, 
                 height=1024, 
                 width=1024, 
                 num_inference_steps=50).images[0]
inferece_time = time.time() - start_time

print(f"inference_time: {inferece_time:.2f}", flush=True)

image.save("flux.png")


'''
inference_size: (1024, 1024)
vram_cost: under 22GB
time_cost: 30.18 sec (50step)
'''