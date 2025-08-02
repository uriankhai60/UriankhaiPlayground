import torch
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig, FluxTransformer2DModel, FluxPipeline
from transformers import BitsAndBytesConfig as BitsAndBytesConfig, T5EncoderModel

repo = "black-forest-labs/FLUX.1-Krea-dev"
quant_config1 = BitsAndBytesConfig(load_in_8bit=True)
quant_config2 = DiffusersBitsAndBytesConfig(load_in_8bit=True)

text_encoder_8bit = T5EncoderModel.from_pretrained(
    repo,
    subfolder="text_encoder_2",
    quantization_config=quant_config1,
    torch_dtype=torch.float16,
)

transformer_8bit = FluxTransformer2DModel.from_pretrained(
    repo,
    subfolder="transformer",
    quantization_config=quant_config2,
    torch_dtype=torch.float16,
)

pipe = FluxPipeline.from_pretrained(
    repo,
    text_encoder_2=text_encoder_8bit,
    transformer=transformer_8bit,
    torch_dtype=torch.float16,
    device_map="balanced",
)

prompt = "A frog holding a sign that says hello world"
image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=4.5,
).images[0]
image.save("flux-krea-dev.png")
