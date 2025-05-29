import os
from diffusers.utils import load_image
from diffusers import FluxTransformer2DModel
from pipeline_flux_control_removal import FluxControlRemovalPipeline
import torch


# ========== 设置路径和提示 ==========
name = 'input2.png'
images_path = 'example/image'
masks_path = 'example/mask'
outputs_path = 'example/output'
os.makedirs(outputs_path, exist_ok=True)
prompt = 'There is nothing here.'

image_path = os.path.join(images_path, name)
mask_path = os.path.join(masks_path, name)
output_path = os.path.join(outputs_path, name)
size = (1024, 1024)

# ========== 图像加载 ==========
image = load_image(image_path)
original_size = image.size
image = image.convert("RGB").resize(size)
mask = load_image(mask_path).convert("RGB").resize(size)
generator = torch.Generator(device="cuda").manual_seed(24)

image_path = os.path.join(images_path, name)
mask_path = os.path.join(masks_path, name)
output_path = os.path.join(outputs_path, name)
size = (1024, 1024)

# ==== 自己训的 transformer ====
transformer = FluxTransformer2DModel.from_pretrained(
    "./transformer",
    torch_dtype=torch.bfloat16,
)

expanded_mlp = torch.nn.Linear(256, 3072, bias=True).to(torch.bfloat16).to("cuda")

expanded_mlp.load_state_dict(torch.load("./expanded_x_embedder.pth"))
transformer.x_embedder = expanded_mlp
transformer.register_to_config(in_channels=transformer.config.in_channels * 4)
# ==== 从官方模型加载其余组件，然后替换 transformer ====
pipe = FluxControlRemovalPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    transformer=transformer,
    torch_dtype=torch.bfloat16
).to("cuda")

pipe.transformer.to(torch.bfloat16)

# ========== 推理 ==========
result = pipe(
    prompt=prompt,
    control_image=image,
    control_mask=mask,
    num_inference_steps=28,
    guidance_scale=3.5,
    generator=generator,
    max_sequence_length=512,
    height=size[1],
    width=size[0],
).images[0]

result.resize(original_size).save(output_path)
print("✅ Successfully inpainted image after merging LoRA.")