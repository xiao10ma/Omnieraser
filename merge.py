import os
import torch
from diffusers import FluxTransformer2DModel
from diffusers.utils import load_image
from pipeline_flux_control_removal import FluxControlRemovalPipeline
import torch.nn as nn
import copy

def manually_merge_lora(module: nn.Module, verbose=True):
    for name, submodule in module.named_children():
        # 递归处理子模块
        manually_merge_lora(submodule, verbose=verbose)

        if hasattr(submodule, "lora_A") and hasattr(submodule, "lora_B"):
            if verbose:
                print(f"Merging LoRA into: {name} ({type(submodule)})")

            with torch.no_grad():
                weight = submodule.base_layer.weight
                scaling = getattr(submodule, "scaling", 1.0)

                for key in submodule.lora_A.keys():
                    A = submodule.lora_A[key].weight  # [r, in]
                    B = submodule.lora_B[key].weight  # [out, r]

                    s = scaling[key] if isinstance(scaling, dict) else scaling
                    lora_weight = (B @ A) * s

                    weight += lora_weight

            # 删除 adapter 层
            del submodule.lora_A
            del submodule.lora_B
            if hasattr(submodule, "lora_dropout"):
                del submodule.lora_dropout
            if hasattr(submodule, "lora_embedding_A"):
                del submodule.lora_embedding_A
            if hasattr(submodule, "lora_embedding_B"):
                del submodule.lora_embedding_B
            if hasattr(submodule, "lora_magnitude_vector"):
                del submodule.lora_magnitude_vector

def replace_lora_linear_with_linear(module):
    for name, submodule in list(module.named_children()):
        # 如果是 LoRALinearLayer
        if "LoraLayer" in str(type(submodule)) or hasattr(submodule, "base_layer"):
            if hasattr(submodule, "base_layer") and isinstance(submodule.base_layer, nn.Linear):
                new_linear = nn.Linear(
                    in_features=submodule.base_layer.in_features,
                    out_features=submodule.base_layer.out_features,
                    bias=submodule.base_layer.bias is not None,
                    device=submodule.base_layer.weight.device,
                    dtype=submodule.base_layer.weight.dtype,
                )
                new_linear.weight.data = submodule.base_layer.weight.data.clone()
                if submodule.base_layer.bias is not None:
                    new_linear.bias.data = submodule.base_layer.bias.data.clone()

                setattr(module, name, new_linear)

        else:
            # 递归替换子模块
            replace_lora_linear_with_linear(submodule)

# ========== 设置路径和提示 ==========
name = 'input.png'
images_path = 'example/image'
masks_path = 'example/mask'
outputs_path = 'example/output'
os.makedirs(outputs_path, exist_ok=True)
prompt = 'There is nothing here.'

image_path = os.path.join(images_path, name)
mask_path = os.path.join(masks_path, name)
output_path = os.path.join(outputs_path, name)
size = (1024, 1024)

# ========== 加载 base transformer 并修改 x_embedder ==========
transformer = FluxTransformer2DModel.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
)

with torch.no_grad():
    ori_x_embedder = copy.deepcopy(transformer.x_embedder)
    initial_input_channels = transformer.config.in_channels
    new_linear = torch.nn.Linear(
        transformer.x_embedder.in_features * 4,
        transformer.x_embedder.out_features,
        bias=transformer.x_embedder.bias is not None,
        dtype=transformer.dtype,
        device=transformer.device,
    )
    new_linear.weight.zero_()
    new_linear.weight[:, :initial_input_channels].copy_(transformer.x_embedder.weight)
    if transformer.x_embedder.bias is not None:
        new_linear.bias.copy_(transformer.x_embedder.bias)
    transformer.x_embedder = new_linear
    transformer.register_to_config(in_channels=initial_input_channels * 4)

# ========== 加载 LoRA 并 merge ==========
# 注意：此处 transformer 必须是支持 peft 的
pipe = FluxControlRemovalPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    transformer=transformer,
    torch_dtype=torch.bfloat16
).to("cuda")

pipe.transformer.to(torch.bfloat16)

# 加载 LoRA 并融合
pipe.load_lora_weights("theSure/Omnieraser", weight_name="pytorch_lora_weights.safetensors")
manually_merge_lora(pipe.transformer)               # 融合权重
replace_lora_linear_with_linear(pipe.transformer)   # 替换模块类型

for name, module in pipe.transformer.named_modules():
    if "lora" in name.lower():
        print("⚠️ Still has lora:", name, type(module))

pipe.transformer.to(torch.bfloat16)
assert pipe.transformer.config.in_channels == initial_input_channels * 4

save_path = "./expanded_x_embedder.pth"
torch.save(transformer.x_embedder.state_dict(), save_path)
new_x_emb = copy.deepcopy(transformer.x_embedder) # 保存 new
transformer.x_embedder = ori_x_embedder # 恢复 original
transformer.register_to_config(in_channels=initial_input_channels) # 恢复 original channels
pipe.transformer.save_pretrained("./transformer")
transformer.x_embedder = new_x_emb
transformer.register_to_config(in_channels=initial_input_channels * 4)

print("✅ Successfully saved merged model.")

# ========== 图像加载 ==========
image = load_image(image_path)
original_size = image.size
image = image.convert("RGB").resize(size)
mask = load_image(mask_path).convert("RGB").resize(size)
generator = torch.Generator(device="cuda").manual_seed(24)

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