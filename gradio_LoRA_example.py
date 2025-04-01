import io
import os
import shutil
import uuid
import torch
import random
from diffusers.utils import load_image
import gradio as gr
import numpy as np

from PIL import Image, ImageCms
import torch
from controlnet_flux import FluxControlNetModel
from transformer_flux import FluxTransformer2DModel
from pipeline_flux_controlnet_inpaint import FluxControlNetInpaintingPipeline


torch.set_grad_enabled(False)
os.environ['GRADIO_TEMP_DIR'] = './tmp'

# torch.cuda.set_device(1)
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
image_path = mask_path =None
image_examples = [
    [
        "example/image/3c43156c-2b44-4ebf-9c47-7707ec60b166.png",
        "example/mask/3c43156c-2b44-4ebf-9c47-7707ec60b166.png"
    ],
    [
        "example/image/0e5124d8-fe43-4b5c-819f-7212f23a6d2a.png",
        "example/mask/0e5124d8-fe43-4b5c-819f-7212f23a6d2a.png"
    ],
    [
        "example/image/0f900fe8-6eab-4f85-8121-29cac9509b94.png",
        "example/mask/0f900fe8-6eab-4f85-8121-29cac9509b94.png"
    ],
    [
        "example/image/1a8e07fa-0138-4c80-90e5-7cd020f6fa2c.png",
        "example/mask/1a8e07fa-0138-4c80-90e5-7cd020f6fa2c.png"
    ],
    [
        "example/image/3ed1ee18-33b0-4964-b679-0e214a0d8848.png",
        "example/mask/3ed1ee18-33b0-4964-b679-0e214a0d8848.png"
    ],
    [
        "example/image/1ea0ad33-d348-467e-a44a-0d8cea188a7c.png",
        "example/mask/1ea0ad33-d348-467e-a44a-0d8cea188a7c.png"
    ]
]

def load_model(base_model_path, controlnet_path, lora_path):
    global pipe
    controlnet = FluxControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.bfloat16)
    gr.Info(str(f"Model loading: {int((20 / 100) * 100)}%"))
    transformer = FluxTransformer2DModel.from_pretrained(base_model_path, subfolder='transformer', torch_dtype=torch.bfloat16)
    gr.Info(str(f"Model loading: {int((40 / 100) * 100)}%"))
    pipe = FluxControlNetInpaintingPipeline.from_pretrained(
        pretrained_model_name_or_path=base_model_path,
        controlnet=controlnet,
        transformer=transformer,
        torch_dtype=torch.bfloat16
    ).to("cuda")
    pipe.transformer.to(torch.bfloat16)
    pipe.controlnet.to(torch.bfloat16)
    gr.Info(str(f"Model loading: {int((80 / 100) * 100)}%"))
    
    gr.Info(str(f"开启LoRA: {lora_path}"))
    pipe.load_lora_weights(lora_path, weight_name="controlnet_flux_rord_pytorch_lora_weights.safetensors")
    
    gr.Info(str(f"Model loading: {int((100 / 100) * 100)}%"))

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def save_images(gallery_data):
    if not gallery_data or len(gallery_data) < 4:
        raise gr.Error("Please generate the image first.")

    try:
        # 生成唯一ID
        unique_id = str(uuid.uuid4())
        
        # 创建保存目录
        os.makedirs('visualization', exist_ok=True)
        os.makedirs('visualization/images', exist_ok=True)
        os.makedirs('visualization/masks', exist_ok=True)
        os.makedirs('visualization/masked_image', exist_ok=True)
        os.makedirs('visualization/results_flux_foreground', exist_ok=True)

        # 直接复制原始图像文件（第一个元素）
        input_img_path = gallery_data[0]['name']
        shutil.copy(input_img_path, f'visualization/images/{unique_id}.png')

        # 复制遮罩文件（第二个元素）
        mask_img_path = gallery_data[1]['name']
        shutil.copy(mask_img_path, f'visualization/masks/{unique_id}.png')
        
        # 复制遮罩文件（第二个元素）
        mask_img_path = gallery_data[2]['name']
        shutil.copy(mask_img_path, f'visualization/masked_image/{unique_id}.png')

        # 复制最终结果文件（第四个元素）
        result_img_path = gallery_data[3]['name']
        shutil.copy(result_img_path, f'visualization/results_flux_foreground/{unique_id}.png')

        return gr.Info(f"Save successfully, ID: {unique_id}")

    except FileNotFoundError as e:
        raise gr.Error(f"The temporary file has expired. Please regenerate the result and save it: {str(e)}")
    except PermissionError as e:
        raise gr.Error(f"The file access permission is insufficient: {str(e)}")
    except Exception as e:
        raise gr.Error(f"Save failure: {str(e)}")
    
def predict(
    input_image,
    prompt,
    ddim_steps,
    seed,
    scale,
    image_paths,
    mask_paths

):
    global image_path, mask_path
    
    if image_paths is not None:
        input_image["image"] = load_image(image_paths).convert("RGB")
        input_image["mask"] = load_image(mask_paths).convert("RGB")
        
    size1, size2 = input_image["image"].convert("RGB").size

    icc_profile = input_image["image"].info.get('icc_profile')
    if icc_profile:
        gr.Info(str(f"Image detected to contain ICC profile, converting color space to sRGB..."))
        srgb_profile = ImageCms.createProfile("sRGB")
        io_handle = io.BytesIO(icc_profile)   
        src_profile = ImageCms.ImageCmsProfile(io_handle)  
        input_image["image"] = ImageCms.profileToProfile(input_image["image"], src_profile, srgb_profile)
        input_image["image"].info.pop('icc_profile', None)

    if size1 < size2:
        input_image["image"] = input_image["image"].convert("RGB").resize((1024, int(size2 / size1 * 1024)))
    else:
        input_image["image"] = input_image["image"].convert("RGB").resize((int(size1 / size2 * 1024), 1024))

    img = np.array(input_image["image"].convert("RGB"))

    W = int(np.shape(img)[0] - np.shape(img)[0] % 8)
    H = int(np.shape(img)[1] - np.shape(img)[1] % 8)

    input_image["image"] = input_image["image"].resize((H, W))
    input_image["mask"] = input_image["mask"].resize((H, W))

    if seed == -1:
        seed = random.randint(1, 2147483647)
        set_seed(random.randint(1, 2147483647))
    else:
        set_seed(seed)

    gr.Info(str(f"Set seed = {seed}"))

    result = pipe(
        prompt=prompt,
        control_image=input_image["image"].convert("RGB"),
        control_mask=input_image["mask"].convert("RGB"),
        width=H,
        height=W,
        num_inference_steps=ddim_steps,
        generator=torch.Generator("cuda").manual_seed(seed),
        controlnet_conditioning_scale=0.9,
        guidance_scale=3.5,
        negative_prompt="",
        true_guidance_scale=scale # default: 3.5 for alpha and 1.0 for beta
    ).images[0]

    mask_np = np.array(input_image["mask"].convert("RGB"))
    Info = ""
    # ===============红色Mask覆盖到原始图像================================
    red = np.array(input_image["image"]).astype("float") * 1
    red[:, :, 0] = 180.0
    red[:, :, 2] = 0
    red[:, :, 1] = 0
    result_m = np.array(input_image["image"])
    result_m = Image.fromarray(
        (
            result_m.astype("float") * (1 - mask_np.astype("float") / 512.0) + mask_np.astype("float") / 512.0 * red
        ).astype("uint8")
    )
    # ===============背景回贴==========================
    # image_np = np.array(result)
    # src_img_np = np.array(input_image["image"])

    # blur = ImageFilter.GaussianBlur(8)
    # mask_blur = input_image["mask"].convert("RGB").filter(blur)
    # mask_np1 = np.array(mask_blur) / 255.0
    # res2_np = image_np * mask_np1 + src_img_np * (1 - mask_np1)
    # res2 = Image.fromarray(res2_np.astype(np.uint8))
    # # =============================================
    # icc_profile = input_image["image"].info.get('icc_profile')
    # if icc_profile:
    #     result_m.info['icc_profile'] = icc_profile
    #     result.info['icc_profile'] = icc_profile
    #     res2.info['icc_profile'] = icc_profile

    dict_res = [input_image["image"], input_image["mask"], result_m, result]

    gr.Info(Info)
    dict_out = [result]
    image_path = None
    mask_path = None
    return dict_out, dict_res
   

def infer(
    input_image,
    ddim_steps,
    seed,
    scale,
    removal_prompt,

):
    img_path = image_path
    msk_path = mask_path
    return predict(input_image, 
                   removal_prompt, 
                   ddim_steps, 
                   seed,
                   scale,
                   img_path,
                   msk_path
    )

def process_example(image_paths, mask_paths):
    global image_path, mask_path
    image = Image.open(image_paths).convert("RGB")
    mask = Image.open(mask_paths).convert("L") 
    black_background = Image.new("RGB", image.size, (0, 0, 0))
    masked_image = Image.composite(black_background, image, mask)
    
    # 设置全局变量
    image_path = image_paths
    mask_path = mask_paths
    
    return masked_image


with gr.Blocks(css="style.css") as demo:
    base_model_path = '/mnt/sdb/yinzijin/checkpoints/FLUX.1-dev'
    controlnet_model_path = '/mnt/sdb/yinzijin/checkpoints/FLUX.1-dev-Controlnet-Inpainting-Beta'
    lora_path = '/home/yinzijin/weirunpu/flux-removal/'
    load_model(base_model_path=base_model_path, 
               controlnet_path=controlnet_model_path,
               lora_path=lora_path)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Input image and draw mask")
            # input_image = gr.Image(source="upload", tool="sketch", type="pil")
            input_image = gr.Image(
                source="upload", 
                tool="sketch", 
                type="pil",
                label="Upload Image & Draw Mask",
                interactive=True,
            )
            task = "object-removal"
            _ = gr.Text(label="Tips", value='This page is for testing purposes only.', interactive=False)

            run_button = gr.Button(label="Run object erase")
            with gr.Accordion("Advanced options", open=False):
                base_model = gr.Text(label="Base Model", value=base_model_path, interactive=True)
                controlnet_model = gr.Text(label="ControlNet Model", value=controlnet_model_path, interactive=True)

                lora_model = gr.Text(label="LoRA Model", value=lora_path, interactive=True)

                load_btn = gr.Button(value="Load the modal")
                load_btn.click(load_model, inputs=[base_model, controlnet_model, lora_model])

                ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=50, value=28, step=1, interactive=True)
                scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=45.0, value=1.0, step=0.1, interactive=True)
                seed = gr.Slider(
                    label="Seed",
                    value=1234,
                    minimum=-1,
                    maximum=2147483647,
                    step=1,
                    randomize=False,
                    interactive=True
                )

                removal_prompt = gr.Textbox(label="Prompt", 
                                            value="There is nothing here.", 
                                            interactive=False,)
            
            
            
            example = gr.Examples(
                label="Input Example",
                examples=image_examples,
                inputs=[
                    gr.Image(label="Image", type="filepath",visible=False),
                    gr.Image(label="Mask", type="filepath",visible=False)
                ],
                outputs=[input_image],
                fn=process_example,
                run_on_click=True,
                examples_per_page=10
            )


        with gr.Column():
            gr.Markdown("### Inpainting result")
            inpaint_result = gr.Gallery(label="Generated images", show_label=False, columns=2, height='auto')
            gr.Markdown("### Initial image, mask, masked_image, model_output")
            gallery = gr.Gallery(label="Generated masks", show_label=False, columns=2, height='auto')

            save_btn = gr.Button("Save the results")
    
        
    # 绑定保存按钮事件
    save_btn.click(
        fn=save_images,
        inputs=[gallery],
        outputs=None
    )
    
    
    run_button.click(
        fn=infer,
        inputs=[
            input_image,
            ddim_steps,
            seed,
            scale,
            removal_prompt,

        ],
        outputs=[inpaint_result, gallery],
    )


if __name__ == '__main__':
    demo.queue()
    demo.launch(share=False, 
                server_name="0.0.0.0", 
                server_port=7999, 
                # debug=True,
                )