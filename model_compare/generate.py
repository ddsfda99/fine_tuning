import os
import torch
from PIL import Image
from datetime import datetime
from diffusers import StableDiffusionPipeline
from safetensors.torch import load_file

PROMPT_FILE = "model_compare/anime/prompts.txt"
OUTPUT_BASE = "model_compare/anime/picture"
MODEL_DIR = "models/stable-diffusion-v1-5"
LORA_DIRS = {
    "lora_anime": "outputs/lora_anime/pytorch_lora_weights.safetensors",
    "lora_dreambooth_anime": "outputs/lora_dreambooth_anime/pytorch_lora_weights.safetensors"
}
TI_PATH = "outputs/textual_inversion_anime/learned_embeds.safetensors"

os.makedirs(OUTPUT_BASE, exist_ok=True)

# 加载基础模型
print("加载基础模型...")
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float16
).to("cuda")
pipe.enable_attention_slicing()

# 加载 prompts
with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
    prompts = [line.strip() for line in f if line.strip()]

# 加载 Textual Inversion embedding
ti_embeddings = load_file(TI_PATH)
ti_token = list(ti_embeddings.keys())[0]  # 自动提取 token，比如 "<anime-style>"
print(f"TI token: {ti_token}")

# 添加新 token 并注入 embedding
tokenizer_len = len(pipe.tokenizer)
pipe.tokenizer.add_tokens(ti_token)
pipe.text_encoder.resize_token_embeddings(tokenizer_len + 1)
pipe.text_encoder.get_input_embeddings().weight.data[-1] = ti_embeddings[ti_token]
print("注入 Textual Inversion embedding 完成")

# 开始生成图像
for idx, prompt in enumerate(prompts):
    print(f"\n正在生成第 {idx+1} 条 Prompt：{prompt}")
    out_dir = os.path.join(OUTPUT_BASE)
    os.makedirs(out_dir, exist_ok=True)

    # base
    pipe.unload_lora_weights()
    image = pipe(prompt, height=512, width=512, num_inference_steps=30).images[0]
    image.save(os.path.join(out_dir, f"{idx+1:02d}_base.png"))
    print("base 图生成完成")

    # LoRA Anime
    pipe.load_lora_weights(LORA_DIRS["lora_anime"])
    image = pipe(prompt, height=512, width=512, num_inference_steps=30).images[0]
    image.save(os.path.join(out_dir, f"{idx+1:02d}_lora.png"))
    print("LoRA Anime 图生成完成")

    # LoRA DreamBooth Anime
    pipe.load_lora_weights(LORA_DIRS["lora_dreambooth_anime"])
    image = pipe(prompt, height=512, width=512, num_inference_steps=30).images[0]
    image.save(os.path.join(out_dir, f"{idx+1:02d}_lora+dreambooth.png"))
    print("DreamBooth Anime 图生成完成")

    # Textual Inversion
    pipe.unload_lora_weights()
    ti_prompt = prompt + " " + ti_token
    image = pipe(ti_prompt, height=512, width=512, num_inference_steps=30).images[0]
    image.save(os.path.join(out_dir, f"{idx+1:02d}_textual_inversion.png"))
    print("Textual Inversion 图生成完成")

print("\n所有模型图像已生成，保存在 model_compare/anime/picture/")
