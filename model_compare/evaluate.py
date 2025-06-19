import os
import torch
import pandas as pd
from PIL import Image
from lpips import LPIPS
from torchvision import transforms
import matplotlib.pyplot as plt

# -------- 参数配置 --------
PROMPT_FILE = "model_compare/anime/prompts.txt"
OUTPUT_DIR = "model_compare/anime/picture"
RESULT_CSV = "model_compare/anime/lpips_anime.csv"

# -------- 加载 LPIPS 模型 --------
lpips_fn = LPIPS(net='vgg').to("cuda")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4815, 0.4578, 0.4082), (0.2686, 0.2613, 0.2758))
])

# -------- 加载 prompts --------
with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
    prompts = [line.strip() for line in f if line.strip()]

results = []

# -------- 遍历所有样本 --------
for idx, prompt in enumerate(prompts):
    key = f"{idx+1:02d}"
    base_path = os.path.join(OUTPUT_DIR, f"{key}_base.png")

    if not os.path.exists(base_path):
        print(f"缺失 base 图像：{base_path}，跳过")
        continue

    base_img = transform(Image.open(base_path).convert("RGB")).unsqueeze(0).to("cuda")

    for mode in ["lora", "lora+dreambooth", "textual_inversion"]:
        target_path = os.path.join(OUTPUT_DIR, f"{key}_{mode}.png")
        if not os.path.exists(target_path):
            print(f"缺失 {mode} 图像：{target_path}，跳过")
            continue

        target_img = transform(Image.open(target_path).convert("RGB")).unsqueeze(0).to("cuda")
        lpips_score = lpips_fn(base_img, target_img).item()

        results.append({
            "index": idx+1,
            "prompt": prompt,
            "mode": mode,
            "lpips": lpips_score
        })
        print(f"[{key} - {mode}] LPIPS = {lpips_score:.4f}")

# -------- 保存 CSV --------
df = pd.DataFrame(results)
df.to_csv(RESULT_CSV, index=False)
print(f"\n所有图像评估完成，结果已保存至 {RESULT_CSV}")

# -------- 可视化 --------
plt.figure(figsize=(10, 5))
for mode in ["lora", "lora+dreambooth", "textual_inversion"]:
    subset = df[df["mode"] == mode]
    plt.plot(subset["index"], subset["lpips"], marker='o', label=mode.capitalize())

plt.xticks(df["index"].unique(), df["prompt"].unique(), rotation=25, ha="right", fontsize=9)
plt.xlabel("Prompt")
plt.ylabel("LPIPS (vs. base image)")
plt.title("LoRA vs DreamBooth vs Textual Inversion - LPIPS Score")
plt.grid(True)
plt.legend(title="Model")
plt.tight_layout()
plt.savefig("model_compare/anime/lpips_comparison_anime.png", dpi=300)
plt.show()
