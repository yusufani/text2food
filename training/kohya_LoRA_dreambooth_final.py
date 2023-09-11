#!/usr/bin/env python
# coding: utf-8

# ![visitors](https://visitor-badge.glitch.me/badge?page_id=linaqruf.lora-dreambooth) [![](https://dcbadge.vercel.app/api/shield/850007095775723532?style=flat)](https://lookup.guru/850007095775723532) [![ko-fi](https://img.shields.io/badge/Support%20me%20on%20Ko--fi-F16061?logo=ko-fi&logoColor=white&style=flat)](https://ko-fi.com/linaqruf) <a href="https://saweria.co/linaqruf"><img alt="Saweria" src="https://img.shields.io/badge/Saweria-7B3F00?style=flat&logo=ko-fi&logoColor=white"/></a>
# 
# # **Kohya LoRA Dreambooth**
# A Colab Notebook For LoRA Training (Dreambooth Method)

# | Notebook Name | Description | Link | V14 |
# | --- | --- | --- | --- |
# | [Kohya LoRA Dreambooth](https://github.com/Linaqruf/kohya-trainer/blob/main/kohya-LoRA-dreambooth.ipynb) | LoRA Training (Dreambooth method) | [![](https://img.shields.io/static/v1?message=Open%20in%20Colab&logo=googlecolab&labelColor=5c5c5c&color=0f80c1&label=%20&style=flat)](https://colab.research.google.com/github/Linaqruf/kohya-trainer/blob/main/kohya-LoRA-dreambooth.ipynb) | [![](https://img.shields.io/static/v1?message=Older%20Version&logo=googlecolab&labelColor=5c5c5c&color=e74c3c&label=%20&style=flat)](https://colab.research.google.com/github/Linaqruf/kohya-trainer/blob/ff701379c65380c967cd956e4e9e8f6349563878/kohya-LoRA-dreambooth.ipynb) | 
# | [Kohya LoRA Fine-Tuning](https://github.com/Linaqruf/kohya-trainer/blob/main/kohya-LoRA-finetuner.ipynb) | LoRA Training (Fine-tune method) | [![](https://img.shields.io/static/v1?message=Open%20in%20Colab&logo=googlecolab&labelColor=5c5c5c&color=0f80c1&label=%20&style=flat)](https://colab.research.google.com/github/Linaqruf/kohya-trainer/blob/main/kohya-LoRA-finetuner.ipynb) | [![](https://img.shields.io/static/v1?message=Older%20Version&logo=googlecolab&labelColor=5c5c5c&color=e74c3c&label=%20&style=flat)](https://colab.research.google.com/github/Linaqruf/kohya-trainer/blob/ff701379c65380c967cd956e4e9e8f6349563878/kohya-LoRA-finetuner.ipynb) | 
# | [Kohya Trainer](https://github.com/Linaqruf/kohya-trainer/blob/main/kohya-trainer.ipynb) | Native Training | [![](https://img.shields.io/static/v1?message=Open%20in%20Colab&logo=googlecolab&labelColor=5c5c5c&color=0f80c1&label=%20&style=flat)](https://colab.research.google.com/github/Linaqruf/kohya-trainer/blob/main/kohya-trainer.ipynb) | [![](https://img.shields.io/static/v1?message=Older%20Version&logo=googlecolab&labelColor=5c5c5c&color=e74c3c&label=%20&style=flat)](https://colab.research.google.com/github/Linaqruf/kohya-trainer/blob/ff701379c65380c967cd956e4e9e8f6349563878/kohya-trainer.ipynb) | 
# | [Kohya Dreambooth](https://github.com/Linaqruf/kohya-trainer/blob/main/kohya-dreambooth.ipynb) | Dreambooth Training | [![](https://img.shields.io/static/v1?message=Open%20in%20Colab&logo=googlecolab&labelColor=5c5c5c&color=0f80c1&label=%20&style=flat)](https://colab.research.google.com/github/Linaqruf/kohya-trainer/blob/main/kohya-dreambooth.ipynb) | [![](https://img.shields.io/static/v1?message=Older%20Version&logo=googlecolab&labelColor=5c5c5c&color=e74c3c&label=%20&style=flat)](https://colab.research.google.com/github/Linaqruf/kohya-trainer/blob/ff701379c65380c967cd956e4e9e8f6349563878/kohya-dreambooth.ipynb) | 
# | [Cagliostro Colab UI](https://github.com/Linaqruf/sd-notebook-collection/blob/main/cagliostro-colab-ui.ipynb) `NEW`| A Customizable Stable Diffusion Web UI| [![](https://img.shields.io/static/v1?message=Open%20in%20Colab&logo=googlecolab&labelColor=5c5c5c&color=0f80c1&label=%20&style=flat)](https://colab.research.google.com/github/Linaqruf/sd-notebook-collection/blob/main/cagliostro-colab-ui.ipynb) | 

# # I. Install Kohya Trainer

# In[ ]:


# @title ## 1.1. Install Dependencies
# @markdown Clone Kohya Trainer from GitHub and check for updates. Use textbox below if you want to checkout other branch or old commit. Leave it empty to stay the HEAD on main.  This will also install the required libraries.
import os
import zipfile
import shutil
import time
import torch
from subprocess import getoutput
# from IPython.utils import capture
# from google.colab import drive

# get_ipython().run_line_magic('store', '-r')
os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# root_dir
root_dir = "/home/daryna-lab-course/text2food/data/kohya-trainer"
deps_dir = os.path.join(root_dir, "deps")
repo_dir = os.path.join(root_dir, "kohya-trainer")
training_dir = os.path.join(root_dir, "LoRA")
pretrained_model = os.path.join(root_dir, "pretrained_model")
vae_dir = os.path.join(root_dir, "vae")
config_dir = os.path.join(training_dir, "config")

# repo_dir
accelerate_config = os.path.join(repo_dir, "accelerate_config/config.yaml")
tools_dir = os.path.join(repo_dir, "tools")
finetune_dir = os.path.join(repo_dir, "finetune")

for store in [
    "root_dir",
    "deps_dir",
    "repo_dir",
    "training_dir",
    "pretrained_model",
    "vae_dir",
    "accelerate_config",
    "tools_dir",
    "finetune_dir",
    "config_dir",
]:
    # with capture.capture_output() as cap:
    #     get_ipython().run_line_magic('store', '{store}')
    #     del cap

    repo_url = "https://github.com/Linaqruf/kohya-trainer"
bitsandytes_main_py = "/home/daryna-lab-course/text2food/data/kohya-trainer/bitsandbytes/bitsandbytes/cuda_setup/main.py"
branch = ""  # @param {type: "string"}
install_xformers = True  # @param {'type':'boolean'}
mount_drive = False  # @param {type: "boolean"}
verbose = False # @param {type: "boolean"}

def read_file(filename):
    with open(filename, "r") as f:
        contents = f.read()
    return contents


def write_file(filename, contents):
    with open(filename, "w") as f:
        f.write(contents)


# def clone_repo(url):
#     if not os.path.exists(repo_dir):
#         os.chdir(root_dir)
#         !git clone {url} {repo_dir}
#     else:
#         os.chdir(repo_dir)
#         !git pull origin {branch} if branch else !git pull


# def ubuntu_deps(url, name, dst):
#     !wget {'-q' if not verbose else ''} --show-progress {url}
#     with zipfile.ZipFile(name, "r") as deps:
#         deps.extractall(dst)
#     !dpkg -i {dst}/*
#     os.remove(name)
#     shutil.rmtree(dst)


def install_dependencies():
    # s = getoutput('nvidia-smi')

    # if 'T4' in s:
    #     !sed -i "s@cpu@cuda@" library/model_util.py

    # !pip install {'-q' if not verbose else ''} --upgrade -r requirements.txt
    # !pip install {'-q' if not verbose else ''} torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1+cu118 torchtext==0.15.1 torchdata==0.6.0 --extra-index-url https://download.pytorch.org/whl/cu118 -U

    # if install_xformers:
        # !pip install {'-q' if not verbose else ''} xformers==0.0.19 triton==2.0.0 -U

    from accelerate.utils import write_basic_config

    if not os.path.exists(accelerate_config):
        write_basic_config(save_location=accelerate_config)


def remove_bitsandbytes_message(filename):
    welcome_message = """
def evaluate_cuda_setup():
    print('')
    print('='*35 + 'BUG REPORT' + '='*35)
    print('Welcome to bitsandbytes. For bug reports, please submit your error trace to: https://github.com/TimDettmers/bitsandbytes/issues')
    print('For effortless bug reporting copy-paste your error into this form: https://docs.google.com/forms/d/e/1FAIpQLScPB8emS3Thkp66nvqwmjTEgxp8Y9ufuWTzFyr9kJ5AoI47dQ/viewform?usp=sf_link')
    print('='*80)"""

    new_welcome_message = """
def evaluate_cuda_setup():
    import os
    if 'BITSANDBYTES_NOWELCOME' not in os.environ or str(os.environ['BITSANDBYTES_NOWELCOME']) == '0':
        print('')
        print('=' * 35 + 'BUG REPORT' + '=' * 35)
        print('Welcome to bitsandbytes. For bug reports, please submit your error trace to: https://github.com/TimDettmers/bitsandbytes/issues')
        print('For effortless bug reporting copy-paste your error into this form: https://docs.google.com/forms/d/e/1FAIpQLScPB8emS3Thkp66nvqwmjTEgxp8Y9ufuWTzFyr9kJ5AoI47dQ/viewform?usp=sf_link')
        print('To hide this message, set the BITSANDBYTES_NOWELCOME variable like so: export BITSANDBYTES_NOWELCOME=1')
        print('=' * 80)"""

    contents = read_file(filename)
    new_contents = contents.replace(welcome_message, new_welcome_message)
    write_file(filename, new_contents)


def main():
    os.chdir(root_dir)

    if mount_drive:
        if not os.path.exists("/content/drive"):
            drive.mount("/content/drive")

    for dir in [
        deps_dir,
        training_dir,
        config_dir,
        pretrained_model,
        vae_dir
    ]:
        os.makedirs(dir, exist_ok=True)

    # clone_repo(repo_url)

    # if branch:
    #     os.chdir(repo_dir)
    #     status = os.system(f"git checkout {branch}")
    #     if status != 0:
    #         raise Exception("Failed to checkout branch or commit")

    # os.chdir(repo_dir)
    
    # get_ipython().system("apt -y update {'-qq' if not verbose else ''}")
    # get_ipython().system("apt install libunwind8-dev {'-qq' if not verbose else ''}")

    # ubuntu_deps(
    #     "https://huggingface.co/Linaqruf/fast-repo/resolve/main/deb-libs.zip",
    #     "deb-libs.zip",
    #     deps_dir,
    # )

    install_dependencies()
    time.sleep(3)
    
    remove_bitsandbytes_message(bitsandytes_main_py)

    os.environ["LD_PRELOAD"] = "libtcmalloc.so"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["BITSANDBYTES_NOWELCOME"] = "1"  
    os.environ["SAFETENSORS_FAST_GPU"] = "1"

    # cuda_path = "/usr/local/cuda-11.8/targets/x86_64-linux/lib/"
    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    # os.environ["LD_LIBRARY_PATH"] = f"{ld_library_path}:{cuda_path}"

main()

## COMMENTED OUT 1.2 

# In[ ]:


# # @title ## 1.2. Start `File Explorer`
# # @markdown This will work in real-time even when you run other cells
# import threading
# # from google.colab import output
# from imjoy_elfinder.app import main

# open_in_new_tab = True  # @param {type:"boolean"}

# def start_file_explorer(root_dir=root_dir, port=8765):
#     try:
#         main(["--root-dir=" + root_dir, "--port=" + str(port)])
#     except Exception as e:
#         print("Error starting file explorer:", str(e))


# def open_file_explorer(open_in_new_tab=False, root_dir=root_dir, port=8765):
#     thread = threading.Thread(target=start_file_explorer, args=[root_dir, port])
#     thread.start()

#     if open_in_new_tab:
#         output.serve_kernel_port_as_window(port)
#     else:
#         output.serve_kernel_port_as_iframe(port, height="500")

# open_file_explorer(open_in_new_tab=open_in_new_tab, root_dir=root_dir, port=8765)


# # II. Pretrained Model Selection

# In[ ]:


# @title ## 2.1. Download Available Model
import os

# get_ipython().run_line_magic('store', '-r')

os.chdir(root_dir)

models = {
    "Animefull-final-pruned": "https://huggingface.co/Linaqruf/personal-backup/resolve/main/models/animefull-final-pruned.ckpt",
    "Anything-v3-1": "https://huggingface.co/cag/anything-v3-1/resolve/main/anything-v3-1.safetensors",
    "AnyLoRA": "https://huggingface.co/Linaqruf/stolen/resolve/main/pruned-models/AnyLoRA_noVae_fp16-pruned.safetensors",
    "AnimePastelDream": "https://huggingface.co/Lykon/AnimePastelDream/resolve/main/AnimePastelDream_Soft_noVae_fp16.safetensors",
    "Chillout-mix": "https://huggingface.co/Linaqruf/stolen/resolve/main/pruned-models/chillout_mix-pruned.safetensors",
    "OpenJourney-v4": "https://huggingface.co/prompthero/openjourney-v4/resolve/main/openjourney-v4.ckpt",
    "Stable-Diffusion-v1-5": "https://huggingface.co/Linaqruf/stolen/resolve/main/pruned-models/stable_diffusion_1_5-pruned.safetensors",
}

v2_models = {
    "stable-diffusion-2-1-base": "https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.safetensors",
    "stable-diffusion-2-1-768v": "https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/v2-1_768-ema-pruned.safetensors",
    "plat-diffusion-v1-3-1": "https://huggingface.co/p1atdev/pd-archive/resolve/main/plat-v1-3-1.safetensors",
    "replicant-v1": "https://huggingface.co/gsdf/Replicant-V1.0/resolve/main/Replicant-V1.0.safetensors",
    "illuminati-diffusion-v1-0": "https://huggingface.co/IlluminatiAI/Illuminati_Diffusion_v1.0/resolve/main/illuminati_diffusion_v1.0.safetensors",
    "illuminati-diffusion-v1-1": "https://huggingface.co/4eJIoBek/Illuminati-Diffusion-v1-1/resolve/main/illuminatiDiffusionV1_v11.safetensors",
    "waifu-diffusion-1-4-anime-e2": "https://huggingface.co/hakurei/waifu-diffusion-v1-4/resolve/main/wd-1-4-anime_e2.ckpt",
    "waifu-diffusion-1-5-e2": "https://huggingface.co/waifu-diffusion/wd-1-5-beta2/resolve/main/checkpoints/wd-1-5-beta2-fp32.safetensors",
    "waifu-diffusion-1-5-e2-aesthetic": "https://huggingface.co/waifu-diffusion/wd-1-5-beta2/resolve/main/checkpoints/wd-1-5-beta2-aesthetic-fp32.safetensors",
}

installModels = []
installv2Models = []

# @markdown ### SD1.x model
model_name = "Stable-Diffusion-v1-5"  # @param ["", "Animefull-final-pruned", "Anything-v3-1", "AnyLoRA", "AnimePastelDream", "Chillout-mix", "OpenJourney-v4", "Stable-Diffusion-v1-5"]
# @markdown ### SD2.x model
v2_model_name = "stable-diffusion-2-1-base"  # @param ["", "stable-diffusion-2-1-base", "stable-diffusion-2-1-768v", "plat-diffusion-v1-3-1", "replicant-v1", "illuminati-diffusion-v1-0", "illuminati-diffusion-v1-1", "waifu-diffusion-1-4-anime-e2", "waifu-diffusion-1-5-e2", "waifu-diffusion-1-5-e2-aesthetic"]

if model_name:
    model_url = models.get(model_name)
    if model_url:
        installModels.append((model_name, model_url))

if v2_model_name:
    v2_model_url = v2_models.get(v2_model_name)
    if v2_model_url:
        installv2Models.append((v2_model_name, v2_model_url))


# def install(checkpoint_name, url):
#     ext = "ckpt" if url.endswith(".ckpt") else "safetensors"

#     hf_token = "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"
#     user_header = f'"Authorization: Bearer {hf_token}"'
    # !aria2c --console-log-level=error --summary-interval=10 --header={user_header} -c -x 16 -k 1M -s 16 -d {pretrained_model} -o {checkpoint_name}.{ext} "{url}"

# def install_checkpoint():
#     for model in installModels:
#         install(model[0], model[1])
#     for v2model in installv2Models:
#         install(v2model[0], v2model[1])


# install_checkpoint()

## DON'T RUN THE CELL BELOW -- COMMENTING IT OUT.

# In[ ]:


# # @title ## 2.2. Download Custom Model
# import os

# get_ipython().run_line_magic('store', '-r')

# os.chdir(root_dir)

# # @markdown ### Custom model
# modelUrls = ""  # @param {'type': 'string'}

# def install(url):
#     base_name = os.path.basename(url)

#     if "drive.google.com" in url:
#         os.chdir(pretrained_model)
#         get_ipython().system('gdown --fuzzy {url}')
#     elif "huggingface.co" in url:
#         if "/blob/" in url:
#             url = url.replace("/blob/", "/resolve/")
#         # @markdown Change this part with your own huggingface token if you need to download your private model
#         hf_token = "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"  # @param {type:"string"}
#         user_header = f'"Authorization: Bearer {hf_token}"'
#         get_ipython().system('aria2c --console-log-level=error --summary-interval=10 --header={user_header} -c -x 16 -k 1M -s 16 -d {pretrained_model} -o {base_name} {url}')
#     else:
#         get_ipython().system('aria2c --console-log-level=error --summary-interval=10 -c -x 16 -k 1M -s 16 -d {pretrained_model} {url}')

# if modelUrls:
#     urls = modelUrls.split(",")
#     for url in urls:
#         install(url.strip())

## DON'T CELL THIS BELOW -- COMMENTING IT OUT

# In[ ]:


# # @title ## 2.3. Download Available VAE (Optional)
# import os

# get_ipython().run_line_magic('store', '-r')

# os.chdir(root_dir)

# vaes = {
#     "none": "",
#     "anime.vae.pt": "https://huggingface.co/Linaqruf/personal-backup/resolve/main/vae/animevae.pt",
#     "waifudiffusion.vae.pt": "https://huggingface.co/hakurei/waifu-diffusion-v1-4/resolve/main/vae/kl-f8-anime.ckpt",
#     "stablediffusion.vae.pt": "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.ckpt",
# }
# install_vaes = []

# # @markdown Select one of the VAEs to download, select `none` for not download VAE:
# vae_name = "anime.vae.pt"  # @param ["none", "anime.vae.pt", "waifudiffusion.vae.pt", "stablediffusion.vae.pt"]

# if vae_name in vaes:
#     vae_url = vaes[vae_name]
#     if vae_url:
#         install_vaes.append((vae_name, vae_url))


# def install(vae_name, url):
#     hf_token = "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"
#     user_header = f'"Authorization: Bearer {hf_token}"'
#     get_ipython().system('aria2c --console-log-level=error --summary-interval=10 --header={user_header} -c -x 16 -k 1M -s 16 -d {vae_dir} -o {vae_name} "{url}"')


# def install_vae():
#     for vae in install_vaes:
#         install(vae[0], vae[1])


# install_vae()


# # III. Data Acquisition
# 
# You have three options for acquiring your dataset:
# 
# 1. Uploading it to Colab's local files.
# 2. Bulk downloading images from Danbooru using the `Simple Booru Scraper`.
# 3. Locating your dataset from Google Drive.
# 

# In[ ]:


# @title ## 3.1. Locating Train Data Directory
# @markdown Define the location of your training data. This cell will also create a folder based on your input. Regularization Images is optional and can be skipped.
import os
# from IPython.utils import capture

# get_ipython().run_line_magic('store', '-r')

train_data_dir = "/home/daryna-lab-course/text2food/data/final_data"  # @param {type:'string'}
reg_data_dir = "/home/daryna-lab-course/text2food/data/reg_data"  # @param {type:'string'} // Make dummy

# for dir in [train_data_dir, reg_data_dir]:
#     if dir:
#         with capture.capture_output() as cap:
#             os.makedirs(dir, exist_ok=True)
#             # get_ipython().run_line_magic('store', 'dir')
#             del cap

print(f"Your train data directory : {train_data_dir}")
if reg_data_dir:
    print(f"Your reg data directory : {reg_data_dir}")

## BELOW UNNECCESARY COMMENTED OUT

# In[ ]:


# # @title ## 3.2. Unzip Dataset

# import os
# import shutil
# from pathlib import Path

# #@title ## Unzip Dataset
# # @markdown Use this section if your dataset is in a `zip` file and has been uploaded somewhere. This code cell will download your dataset and automatically extract it to the `train_data_dir` if the `unzip_to` variable is empty.
# zipfile_url = "" #@param {type:"string"}
# zipfile_name = "zipfile.zip"
# unzip_to = "" #@param {type:"string"}

# hf_token = "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"
# user_header = f'"Authorization: Bearer {hf_token}"'

# if unzip_to:
#     os.makedirs(unzip_to, exist_ok=True)
# else:
#     unzip_to = train_data_dir


# def download_dataset(url):
#     if url.startswith("/content"):
#         return url
#     elif "drive.google.com" in url:
#         os.chdir(root_dir)
#         get_ipython().system('gdown --fuzzy {url}')
#         return f"{root_dir}/{zipfile_name}"
#     elif "huggingface.co" in url:
#         if "/blob/" in url:
#             url = url.replace("/blob/", "/resolve/")
#         get_ipython().system('aria2c --console-log-level=error --summary-interval=10 --header={user_header} -c -x 16 -k 1M -s 16 -d {root_dir} -o {zipfile_name} {url}')
#         return f"{root_dir}/{zipfile_name}"
#     else:
#         get_ipython().system('aria2c --console-log-level=error --summary-interval=10 -c -x 16 -k 1M -s 16 -d {root_dir} -o {zipfile_name} {url}')
#         return f"{root_dir}/{zipfile_name}"


# def extract_dataset(zip_file, output_path):
#     if zip_file.startswith("/content"):
#         get_ipython().system('unzip -j -o {zip_file} -d "{output_path}"')
#     else:
#         get_ipython().system('unzip -j -o "{zip_file}" -d "{output_path}"')


# def remove_files(train_dir, files_to_move):
#     for filename in os.listdir(train_dir):
#         file_path = os.path.join(train_dir, filename)
#         if filename in files_to_move:
#             if not os.path.exists(file_path):
#                 shutil.move(file_path, training_dir)
#             else:
#                 os.remove(file_path)


# zip_file = download_dataset(zipfile_url)
# extract_dataset(zip_file, unzip_to)
# os.remove(zip_file)

# files_to_move = (
#     "meta_cap.json",
#     "meta_cap_dd.json",
#     "meta_lat.json",
#     "meta_clean.json",
# )

# remove_files(train_data_dir, files_to_move)


# # IV. Data Preprocessing

# In[ ]:


# @title ## 4.1. Data Cleaning
import os
import random
import concurrent.futures
from tqdm import tqdm
from PIL import Image

# get_ipython().run_line_magic('store', '-r')

os.chdir(root_dir)

test = os.listdir(train_data_dir)
# @markdown This section will delete unnecessary files and unsupported media such as `.mp4`, `.webm`, and `.gif`. 
# @markdown Set the `convert` parameter to convert your transparent dataset with an alpha channel (RGBA) to RGB and give it a white background. 
convert = True  # @param {type:"boolean"}
# @markdown You can choose to give it a `random_color` background instead of white by checking the corresponding option.
random_color = False  # @param {type:"boolean"}
# @markdown Use the `recursive` option to preprocess subfolders as well.
recursive = True #  @param {type:"boolean"}
 

batch_size = 32
supported_types = [
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".caption",
    ".npz",
    ".txt",
    ".json",
]

background_colors = [
    (255, 255, 255),
    (0, 0, 0),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
]

problematic_images= []
def clean_directory(directory):
    for item in tqdm(os.listdir(directory) , desc='Controlling unsupported files'):
        file_path = os.path.join(directory, item)
        if os.path.isfile(file_path):
            file_ext = os.path.splitext(item)[1]
            if file_ext not in supported_types:
                print(f"Deleting file {item} from {directory}")
                os.remove(file_path)
        elif os.path.isdir(file_path) and recursive:
            clean_directory(file_path)

def process_image(image_path):
    try:
      img = Image.open(image_path)
      img.load()
      img_dir, image_name = os.path.split(image_path)
      if img.mode in ("RGBA", "LA"):
          if random_color:
              background_color = random.choice(background_colors)
          else:
              background_color = (255, 255, 255)
          bg = Image.new("RGB", img.size, background_color)
          bg.paste(img, mask=img.split()[-1])

          if image_name.endswith(".webp"):
              bg = bg.convert("RGB")
              new_image_path = os.path.join(img_dir, image_name.replace(".webp", ".jpg"))
              bg.save(new_image_path, "JPEG")
              os.remove(image_path)
              print(f" Converted image: {image_name} to {os.path.basename(new_image_path)}")
          else:
              bg.save(image_path, "PNG")
              print(f" Converted image: {image_name}")
      else:
          if image_name.endswith(".webp"):
              new_image_path = os.path.join(img_dir, image_name.replace(".webp", ".jpg"))
              img.save(new_image_path, "JPEG")
              os.remove(image_path)
              print(f" Converted image: {image_name} to {os.path.basename(new_image_path)}")
          else:
              img.save(image_path, "PNG")
    except Exception as e :
      print(e)
      problematic_images.append(image_path)



def find_images(directory):
    images = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".png") or file.endswith(".webp"):
                images.append(os.path.join(root, file))
    return images

clean_directory(train_data_dir)
#print('cleaned')
images = find_images(train_data_dir)
#print('found')
num_batches = len(images) // batch_size + 1
#print('Num batches' , num_batches)

if convert:
    with concurrent.futures.ThreadPoolExecutor() as executor:
        #print('start')
        for i in tqdm(range(num_batches)):
            #print(i)
            start = i * batch_size
            end = start + batch_size
            #print(start)
            batch = images[start:end]
            executor.map(process_image, batch)

print("All images have been converted")

if len(problematic_images) > 0 :
  print(f'We found {len(problematic_images)} images has a problem in proccessing')
  print(problematic_images)
  a = input('Do you want to delete them').lower()
  if a == 'y':
    for i in problematic_images:
      os.remove(i)
else:
  print('All images are fine :D ')

## UNNCESSARY -- COMMENTED OUT 4.2

# ## 4.2. Data Annotation
# You can choose to train a model using captions. We're using [BLIP](https://huggingface.co/spaces/Salesforce/BLIP) for image captioning and [Waifu Diffusion 1.4 Tagger](https://huggingface.co/spaces/SmilingWolf/wd-v1-4-tags) for image tagging similar to Danbooru.
# - Use BLIP Captioning for: `General Images`
# - Use Waifu Diffusion 1.4 Tagger V2 for: `Anime and Manga-style Images`

# In[ ]:


#@title ### 4.2.1. BLIP Captioning
#@markdown BLIP is a pre-training framework for unified vision-language understanding and generation, which achieves state-of-the-art results on a wide range of vision-language tasks. It can be used as a tool for image captioning, for example, `astronaut riding a horse in space`. 
# import os

# os.chdir(finetune_dir)

# batch_size = 8 #@param {type:'number'}
# max_data_loader_n_workers = 2 #@param {type:'number'}
# beam_search = True #@param {type:'boolean'}
# min_length = 5 #@param {type:"slider", min:0, max:100, step:5.0}
# max_length = 75 #@param {type:"slider", min:0, max:100, step:5.0}
# #@markdown Use the `recursive` option to process subfolders as well, useful for multi-concept training.
# recursive = False #@param {type:"boolean"} 
# #@markdown Debug while captioning, it will print your image file with generated captions.
# verbose_logging = True #@param {type:"boolean"}

# config = {
#     "_train_data_dir" : train_data_dir,
#     "batch_size" : batch_size,
#     "beam_search" : beam_search,
#     "min_length" : min_length,
#     "max_length" : max_length,
#     "debug" : verbose_logging,
#     "caption_extension" : ".caption",
#     "max_data_loader_n_workers" : max_data_loader_n_workers,
#     "recursive" : recursive
# }

# args = ""
# for k, v in config.items():
#     if k.startswith("_"):
#         args += f'"{v}" '
#     elif isinstance(v, str):
#         args += f'--{k}="{v}" '
#     elif isinstance(v, bool) and v:
#         args += f"--{k} "
#     elif isinstance(v, float) and not isinstance(v, bool):
#         args += f"--{k}={v} "
#     elif isinstance(v, int) and not isinstance(v, bool):
#         args += f"--{k}={v} "

# final_args = f"python make_captions.py {args}"

# os.chdir(finetune_dir)
# get_ipython().system('{final_args}')

## ACtually 4.2.3 necessary but we will do it ourselves.

# In[ ]:


# @title ### 4.2.3. Custom Caption/Tag
# import os

# # get_ipython().run_line_magic('store', '-r')

# os.chdir(root_dir)

# # @markdown Add or remove custom tags here. You can refer to this [cheatsheet](https://rentry.org/kohyaminiguide#c-custom-tagscaption) for more information.
# extension = ".txt"  # @param [".txt", ".caption"]
# custom_tag = ""  # @param {type:"string"}
# # @markdown Use `sub_folder` option to specify a subfolder for multi-concept training. 
# # @markdown > Specify `--all` to process all subfolders/`recursive`
# sub_folder = "" #@param {type: "string"}
# # @markdown Enable this to append custom tags at the end of lines.
# append = False  # @param {type:"boolean"}
# # @markdown Enable this if you want to remove captions/tags instead.
# remove_tag = False  # @param {type:"boolean"}
# recursive = False

# if sub_folder == "":
#     image_dir = train_data_dir
# elif sub_folder == "--all":
#     image_dir = train_data_dir
#     recursive = True
# elif sub_folder.startswith("/content"):
#     image_dir = sub_folder
# else:
#     image_dir = os.path.join(train_data_dir, sub_folder)
#     os.makedirs(image_dir, exist_ok=True)

# def read_file(filename):
#     with open(filename, "r") as f:
#         contents = f.read()
#     return contents

# def write_file(filename, contents):
#     with open(filename, "w") as f:
#         f.write(contents)

# def process_tags(filename, custom_tag, append, remove_tag):
#     contents = read_file(filename)
#     tags = [tag.strip() for tag in contents.split(',')]
#     custom_tags = [tag.strip() for tag in custom_tag.split(',')]

#     for custom_tag in custom_tags:
#         custom_tag = custom_tag.replace("_", " ")
#         if remove_tag:
#             while custom_tag in tags:
#                 tags.remove(custom_tag)
#         else:
#             if custom_tag not in tags:
#                 if append:
#                     tags.append(custom_tag)
#                 else:
#                     tags.insert(0, custom_tag)

#     contents = ', '.join(tags)
#     write_file(filename, contents)

# def process_directory(image_dir, tag, append, remove_tag, recursive):
#     for filename in os.listdir(image_dir):
#         file_path = os.path.join(image_dir, filename)
        
#         if os.path.isdir(file_path) and recursive:
#             process_directory(file_path, tag, append, remove_tag, recursive)
#         elif filename.endswith(extension):
#             process_tags(file_path, tag, append, remove_tag)

# tag = custom_tag

# if not any(
#     [filename.endswith(extension) for filename in os.listdir(image_dir)]
# ):
#     for filename in os.listdir(image_dir):
#         if filename.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
#             open(
#                 os.path.join(image_dir, filename.split(".")[0] + extension),
#                 "w",
#             ).close()

# if custom_tag:
#     process_directory(image_dir, tag, append, remove_tag, recursive)


# # V. Training Model
# 
# 

# In[ ]:


# @title ## 5.1. Model Config
# from google.colab import drive

v2 = True  # @param {type:"boolean"}
v_parameterization = False  # @param {type:"boolean"}
project_name = "mmc4"  # @param {type:"string"}
if not project_name:
    project_name = "last"
# %store project_name
pretrained_model_name_or_path = "/home/daryna-lab-course/text2food/data/kohya-trainer/stable-diffusion-2-1-base.safetensors"  # @param {type:"string"}
# pretrained_model_name_or_path = os.path.join(root_dir,pretrained_model_name_or_path)
vae = ""  # @param {type:"string"}
output_dir = "output"  # @param {'type':'string'}
output_dir = os.path.join(root_dir,output_dir)
# @markdown `output_dir` is relative path from root dir
output_to_drive = False

if output_to_drive:
    if not os.path.exists("/content/drive"):
        drive.mount("/content/drive")

sample_dir = os.path.join(output_dir, "sample")
for dir in [output_dir, sample_dir]:
    os.makedirs(dir, exist_ok=True)

print("Project Name: ", project_name)
print("Model Version: Stable Diffusion V1.x") if not v2 else ""
print("Model Version: Stable Diffusion V2.x") if v2 and not v_parameterization else ""
print("Model Version: Stable Diffusion V2.x 768v") if v2 and v_parameterization else ""
print(
    "Pretrained Model Path: ", pretrained_model_name_or_path
) if pretrained_model_name_or_path else print("No Pretrained Model path specified.")
print("VAE Path: ", vae) if vae else print("No VAE path specified.")
print("Output Path: ", output_dir)


# In[ ]:


# @title ## 5.2. Dataset Config
import os
import toml
import glob

dataset_repeats = 1  # @param {type:"number"}
# @markdown `activation_word` is not used in training if you train with captions/tags, but it is still printed to metadata.
activation_word = "foods"  # @param {type:"string"}
caption_extension = ".txt"  # @param ["none", ".txt", ".caption"]
# @markdown Please refer to `4.2.3. Custom Caption/Tag (Optional)` if you want to append `activation_word` to captions/tags
resolution = 512  # @param {type:"slider", min:512, max:1024, step:128}
flip_aug = False  # @param {type:"boolean"}
keep_tokens = 0  # @param {type:"number"}

def parse_folder_name(folder_name, default_num_repeats, default_class_token):
    folder_name_parts = folder_name.split("_")

    if len(folder_name_parts) == 2:
        if folder_name_parts[0].isdigit():
            num_repeats = int(folder_name_parts[0])
            class_token = folder_name_parts[1].replace("_", " ")
        else:
            num_repeats = default_num_repeats
            class_token = default_class_token
    else:
        num_repeats = default_num_repeats
        class_token = default_class_token

    return num_repeats, class_token

def find_image_files(path):
    supported_extensions = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    return [file for file in glob.glob(path + '/**/*', recursive=True) if file.lower().endswith(supported_extensions)]

def process_data_dir(data_dir, default_num_repeats, default_class_token, is_reg=False):
    subsets = []

    images = find_image_files(data_dir)
    if images:
        subsets.append({
            "image_dir": data_dir,
            "class_tokens": default_class_token,
            "num_repeats": default_num_repeats,
            **({"is_reg": is_reg} if is_reg else {}),
        })

    for root, dirs, files in os.walk(data_dir):
        for folder in dirs:
            folder_path = os.path.join(root, folder)
            images = find_image_files(folder_path)

            if images:
                num_repeats, class_token = parse_folder_name(folder, default_num_repeats, default_class_token)

                subset = {
                    "image_dir": folder_path,
                    "class_tokens": class_token,
                    "num_repeats": num_repeats,
                }

                if is_reg:
                    subset["is_reg"] = True

                subsets.append(subset)

    return subsets


train_subsets = process_data_dir(train_data_dir, dataset_repeats, activation_word)
reg_subsets = process_data_dir(reg_data_dir, dataset_repeats, activation_word, is_reg=True)

subsets = train_subsets + reg_subsets

config = {
    "general": {
        "enable_bucket": True,
        "caption_extension": caption_extension,
        "shuffle_caption": True,
        "keep_tokens": keep_tokens,
        "bucket_reso_steps": 64,
        "bucket_no_upscale": False,
    },
    "datasets": [
        {
            "resolution": resolution,
            "min_bucket_reso": 320 if resolution > 640 else 256,
            "max_bucket_reso": 1280 if resolution > 640 else 1024,
            "caption_dropout_rate": 0,
            "caption_tag_dropout_rate": 0,
            "caption_dropout_every_n_epochs": 0,
            "flip_aug": flip_aug,
            "color_aug": False,
            "face_crop_aug_range": None,
            "subsets": subsets,
        }
    ],
}

dataset_config = os.path.join(config_dir, "dataset_config.toml")

for key in config:
    if isinstance(config[key], dict):
        for sub_key in config[key]:
            if config[key][sub_key] == "":
                config[key][sub_key] = None
    elif config[key] == "":
        config[key] = None

config_str = toml.dumps(config)

with open(dataset_config, "w") as f:
    f.write(config_str)

print(config_str)


# In[ ]:


# @title ## 5.3. LoRA and Optimizer Config

# @markdown ### LoRA Config:
network_category = "LoRA"  # @param ["LoRA", "LoCon", "LoCon_Lycoris", "LoHa"]

# @markdown Recommended values:

# @markdown | network_category | network_dim | network_alpha | conv_dim | conv_alpha |
# @markdown | :---: | :---: | :---: | :---: | :---: |
# @markdown | LoRA | 32 | 1 | - | - |
# @markdown | LoCon | 16 | 8 | 8 | 1 |
# @markdown | LoHa | 8 | 4 | 4 | 1 |

# @markdown - Note that `dropout` and `cp_decomposition` are not available in this notebook.

# @markdown `conv_dim` and `conv_alpha` are needed to train `LoCon` and `LoHa`; skip them if you are training normal `LoRA`. However, when in doubt, set `dim = alpha`.
conv_dim = 32  # @param {'type':'number'}
conv_alpha = 16  # @param {'type':'number'}
# @markdown It's recommended not to set `network_dim` and `network_alpha` higher than 64, especially for `LoHa`.
# @markdown If you want to use a higher value for `dim` or `alpha`, consider using a higher learning rate, as models with higher dimensions tend to learn faster.
network_dim = 32  # @param {'type':'number'}
network_alpha = 16  # @param {'type':'number'}
# @markdown You can specify this field for resume training.
network_weight = ""  # @param {'type':'string'}
network_module = "lycoris.kohya" if network_category in ["LoHa", "LoCon_Lycoris"] else "networks.lora"
network_args = "" if network_category == "LoRA" else [
    f"conv_dim={conv_dim}", f"conv_alpha={conv_alpha}",
    ]
# @markdown ### <br>Optimizer Config:
# @markdown `NEW` Gamma for reducing the weight of high-loss timesteps. Lower numbers have a stronger effect. The paper recommends 5. Read the paper [here](https://arxiv.org/abs/2303.09556).
min_snr_gamma = -1 #@param {type:"number"}
# @markdown `AdamW8bit` was the old `--use_8bit_adam`.
optimizer_type = "AdamW8bit"  # @param ["AdamW", "AdamW8bit", "Lion", "SGDNesterov", "SGDNesterov8bit", "DAdaptation", "AdaFactor"]
# @markdown Additional arguments for optimizer, e.g: `["decouple=True","weight_decay=0.6"]`
optimizer_args = ""  # @param {'type':'string'}
# @markdown Set `unet_lr` to `1.0` if you use `DAdaptation` optimizer, because it's a [free learning rate](https://github.com/facebookresearch/dadaptation) algorithm.
# @markdown However, it is recommended to set `text_encoder_lr = 0.5 * unet_lr`.
# @markdown Also, you don't need to specify `learning_rate` value if both `unet_lr` and `text_encoder_lr` are defined.
train_unet = True  # @param {'type':'boolean'}
unet_lr = 1e-4  # @param {'type':'number'}
train_text_encoder = True  # @param {'type':'boolean'}
text_encoder_lr = 5e-5  # @param {'type':'number'}
lr_scheduler = "constant"  # @param ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup", "adafactor"] {allow-input: false}
lr_warmup_steps = 0  # @param {'type':'number'}
# @markdown You can define `num_cycles` value for `cosine_with_restarts` or `power` value for `polynomial` in the field below.
lr_scheduler_num_cycles = 0  # @param {'type':'number'}
lr_scheduler_power = 0  # @param {'type':'number'}

if network_category == "LoHa":
  network_args.append("algo=loha")
elif network_category == "LoCon_Lycoris":
  network_args.append("algo=lora")

print("- LoRA Config:")
print(f"  - Min-SNR Weighting: {min_snr_gamma}") if not min_snr_gamma == -1 else ""
print(f"  - Loading network module: {network_module}")
if not network_category == "LoRA":
  print(f"  - network args: {network_args}")
print(f"  - {network_module} linear_dim set to: {network_dim}")
print(f"  - {network_module} linear_alpha set to: {network_alpha}")
if not network_category == "LoRA":
  print(f"  - {network_module} conv_dim set to: {conv_dim}")
  print(f"  - {network_module} conv_alpha set to: {conv_alpha}")

if not network_weight:
    print("  - No LoRA weight loaded.")
else:
    if os.path.exists(network_weight):
        print(f"  - Loading LoRA weight: {network_weight}")
    else:
        print(f"  - {network_weight} does not exist.")
        network_weight = ""

print("- Optimizer Config:")
print(f"  - Additional network category: {network_category}")
print(f"  - Using {optimizer_type} as Optimizer")
if optimizer_args:
    print(f"  - Optimizer Args: {optimizer_args}")
if train_unet and train_text_encoder:
    print("  - Train UNet and Text Encoder")
    print(f"    - UNet learning rate: {unet_lr}")
    print(f"    - Text encoder learning rate: {text_encoder_lr}")
if train_unet and not train_text_encoder:
    print("  - Train UNet only")
    print(f"    - UNet learning rate: {unet_lr}")
if train_text_encoder and not train_unet:
    print("  - Train Text Encoder only")
    print(f"    - Text encoder learning rate: {text_encoder_lr}")
print(f"  - Learning rate warmup steps: {lr_warmup_steps}")
print(f"  - Learning rate Scheduler: {lr_scheduler}")
if lr_scheduler == "cosine_with_restarts":
    print(f"  - lr_scheduler_num_cycles: {lr_scheduler_num_cycles}")
elif lr_scheduler == "polynomial":
    print(f"  - lr_scheduler_power: {lr_scheduler_power}")


# In[ ]:


# @title ## 5.4. Training Config

import toml
import os

# %store -r
lowram = False  # @param {type:"boolean"}
enable_sample_prompt = True  # @param {type:"boolean"}
sampler = "ddim"  # @param ["ddim", "pndm", "lms", "euler", "euler_a", "heun", "dpm_2", "dpm_2_a", "dpmsolver","dpmsolver++", "dpmsingle", "k_lms", "k_euler", "k_euler_a", "k_dpm_2", "k_dpm_2_a"]
noise_offset = 0.0  # @param {type:"number"}
num_epochs = 1  # @param {type:"number"}
vae_batch_size = 4  # @param {type:"number"}
train_batch_size = 20  # @param {type:"number"}
mixed_precision = "fp16"  # @param ["no","fp16","bf16"] {allow-input: false}
save_precision = "fp16"  # @param ["float", "fp16", "bf16"] {allow-input: false}
save_n_epochs_type = "save_every_n_epochs"  # @param ["save_every_n_epochs", "save_n_epoch_ratio"] {allow-input: false}
save_n_epochs_type_value = 1  # @param {type:"number"}
save_model_as = "safetensors"  # @param ["ckpt", "pt", "safetensors"] {allow-input: false}
max_token_length = 225  # @param {type:"number"}
clip_skip = 2  # @param {type:"number"}
gradient_checkpointing = False  # @param {type:"boolean"}
gradient_accumulation_steps = 1  # @param {type:"number"}
seed = -1  # @param {type:"number"}
logging_dir = os.path.join(root_dir,'logs')
prior_loss_weight = 1.0

os.chdir(repo_dir)

sample_str = f"""
  masterpiece, best quality, food photo,  \
  --n lowres, blurry \
  --w 512 \
  --h 768 \
  --l 7 \
  --s 28
"""

config = {
    "model_arguments": {
        "v2": v2,
        "v_parameterization": v_parameterization if v2 and v_parameterization else False,
        "pretrained_model_name_or_path": pretrained_model_name_or_path,
        "vae": vae,
    },
    "additional_network_arguments": {
        "no_metadata": False,
        "unet_lr": float(unet_lr) if train_unet else None,
        "text_encoder_lr": float(text_encoder_lr) if train_text_encoder else None,
        "network_weights": network_weight,
        "network_module": network_module,
        "network_dim": network_dim,
        "network_alpha": network_alpha,
        "network_args": network_args,
        "network_train_unet_only": True if train_unet and not train_text_encoder else False,
        "network_train_text_encoder_only": True if train_text_encoder and not train_unet else False,
        "training_comment": None,
    },
    "optimizer_arguments": {
        "min_snr_gamma": min_snr_gamma if not min_snr_gamma == -1 else None,
        "optimizer_type": optimizer_type,
        "learning_rate": unet_lr,
        "max_grad_norm": 1.0,
        "optimizer_args": eval(optimizer_args) if optimizer_args else None,
        "lr_scheduler": lr_scheduler,
        "lr_warmup_steps": lr_warmup_steps,
        "lr_scheduler_num_cycles": lr_scheduler_num_cycles if lr_scheduler == "cosine_with_restarts" else None,
        "lr_scheduler_power": lr_scheduler_power if lr_scheduler == "polynomial" else None,
    },
    "dataset_arguments": {
        "cache_latents": True,
        "debug_dataset": False,
        "vae_batch_size": vae_batch_size,
    },
    "training_arguments": {
        "output_dir": output_dir,
        "output_name": project_name,
        "save_precision": save_precision,
        "save_every_n_epochs": save_n_epochs_type_value if save_n_epochs_type == "save_every_n_epochs" else None,
        "save_n_epoch_ratio": save_n_epochs_type_value if save_n_epochs_type == "save_n_epoch_ratio" else None,
        "save_last_n_epochs": None,
        "save_state": None,
        "save_last_n_epochs_state": None,
        "resume": None,
        "train_batch_size": train_batch_size,
        "max_token_length": 225,
        "mem_eff_attn": False,
        "xformers": True,
        "max_train_epochs": num_epochs,
        "max_data_loader_n_workers": 8,
        "persistent_data_loader_workers": True,
        "seed": seed if seed > 0 else None,
        "gradient_checkpointing": gradient_checkpointing,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "mixed_precision": mixed_precision,
        "clip_skip": clip_skip if not v2 else None,
        "logging_dir": logging_dir,
        "log_prefix": project_name,
        "noise_offset": noise_offset if noise_offset > 0 else None,
        "lowram": lowram,
    },
    "sample_prompt_arguments": {
        "sample_every_n_steps": None,
        "sample_every_n_epochs": 1 if enable_sample_prompt else 999999,
        "sample_sampler": sampler,
    },
    "dreambooth_arguments": {
        "prior_loss_weight": 1.0,
    },
    "saving_arguments": {
        "save_model_as": save_model_as
    },
}

config_path = os.path.join(config_dir, "config_file.toml")
prompt_path = os.path.join(config_dir, "sample_prompt.txt")

for key in config:
    if isinstance(config[key], dict):
        for sub_key in config[key]:
            if config[key][sub_key] == "":
                config[key][sub_key] = None
    elif config[key] == "":
        config[key] = None

config_str = toml.dumps(config)

def write_file(filename, contents):
    with open(filename, "w") as f:
        f.write(contents)

write_file(config_path, config_str)
write_file(prompt_path, sample_str)

print(config_str)


# In[ ]:


#@title ## 5.5. Start Training

#@markdown Check your config here if you want to edit something: 
#@markdown - `sample_prompt` : /content/LoRA/config/sample_prompt.txt
#@markdown - `config_file` : /content/LoRA/config/config_file.toml
#@markdown - `dataset_config` : /content/LoRA/config/dataset_config.toml

#@markdown Generated sample can be seen here: /content/LoRA/output/sample


#@markdown You can import config from another session if you want.
sample_prompt = "/home/daryna-lab-course/text2food/data/kohya-trainer/LoRA/config/sample_prompt.txt" #@param {type:'string'}
config_file = "/home/daryna-lab-course/text2food/data/kohya-trainer/LoRA/config/config_file.toml" #@param {type:'string'}
dataset_config = "/home/daryna-lab-course/text2food/data/kohya-trainer/LoRA/config/dataset_config.toml" #@param {type:'string'}

accelerate_conf = {
    "config_file" : accelerate_config,
    "num_cpu_threads_per_process" : 1,
}

train_conf = {
    "sample_prompts" : sample_prompt,
    "dataset_config" : dataset_config,
    "config_file" : config_file
}

def train(config):
    args = ""
    for k, v in config.items():
        if k.startswith("_"):
            args += f'"{v}" '
        elif isinstance(v, str):
            args += f'--{k}="{v}" '
        elif isinstance(v, bool) and v:
            args += f"--{k} "
        elif isinstance(v, float) and not isinstance(v, bool):
            args += f"--{k}={v} "
        elif isinstance(v, int) and not isinstance(v, bool):
            args += f"--{k}={v} "

    return args

accelerate_args = train(accelerate_conf)
train_args = train(train_conf)
final_args = f"accelerate launch {accelerate_args} /home/daryna-lab-course/text2food/data/kohya-trainer/train_network.py {train_args}"
print(os.listdir())
os.chdir("/home/daryna-lab-course/text2food/data/kohya-trainer")
print(os.listdir())
os.system(final_args)


# # # VI. Testing

# # In[ ]:


# # @title ## 6.1. Visualize loss graph (Optional)
# training_logs_path = "/logs"  # @param {type : "string"}

# os.chdir(repo_dir)
# # %load_ext tensorboard
# # %tensorboard --logdir {training_logs_path}

# # In[ ]:


# # @title ## 6.2. Interrogating LoRA Weights
# # @markdown Now you can check if your LoRA trained properly.
# import os
# import torch
# import json
# from safetensors.torch import load_file
# from safetensors.torch import safe_open

# # @markdown If you used `clip_skip = 2` during training, the values of `lora_te_text_model_encoder_layers_11_*` will be `0.0`, this is normal. These layers are not trained at this value of `Clip Skip`.
# network_weight = ""  # @param {'type':'string'}
# verbose = False  # @param {type:"boolean"}

# def is_safetensors(path):
#     return os.path.splitext(path)[1].lower() == ".safetensors"

# def load_weight_data(file_path):
#     if is_safetensors(file_path):
#         return load_file(file_path)
#     else:
#         return torch.load(file_path, map_location="cuda")

# def extract_lora_weights(weight_data):
#     lora_weights = [
#         (key, weight_data[key])
#         for key in weight_data.keys()
#         if "lora_up" in key or "lora_down" in key
#     ]
#     return lora_weights

# def print_lora_weight_stats(lora_weights):
#     print(f"Number of LoRA modules: {len(lora_weights)}")

#     for key, value in lora_weights:
#         value = value.to(torch.float32)
#         print(f"{key}, {torch.mean(torch.abs(value))}, {torch.min(torch.abs(value))}")

# def print_metadata(file_path):
#     if is_safetensors(file_path):
#         with safe_open(file_path, framework="pt") as f:
#             metadata = f.metadata()
#         if metadata is not None:
#             print(f"\nLoad metadata for: {file_path}")
#             print(json.dumps(metadata, indent=4))
#     else:
#         print("No metadata saved, your model is not in safetensors format")

# def main(file_path, verbose: bool):
#     weight_data = load_weight_data(file_path)

#     if verbose:
#         lora_weights = extract_lora_weights(weight_data)
#         print_lora_weight_stats(lora_weights)

#     print_metadata(file_path)

# if __name__ == "__main__":
#     main(network_weight, verbose)


# In[ ]:


# @title ## 6.3. Inference
# get_ipython().run_line_magic('store', '-r')

# @markdown ### LoRA Config
# @markdown Currently, `LoHa` and `LoCon_Lycoris` are not supported. Please run `Portable Web UI` instead
# network_weight = ""  # @param {'type':'string'}
# network_mul = 0.7  # @param {type:"slider", min:-1, max:2, step:0.05}
# network_module = "networks.lora"
# network_args = ""

# # @markdown ### <br> General Config
# v2 = False  # @param {type:"boolean"}
# v_parameterization = False  # @param {type:"boolean"}
# prompt = "masterpiece, best quality, 1girl, aqua eyes, baseball cap, blonde hair, closed mouth, earrings, green background, hat, hoop earrings, jewelry, looking at viewer, shirt, short hair, simple background, solo, upper body, yellow shirt"  # @param {type: "string"}
# negative = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"  # @param {type: "string"}
# model = "/content/pretrained_model/AnyLoRA.safetensors"  # @param {type: "string"}
# vae = ""  # @param {type: "string"}
# outdir = "/content/tmp"  # @param {type: "string"}
# scale = 7  # @param {type: "slider", min: 1, max: 40}
# sampler = "ddim"  # @param ["ddim", "pndm", "lms", "euler", "euler_a", "heun", "dpm_2", "dpm_2_a", "dpmsolver","dpmsolver++", "dpmsingle", "k_lms", "k_euler", "k_euler_a", "k_dpm_2", "k_dpm_2_a"]
# steps = 28  # @param {type: "slider", min: 1, max: 100}
# precision = "fp16"  # @param ["fp16", "bf16"] {allow-input: false}
# width = 512  # @param {type: "integer"}
# height = 768  # @param {type: "integer"}
# images_per_prompt = 4  # @param {type: "integer"}
# batch_size = 4  # @param {type: "integer"}
# clip_skip = 2  # @param {type: "slider", min: 1, max: 40}
# seed = -1  # @param {type: "integer"}

# final_prompt = f"{prompt} --n {negative}"

# config = {
#     "v2": v2,
#     "v_parameterization": v_parameterization,
#     "network_module": network_module,
#     "network_weight": network_weight,
#     "network_mul": float(network_mul),
#     "network_args": eval(network_args) if network_args else None,
#     "ckpt": model,
#     "outdir": outdir,
#     "xformers": True,
#     "vae": vae if vae else None,
#     "fp16": True,
#     "W": width,
#     "H": height,
#     "seed": seed if seed > 0 else None,
#     "scale": scale,
#     "sampler": sampler,
#     "steps": steps,
#     "max_embeddings_multiples": 3,
#     "batch_size": batch_size,
#     "images_per_prompt": images_per_prompt,
#     "clip_skip": clip_skip if not v2 else None,
#     "prompt": final_prompt,
# }

# args = ""
# for k, v in config.items():
#     if k.startswith("_"):
#         args += f'"{v}" '
#     elif isinstance(v, str):
#         args += f'--{k}="{v}" '
#     elif isinstance(v, bool) and v:
#         args += f"--{k} "
#     elif isinstance(v, float) and not isinstance(v, bool):
#         args += f"--{k}={v} "
#     elif isinstance(v, int) and not isinstance(v, bool):
#         args += f"--{k}={v} "

# final_args = f"python gen_img_diffusers.py {args}"

# os.chdir(repo_dir)
# get_ipython().system('{final_args}')


# # In[ ]:


# #@title ## 6.4. Launch Portable Web UI
# import os
# import random
# import shutil
# import zipfile
# import time
# import json
# from google.colab import drive
# from datetime import timedelta
# from subprocess import getoutput
# from IPython.display import clear_output, display, HTML
# from IPython.utils import capture
# from tqdm import tqdm

# webui_dir = os.path.join(root_dir, "stable-diffusion-webui")
# tmp_dir = os.path.join(root_dir, "tmp")
# patches_dir = os.path.join(root_dir, "patches")
# deps_dir = os.path.join(root_dir, "deps")
# extensions_dir = os.path.join(webui_dir, "extensions")
# control_dir = os.path.join(webui_dir, "models/ControlNet")

# webui_models_dir = os.path.join(webui_dir, "models/Stable-diffusion")
# webui_lora_dir = os.path.join(webui_dir, "models/Lora")
# webui_vaes_dir = os.path.join(webui_dir, "models/VAE")

# control_net_max_models_num = 2
# theme = "ogxBGreen"

# default_prompt = "masterpiece, best quality,"
# default_neg_prompt = "(worst quality, low quality:1.4)"
# default_sampler = "DPM++ 2M Karras"
# default_steps = 20
# default_width = 512
# default_height = 768
# default_denoising_strength = 0.55
# default_cfg_scale = 7

# config_file = os.path.join(webui_dir, "config.json")
# ui_config_file = os.path.join(webui_dir, "ui-config.json")
# webui_style_path = os.path.join(webui_dir, "style.css")

# os.chdir(root_dir)

# for dir in [patches_dir, deps_dir]:
#     os.makedirs(dir, exist_ok=True)

# package_url = [
#     f"https://huggingface.co/Linaqruf/fast-repo/resolve/main/anapnoe-webui.tar.lz4",
#     f"https://huggingface.co/Linaqruf/fast-repo/resolve/main/anapnoe-webui-deps.tar.lz4",
#     f"https://huggingface.co/Linaqruf/fast-repo/resolve/main/anapnoe-webui-cache.tar.lz4",
# ]

# def pre_download(desc):
#     for package in tqdm(package_url, desc=desc):
#         with capture.capture_output() as cap:
#             package_name = os.path.basename(package)
#             get_ipython().system('aria2c --console-log-level=error --summary-interval=10 -c -x 16 -k 1M -s 16 -d {root_dir} -o {package_name} {package}')
#             if package_name == f"anapnoe-webui-deps.tar.lz4":
#                 get_ipython().system('tar -xI lz4 -f {package_name} --overwrite-dir --directory=/usr/local/lib/python3.10/dist-packages/')
#             else:
#                 get_ipython().system('tar -xI lz4 -f {package_name} --directory=/')
#             os.remove(package_name)
#             del cap

#     if os.path.exists("/usr/local/lib/python3.10/dist-packages/ffmpy-0.3.0.dist-info"):
#         shutil.rmtree("/usr/local/lib/python3.10/dist-packages/ffmpy-0.3.0.dist-info")

#     s = getoutput("nvidia-smi")
#     with capture.capture_output() as cap:
#         if not "T4" in s:
#             get_ipython().system('pip uninstall -y xformers')
#             get_ipython().system('pip install -q xformers==0.0.18 triton')
#         del cap


# def read_config(filename):
#     if filename.endswith(".json"):
#         with open(filename, "r") as f:
#           config = json.load(f)
#     else:
#         with open(filename, 'r') as f:
#           config = f.read()
#     return config


# def write_config(filename, config):
#     if filename.endswith(".json"):
#         with open(filename, "w") as f:
#             json.dump(config, f, indent=4)
#     else:
#         with open(filename, 'w', encoding="utf-8") as f:
#             f.write(config)


# def open_theme(filename):
#     themes_folder = os.path.join(webui_dir, "extensions-builtin/sd_theme_editor/themes")
#     themes_file = os.path.join(themes_folder, f"{filename}.css")
#     webui_style_path = os.path.join(webui_dir, "style.css")

#     style_config = read_config(webui_style_path)
#     style_css_contents = style_config.split("/*BREAKPOINT_CSS_CONTENT*/")[1]

#     theme_config = read_config(themes_file)
#     style_data = ":host{" + theme_config + "}" + "/*BREAKPOINT_CSS_CONTENT*/" + style_css_contents
#     write_config(webui_style_path, style_data)


# def change_config(filename):
#     config = read_config(filename)
#     if not "stable-diffusion-webui" in config["disabled_extensions"]:
#         config["disabled_extensions"].append("stable-diffusion-webui")
#     config["outdir_txt2img_samples"] = os.path.join(tmp_dir, "outputs/txt2img-images")
#     config["outdir_img2img_samples"] = os.path.join(tmp_dir, "outputs/img2img-images")
#     config["outdir_extras_samples"] = os.path.join(tmp_dir, "outputs/extras-images")
#     config["outdir_txt2img_grids"] = os.path.join(tmp_dir, "outputs/txt2img-grids")
#     config["outdir_img2img_grids"] = os.path.join(tmp_dir, "outputs/img2img-grids")
#     config["outdir_save"] = os.path.join(tmp_dir, "log/images")
#     config["control_net_max_models_num"] = control_net_max_models_num
#     config["control_net_models_path"] = control_dir
#     config["control_net_allow_script_control"] = True
#     config["additional_networks_extra_lora_path"] = webui_lora_dir
#     config["CLIP_stop_at_last_layers"] = 2
#     config["eta_noise_seed_delta"] = 0
#     config["show_progress_every_n_steps"] = 10
#     config["show_progressbar"] = True
#     config["quicksettings"] = "sd_model_checkpoint, sd_vae, CLIP_stop_at_last_layers, use_old_karras_scheduler_sigmas, always_discard_next_to_last_sigma"
#     write_config(filename, config)


# def change_ui_config(filename):
#     config = read_config(filename)
#     config["txt2img/Prompt/value"] = default_prompt
#     config["txt2img/Negative prompt/value"] = default_neg_prompt
#     config["txt2img/Sampling method/value"] = default_sampler
#     config["txt2img/Sampling steps/value"] = default_steps
#     config["txt2img/Width/value"] = default_width
#     config["txt2img/Height/value"] = default_height
#     config["txt2img/Upscaler/value"] = "Latent (nearest-exact)"
#     config["txt2img/Denoising strength/value"] = default_denoising_strength
#     config["txt2img/CFG Scale/value"] = default_cfg_scale
#     config["img2img/Prompt/value"] = default_prompt
#     config["img2img/Negative prompt/value"] = default_neg_prompt
#     config["img2img/Sampling method/value"] = default_sampler
#     config["img2img/Sampling steps/value"] = default_steps
#     config["img2img/Width/value"] = default_width
#     config["img2img/Height/value"] = default_height
#     config["img2img/Denoising strength/value"] = default_denoising_strength
#     config["img2img/CFG Scale/value"] = default_cfg_scale
#     write_config(filename, config)


# def update_extensions():
#     start_time = time.time()
#     extensions_updated = []
#     with tqdm(
#         total=len(os.listdir(extensions_dir)),
#         desc="[1;32mUpdating extensions",
#         mininterval=0,
#     ) as pbar:
#         for dir in os.listdir(extensions_dir):
#             if os.path.isdir(os.path.join(extensions_dir, dir)):
#                 os.chdir(os.path.join(extensions_dir, dir))
#                 try:
#                     with capture.capture_output() as cap:
#                         get_ipython().system('git fetch origin')
#                         get_ipython().system('git pull')
#                 except Exception as e:
#                     print(f"[1;32mAn error occurred while updating {dir}: {e}")

#                 output = cap.stdout.strip()
#                 if "Already up to date." not in output:
#                     extensions_updated.append(dir)
#                 pbar.update(1)

#     print("\n")
#     for ext in extensions_updated:
#         print(f"[1;32m- {ext} updated to new version")

#     end_time = time.time()
#     elapsed_time = int(end_time - start_time)

#     if elapsed_time < 60:
#         print(f"\n[1;32mAll extensions are up to date. Took {elapsed_time} sec")
#     else:
#         mins, secs = divmod(elapsed_time, 60)
#         print(f"\n[1;32mAll extensions are up to date. Took {mins} mins {secs} sec")


# def main():
#     start_time = time.time()

#     print("[1;32mInstalling...\n")

#     if not os.path.exists(webui_dir):
#         desc = "[1;32mUnpacking Webui"
#         pre_download(desc)
#     else:
#         print("[1;32mAlready installed, skipping...")

#     with capture.capture_output() as cap:
#         os.chdir(os.path.join(webui_dir, "repositories/stable-diffusion-stability-ai"))
#         get_ipython().system('git apply {patches_dir}/stablediffusion-lowram.patch')

#         get_ipython().system('sed -i "s@os.path.splitext(checkpoint_.*@os.path.splitext(checkpoint_file); map_location=\'cuda\'@" {webui_dir}/modules/sd_models.py')
#         get_ipython().system("sed -i 's@ui.create_ui().*@ui.create_ui();shared.demo.queue(concurrency_count=999999,status_update_rate=0.1)@' {webui_dir}/webui.py")

#         get_ipython().system('sed -i "s@\'cpu\'@\'cuda\'@" {webui_dir}/modules/extras.py')
#         del cap
      
#     end_time = time.time()
#     elapsed_time = int(end_time - start_time)

#     change_config(config_file)
#     change_ui_config(ui_config_file)
#     open_theme(theme)

#     if elapsed_time < 60:
#         print(f"[1;32mFinished unpacking. Took {elapsed_time} sec")
#     else:
#         mins, secs = divmod(elapsed_time, 60)
#         print(f"[1;32mFinished unpacking. Took {mins} mins {secs} sec")

#     update_extensions()

#     #@markdown > Get <b>your</b> `ngrok_token` [here](https://dashboard.ngrok.com/get-started/your-authtoken) 
#     ngrok_token = "" #@param {type: 'string'}
#     ngrok_region = "ap" #@param ["us", "eu", "au", "ap", "sa", "jp", "in"]

#     with capture.capture_output() as cap:
#       for file in os.listdir(output_dir):
#         file_path = os.path.join(output_dir, file)
#         if file_path.endswith((".safetensors", ".pt", ".ckpt")):
#           get_ipython().system('ln "{file_path}" {webui_lora_dir}')

#       for file in os.listdir(pretrained_model):
#         file_path = os.path.join(pretrained_model, file)
#         if file_path.endswith((".safetensors", ".ckpt")):
#           get_ipython().system('ln "{file_path}" {webui_models_dir}')

#       for file in os.listdir(vae_dir):
#         file_path = os.path.join(vae_dir, file)
#         if file_path.endswith(".vae.pt"):
#           get_ipython().system('ln "{file_path}" {webui_vaes_dir}')

#       del cap

#     os.chdir(webui_dir)

#     print("[1;32m")

#     config = {
#         "enable-insecure-extension-access": True,
#         "disable-safe-unpickle": True,
#         "multiple": True if not ngrok_token else False,
#         "ckpt-dir": webui_models_dir,
#         "vae-dir": webui_vaes_dir,
#         "share": True if not ngrok_token else False,
#         "no-half-vae": True,
#         "lowram": True,
#         "gradio-queue": True,
#         "no-hashing": True,
#         "disable-console-progressbars": True,
#         "ngrok": ngrok_token if ngrok_token else None,
#         "ngrok-region": ngrok_region if ngrok_token else None,
#         "xformers": True,
#         "opt-sub-quad-attention": True,
#         "opt-channelslast": True,
#         "theme": "dark"
#     }

#     args = ""
#     for k, v in config.items():
#         if k.startswith("_"):
#             args += f'"{v}" '
#         elif isinstance(v, str):
#             args += f'--{k}="{v}" '
#         elif isinstance(v, bool) and v:
#             args += f"--{k} "
#         elif isinstance(v, float) and not isinstance(v, bool):
#             args += f"--{k}={v} "
#         elif isinstance(v, int) and not isinstance(v, bool):
#             args += f"--{k}={v} "

#     final_args = f"python launch.py {args}"

#     os.chdir(webui_dir)
#     get_ipython().system('{final_args}')

# main()


# # # VII. Deployment

# # In[ ]:


# # @title ## 7.1. Upload Config
# from huggingface_hub import login
# from huggingface_hub import HfApi
# from huggingface_hub.utils import validate_repo_id, HfHubHTTPError

# # @markdown Login to Huggingface Hub
# # @markdown > Get **your** huggingface `WRITE` token [here](https://huggingface.co/settings/tokens)
# write_token = ""  # @param {type:"string"}
# # @markdown Fill this if you want to upload to your organization, or just leave it empty.
# orgs_name = ""  # @param{type:"string"}
# # @markdown If your model/dataset repo does not exist, it will automatically create it.
# model_name = "your-model-name"  # @param{type:"string"}
# dataset_name = "your-dataset-name"  # @param{type:"string"}
# make_private = False  # @param{type:"boolean"}

# def authenticate(write_token):
#     login(write_token, add_to_git_credential=True)
#     api = HfApi()
#     return api.whoami(write_token), api


# def create_repo(api, user, orgs_name, repo_name, repo_type, make_private=False):
#     global model_repo
#     global datasets_repo
    
#     if orgs_name == "":
#         repo_id = user["name"] + "/" + repo_name.strip()
#     else:
#         repo_id = orgs_name + "/" + repo_name.strip()

#     try:
#         validate_repo_id(repo_id)
#         api.create_repo(repo_id=repo_id, repo_type=repo_type, private=make_private)
#         print(f"{repo_type.capitalize()} repo '{repo_id}' didn't exist, creating repo")
#     except HfHubHTTPError as e:
#         print(f"{repo_type.capitalize()} repo '{repo_id}' exists, skipping create repo")
    
#     if repo_type == "model":
#         model_repo = repo_id
#         print(f"{repo_type.capitalize()} repo '{repo_id}' link: https://huggingface.co/{repo_id}\n")
#     else:
#         datasets_repo = repo_id
#         print(f"{repo_type.capitalize()} repo '{repo_id}' link: https://huggingface.co/datasets/{repo_id}\n")

# user, api = authenticate(write_token)

# if model_name:
#     create_repo(api, user, orgs_name, model_name, "model", make_private)
# if dataset_name:
#     create_repo(api, user, orgs_name, dataset_name, "dataset", make_private)


# # ## 7.2. Upload with Huggingface Hub

# # In[ ]:


# # @title ### 7.2.1. Upload LoRA
# from huggingface_hub import HfApi
# from pathlib import Path

# get_ipython().run_line_magic('store', '-r')

# api = HfApi()

# # @markdown This will be uploaded to model repo
# model_path = "/content/LoRA/output"  # @param {type :"string"}
# path_in_repo = ""  # @param {type :"string"}

# # @markdown Now you can save your config file for future use
# config_path = "/content/LoRA/config"  # @param {type :"string"}

# # @markdown Other Information
# commit_message = ""  # @param {type :"string"}

# if not commit_message:
#     commit_message = f"feat: upload {project_name} lora model"

# def upload_to_hf(model_path, is_folder, is_config):
#     path_obj = Path(model_path)
#     trained_model = path_obj.parts[-1]

#     if path_in_repo:
#         trained_model = path_in_repo

#     if is_config:
#         trained_model = f"{project_name}_config"

#     print(f"Uploading {trained_model} to https://huggingface.co/{model_repo}")
#     print("Please wait...")

#     if is_folder:
#         api.upload_folder(
#             folder_path=model_path,
#             path_in_repo=trained_model,
#             repo_id=model_repo,
#             commit_message=commit_message,
#             ignore_patterns=".ipynb_checkpoints",
#         )
#         print(f"Upload success, located at https://huggingface.co/{model_repo}/tree/main\n")
#     else:
#         api.upload_file(
#             path_or_fileobj=model_path,
#             path_in_repo=trained_model,
#             repo_id=model_repo,
#             commit_message=commit_message,
#         )
#         print(f"Upload success, located at https://huggingface.co/{model_repo}/blob/main/{trained_model}\n")

# def upload():
#     is_model_file = model_path.endswith((".ckpt", ".safetensors", ".pt"))
#     upload_to_hf(model_path, not is_model_file, False)

#     if config_path:
#         upload_to_hf(config_path, True, True)

# upload()


# # In[ ]:


# # @title ### 7.2.2. Upload Dataset
# from huggingface_hub import HfApi
# from pathlib import Path
# import shutil
# import zipfile
# import os

# api = HfApi()

# # @markdown This will be compressed to zip and  uploaded to datasets repo, leave it empty if not necessary
# train_data_path = "/content/LoRA/train_data"  # @param {type :"string"}

# # @markdown `Nerd stuff, only if you want to save training logs`
# logs_path = "/content/LoRA/logs"  # @param {type :"string"}

# tmp_dataset = f"/content/LoRA/{project_name}_dataset" if project_name else "/content/LoRA/tmp_dataset"
# tmp_train_data = f"{tmp_dataset}/train_data"
# dataset_zip = f"{tmp_dataset}.zip"

# # @markdown Other Information
# commit_message = ""  # @param {type :"string"}

# if not commit_message:
#     commit_message = f"feat: upload {project_name} dataset and logs"

# os.makedirs(tmp_dataset, exist_ok=True)
# os.makedirs(tmp_train_data, exist_ok=True)

# def upload_dataset(dataset_path, is_zip):
#     path_obj = Path(dataset_path)
#     dataset_name = path_obj.parts[-1]

#     print(f"Uploading {dataset_name} to https://huggingface.co/datasets/{datasets_repo}")
#     print("Please wait...")

#     if is_zip:
#         api.upload_file(
#             path_or_fileobj=dataset_path,
#             path_in_repo=dataset_name,
#             repo_id=datasets_repo,
#             repo_type="dataset",
#             commit_message=commit_message,
#         )
#         print(f"Upload success, located at https://huggingface.co/datasets/{datasets_repo}/blob/main/{dataset_name}\n")
#     else:
#         api.upload_folder(
#             folder_path=dataset_path,
#             path_in_repo=dataset_name,
#             repo_id=datasets_repo,
#             repo_type="dataset",
#             commit_message=commit_message,
#             ignore_patterns=".ipynb_checkpoints",
#         )
#         print(f"Upload success, located at https://huggingface.co/datasets/{datasets_repo}/tree/main/{dataset_name}\n")

# def zip_file(folder_path):
#     zip_path = f"{folder_path}.zip"
#     with zipfile.ZipFile(zip_path, "w") as zip_file:
#         for root, dirs, files in os.walk(folder_path):
#             for file in files:
#                 zip_file.write(os.path.join(root, file))

# def move(src_path, dst_path, move_metadata):
#     metadata_files = [
#         "meta_cap.json",
#         "meta_cap_dd.json",
#         "meta_lat.json",
#         "meta_clean.json",
#         "meta_final.json",
#     ]

#     if os.path.exists(src_path):
#         shutil.move(src_path, dst_path)

#     if move_metadata:
#         parent_meta_path = os.path.dirname(src_path)

#         for filename in os.listdir(parent_meta_path):
#             file_path = os.path.join(parent_meta_path, filename)
#             if filename in metadata_files:
#                 shutil.move(file_path, dst_path)

# def upload():
#     if train_data_path:
#         move(train_data_path, tmp_train_data, False)
#         zip_file(tmp_dataset)
#         upload_dataset(dataset_zip, True)
#         os.remove(dataset_zip)
#     if logs_path:
#         upload_dataset(logs_path, False)

# upload()


# # ## 7.3. Upload with GIT (Alternative)

# # In[ ]:


# # @title ### 7.3.1. Clone Repository

# clone_model = True  # @param {'type': 'boolean'}
# clone_dataset = True  # @param {'type': 'boolean'}

# def clone_repository(repo_url, local_path):
#     get_ipython().system('git lfs install --skip-smudge')
#     os.environ["GIT_LFS_SKIP_SMUDGE"] = "1"
#     get_ipython().system('git clone {repo_url} {local_path}')

# if clone_model:
#     clone_repository(f"https://huggingface.co/{model_repo}", f"/content/{model_name}")

# if clone_dataset:
#     clone_repository(f"https://huggingface.co/datasets/{datasets_repo}", f"/content/{dataset_name}")


# # In[ ]:


# # @title ### 7.3.2. Commit using Git
# import os

# os.chdir(root_dir)

# # @markdown Choose which repo you want to commit
# commit_model = True  # @param {'type': 'boolean'}
# commit_dataset = True  # @param {'type': 'boolean'}
# # @markdown Other Information
# commit_message = ""  # @param {type :"string"}

# if not commit_message:
#     commit_message = f"feat: upload {project_name} lora model and dataset"

# get_ipython().system('git config --global user.email "example@mail.com"')
# get_ipython().system('git config --global user.name "example"')

# def commit(repo_folder, commit_message):
#     os.chdir(os.path.join(root_dir, repo_folder))
#     get_ipython().system('git lfs install')
#     get_ipython().system('huggingface-cli lfs-enable-largefiles .')
#     get_ipython().system('git add .')
#     get_ipython().system('git commit -m "{commit_message}"')
#     get_ipython().system('git push')


# if commit_model:
#     commit(model_name, commit_message)

# if commit_dataset:
#     commit(dataset_name, commit_message)

