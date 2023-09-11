# [Text2food Paper](https://drive.google.com/file/d/1PgcdS6RmvcOIN32PbTyw_q9A7Mr8WEHU/view?usp=sharing)
This is a project aim to generate high quality food images from given prompt by finetuning Stable diffiusion 2.1 with LORA.

![Alternative Text](poster.png)


## Data Processing
Our pipeline consist of 2 different code MMC4 and Special food collection. Run the corresponding files in order. 

We used [MMC4 Multimodal-C4 core fewer-faces dataset ](https://github.com/allenai/mmc4#corpus-stats-v11) to curate our final dataset. 

## Training
The training script kohya_LoRA_dreambooth_final.py can be found under the training folder.

## Inference
Simple run the UI ipynb file and dont forget to add your ngrok token. It will create an amazing UI, you should select your model before testing it. You can use our model `mmc4+unique-food-continued-000010.safetensors` to test it

 **[!WARNING]**
Due to the constrained capacity of GitLab repositories, we have chosen not to upload the bulk of the data utilized in this project. Consequently, running the provided files may not consistently yield successful outcomes. For a comprehensive representation of our work, we have stored our complete efforts on Google Drive, accessible through the account details that have been provided to us.