# LDA - Topic Clustering
## Prerequistes
MALLET Library -> https://mimno.github.io/Mallet/index  
In order to run Mallet Java must be install on the local machine  
Gensim 3.8.  
Python 3.8  
Some additional libraries listed in lda.py  
  
The data (shards) are downloaded from https://github.com/allenai/mmc4  
A sample shards file can be found in all_shards folder. Note that this folder is only a small subset of the actual data.  

  
## Description
This folder contains the code for doing topic clustering and data extraction from MMC4 library using the extracted words from MMC4 relating to food/cooking.  


## Usage
First run combine_json_files.py with  
```python3 combine_json_files.py --num_files [NUMBER OF FILES] output_file_name```  
output file can be anything, in our case we use 'combined.json'  
--num_files is optional. For [NUMBER OF FILES] 2500 seems to be more than enough. size ~2.8gb.  
Running lda.py with 5000 takes quite a lot of time.  

in order to run lda.py give a combined jsonl file. To be done -> iterate over a seperate jsonl files. Combining files into one makes the data exponentially large.  
``` python3 lda.py combined.jsonl ```  
  

Run extract_matches to extract the food related data.  
``` python3 extract_matches.py```   
It produces a file called extracted_data.json which contains image_name, image_url, and the corresponding text.  


## Authors and acknowledgment
Contributions: Arda Andirin - arda.andirin@tum.de  

Special thanks to contributors of MMC4 repo and Jack Hessel for providing us with the topic clustering code (lda.py).  

```
@article{zhu2023multimodal,
  title={{Multimodal C4}: An Open, Billion-scale Corpus of Images Interleaved With Text},
  author={Wanrong Zhu and Jack Hessel and Anas Awadalla and Samir Yitzhak Gadre and Jesse Dodge and Alex Fang and Youngjae Yu and Ludwig Schmidt and William Yang Wang and Yejin Choi},
  journal={arXiv preprint arXiv:2304.06939},
  year={2023}
}
```