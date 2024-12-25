import argparse
import os
import pandas as pd

from PIL import Image 
import requests 
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
import torch
import re


def build_inputs(quest, memory=None):
    chat = []
    if memory is None:
       chat = [
            {
              "role": "user", "content": [
                {"type": "text", "content": quest}, 
                {"type": "image"}
              ]
            }
        ]
    else:
        chat = [
            {
              "role": "user", "content": [
                {"type": "text", "content": memory["quest"]}, 
                {"type": "image"}
              ]
            },
            {"role": "assistant", "content": memory["output"]},
            {
              "role": "user", "content": [
                {"type": "text", "content": quest}, 
              ]
            }
        ]
    
    chat = processor.apply_chat_template(chat)
    return chat
    


promts = {
    "ita":{
        1: "Qual è il soggetto rappresentato in questa immagine?",
        2: "Sei a CITY, qual è il soggetto rappresentato in questa immagine?"
    },
    "eng":{
        1: "Which is the subject captured in this picture?", 
        2: "You are in CITY, which is the subject captured in this picture?"
    }
}


parser = argparse.ArgumentParser(description="Runs Pixtral on a Heritage dataset of images.")
parser.add_argument('--ita', action='store_true', help="Italian questioning")
parser.add_argument('--eng', action='store_true', help="English questioning")

if __name__ == "__main__":
    args = parser.parse_args()

    if args.ita:
        prompt = promts["ita"]
        lang = "ita"
    else:
        prompt = promts["eng"]
        lang = "eng"



    quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16)

    model_id = "mistral-community/pixtral-12b"
    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(model_id,
                                                      torch_dtype=torch.float16,
                                                      device_map='cuda',
                                                      quantization_config=quantization_config)
    

    # collect data
    image_list = [f"./dataset/{i}" for i in os.listdir("dataset/")]

    # read dataset
    df = pd.read_csv("dataset.csv", sep=";")
    
    for i in image_list:

        print(i)
        
        id_image = os.path.basename(i).split("_")[0].replace("i","")
        
        local_city = df.at[int(id_image)-1, "city"].strip()
        local_subject = df.at[int(id_image)-1, "subject"].strip()

        if "wiki" in os.path.basename(i):
            local_dataset = "wiki"
        else:    
            local_dataset = os.path.basename(i).split("_")[1].split(".")[0]
        
        # first question
        chat1 = build_inputs(prompt[1])
        inputs1 = processor(text=chat1, images=[Image.open(i)], return_tensors="pt").to(model.device)
        generate_ids1 = model.generate(**inputs1, max_new_tokens=500)
        res1 = processor.batch_decode(generate_ids1, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        res1 = res1.replace(prompt[1], "")
        res1 = re.sub(r'\s+', ' ', res1)
        #print(i, res1)
        
        # second question
        memory = {"quest": prompt[1], "output": res1}
        chat2 = build_inputs(prompt[2].replace("CITY", local_city), memory=memory)
        inputs2 = processor(text=chat2, images=[Image.open(i)], return_tensors="pt").to(model.device)
        generate_ids2 = model.generate(**inputs2, max_new_tokens=500)
        res2 = processor.batch_decode(generate_ids2, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        res2 = res2.replace(prompt[1], "")
        res2 = res2.replace(prompt[2].replace("CITY", local_city), "")

        res2 = res2.replace("\n","")
        res2 = re.sub(r'\s+', ' ', res2)
        #print(i, res2)

        # save data
        content = f"{id_image}|{i}|{local_dataset}|{local_city}|{local_subject}|{res1}|{res2}\n";
        with open(f"pixtral_results2_{lang}.csv", 'a') as file:
            file.write(content)
        

        
    
    
