from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests
import argparse
import pandas as pd
import os
import re



def build_inputs(quest, memory=None):
    chat = []
    if memory is None:
       chat = [
            {
              "role": "user", 
              "content": [
                {"type": "text", "text": quest}, 
                {"type": "image"}
              ]
            }
        ]
    else:
        chat = [
            {
              "role": "user", 
                "content": [
                {"type": "text", "text": memory["quest"]}, 
                {"type": "image"}
              ]
            },
            {"role": "assistant", 
             "content" : [
                 {"type": "text", "text": memory["output"]}
             ]
            },
            {
              "role": "user", 
                "content": [
                {"type": "text", "text": quest}, 
              ]
            }
        ]
    
    chat = processor.apply_chat_template(chat, add_generation_prompt=True)
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


parser = argparse.ArgumentParser(description="Runs DeepSeek VL on a Heritage dataset of images.")
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


    # load processor and model
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
    model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True) 
    model.to("cuda:0")


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

        # load image
        image = Image.open(i)

        # run inference
        ## question 1
        chat1 = build_inputs(prompt[1])
        inputs = processor(images=image, text=chat1, return_tensors="pt").to("cuda:0")
        output = model.generate(**inputs, max_new_tokens=100)
        res1 = processor.decode(output[0], skip_special_tokens=True)
        res1 = re.sub(r'\s+', ' ', res1)
        res1 = res1.replace(f"[INST] {prompt[1]} [/INST] ", "")
        # print("res1:", res1)
        
        ## question 2
        memory = {"quest": prompt[1], "output": res1}
        chat2 = build_inputs(prompt[2].replace("CITY", local_city), memory=memory)
        inputs = processor(images=image, text=chat2, return_tensors="pt").to("cuda:0")
        
        output = model.generate(**inputs, max_new_tokens=100)
        res2 = processor.decode(output[0], skip_special_tokens=True)
        res2 = re.sub(r'\s+', ' ', res2)
        res2 = res2[len(chat2)-7:]
        # print("\n\nres2:", res2)
        
        
        # save data
        content = f"{id_image}|{i}|{local_dataset}|{local_city}|{local_subject}|{res1}|{res2}\n";
        with open(f"./results/llava1.6_results_{lang}.csv", 'a') as file:
            file.write(content)
