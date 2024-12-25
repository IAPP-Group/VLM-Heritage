from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import requests
import torch


import argparse
import os
import pandas as pd
import re
import numpy as np
from PIL import Image


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


def molmo_generation(model, processor, image_path, query):
    # process the image and text
    inputs = processor.process(images=[Image.open(image_path)], text=query)
    inputs["images"] = inputs["images"].to(torch.bfloat16)
    
    # move inputs to the correct device and make a batch of size 1
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
    
    # generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
    output = model.generate_from_batch(
        inputs,
        GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
        tokenizer=processor.tokenizer
    )
    
    # only get generated tokens; decode them to text
    generated_tokens = output[0,inputs['input_ids'].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return generated_text


parser = argparse.ArgumentParser(description="Runs MOLMO on a Heritage dataset of images.")
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


    # load the processor
    processor = AutoProcessor.from_pretrained(
        'allenai/Molmo-7B-O-0924',
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map='auto'
    )

    # load the model
    model = AutoModelForCausalLM.from_pretrained(
        'allenai/Molmo-7B-O-0924',
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map='auto'
    )
            

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


        # inference
        
        ## question1
        query1 = prompt[1]
        res1 = molmo_generation(model, processor, i, query1)
        res1 = re.sub(r'\s+', ' ', res1)
        #print(res1)
        
        ## question2 WITHOUT HISTORY
        query2 = prompt[2].replace("CITY", local_city)
        res2 = molmo_generation(model, processor, i, query2)
        res2 = re.sub(r'\s+', ' ', res2)
        #print(res2)


        # save data
        content = f"{id_image}|{i}|{local_dataset}|{local_city}|{local_subject}|{res1}|{res2}\n";
        with open(f"./results/molmo_results_{lang}.csv", 'a') as file:
            file.write(content)
