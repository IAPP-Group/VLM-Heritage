import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

import argparse
import os
import pandas as pd
import re


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


parser = argparse.ArgumentParser(description="Runs CogVLM2 on a Heritage dataset of images.")
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

    
    # load model
    MODEL_PATH = "THUDM/cogvlm2-llama3-chat-19B-int4"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    gen_kwargs = {
    "max_new_tokens": 2048,
    "pad_token_id": 128002,
    }
    TORCH_TYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[
        0] >= 8 else torch.float16
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=TORCH_TYPE,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).eval()
    

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

        # run CogVLM2 inference
        history = []
        query = prompt[1]
        image = Image.open(i).convert('RGB')
        input_by_model = model.build_conversation_input_ids(
                tokenizer,
                query=query,
                history=history,
                images=[image],
                template_version='chat')
        
        inputs = {
            'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
            'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
            'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
            'images': [[input_by_model['images'][0].to(DEVICE).to(TORCH_TYPE)]] if image is not None else None,
        }

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = tokenizer.decode(outputs[0])
            res1 = response.split("<|end_of_text|>")[0]
            # print("\nCogVLM2:", res1)
            
            res1 = re.sub(r'\s+', ' ', res1)
            history.append((query, res1))
            print(i, res1)


        # second question
        query = prompt[2].replace("CITY", local_city)
        input_by_model = model.build_conversation_input_ids(
                tokenizer,
                query=query,
                history=history,
                images=[],
                template_version='chat')
        
        inputs = {
            'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
            'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
            'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
            'images': None,
        }

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = tokenizer.decode(outputs[0])
            res2 = response.split("<|end_of_text|>")[0]
            # print("\nCogVLM2:", res2)
            res2 = re.sub(r'\s+', ' ', res2)
            print(i, res2)
    

        # save data
        content = f"{id_image}|{i}|{local_dataset}|{local_city}|{local_subject}|{res1}|{res2}\n";
        with open(f"cogvlm2_results_{lang}.csv", 'a') as file:
            file.write(content)

