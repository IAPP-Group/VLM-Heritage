import argparse
import os
import pandas as pd

from PIL import Image 
import requests 
from transformers import AutoModelForCausalLM 
from transformers import AutoProcessor 
import torch

def build_inputs(image_path, quest, memory=None):
    images = []
    messages = []
    placeholder = ""

    if memory is not None:
        messages.append({"role": "user", "content": quest})
    else:
        image = Image.open(image_path)
        images.append(image)
        placeholder = f"<|image_1|>\n"
    
        messages = [
            {"role": "user", "content": placeholder+quest},
        ]
    
    prompt = processor.tokenizer.apply_chat_template(
      messages, 
      tokenize=False, 
      add_generation_prompt=True
    )

    inputs = processor(prompt, images, return_tensors="pt").to("cuda:0") 

    return messages, inputs


def get_response(inputs):
    generation_args = { 
        "max_new_tokens": 1000, 
        "temperature": 0.0, 
        "do_sample": False, 
    } 
    
    generate_ids = model.generate(**inputs, 
      eos_token_id=processor.tokenizer.eos_token_id, 
      **generation_args
    )
    
    # remove input tokens 
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(generate_ids, 
      skip_special_tokens=True, 
      clean_up_tokenization_spaces=False)[0] 
    
    return response


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


parser = argparse.ArgumentParser(description="Runs Phi3 vision on a Heritage dataset of images.")
parser.add_argument('--ita', action='store_true', help="Italian questioning")
parser.add_argument('--eng', action='store_true', help="English questioning")

if __name__ == "__main__":
    model_id = "microsoft/Phi-3.5-vision-instruct" 
    args = parser.parse_args()

    if args.ita:
        prompt = promts["ita"]
        lang = "ita"
    else:
        prompt = promts["eng"]
        lang = "eng"
    
    # Note: set _attn_implementation='eager' if you don't have flash_attn installed
    model = AutoModelForCausalLM.from_pretrained(
      model_id, 
      device_map="cuda", 
      trust_remote_code=True, 
      torch_dtype="auto", 
      _attn_implementation='eager'
      #_attn_implementation='flash_attention_2'    
    )
    
    # for best performance, use num_crops=4 for multi-frame, num_crops=16 for single-frame.
    processor = AutoProcessor.from_pretrained(model_id, 
      trust_remote_code=True, 
      num_crops=4,                                      
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
        
        # first question
        msg, inputs1 = build_inputs(i, prompt[1])
        res1 = get_response(inputs1)
        # print(i, res1)
        
        # second question
        msg1 = msg.append({"role": "assistant", "content": res1})
        msg2, inputs2 = build_inputs(i, prompt[2].replace("CITY", local_city), memory=msg1)
        res2 = get_response(inputs2)
        # print(i, res2)

        # save data
        content = f"{id_image}|{i}|{local_dataset}|{local_city}|{local_subject}|{res1}|{res2}\n";
        with open(f"phi3_results_{lang}.csv", 'a') as file:
            file.write(content)
        

        
    
    