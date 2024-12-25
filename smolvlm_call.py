import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image


import argparse
import os
import pandas as pd
import re
import numpy as np


def ask_first_question(img, q1):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": q1}
            ]
        },
    ]

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[img], return_tensors="pt")
    inputs = inputs.to(DEVICE)

    generated_ids = model.generate(**inputs, max_new_tokens=500)
    output_ids = generated_ids[:, inputs.input_ids.shape[-1]:]
    generated_texts = processor.batch_decode(
        output_ids,
        skip_special_tokens=True,
    )

    return generated_texts[0]


def ask_second_question(img, q1, a1, q2):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": q1}
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": a1}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": q2}
            ],
        },
    ]

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[img], return_tensors="pt")
    inputs = inputs.to(DEVICE)

    generated_ids = model.generate(**inputs, max_new_tokens=500)
    output_ids = generated_ids[:, inputs.input_ids.shape[-1]:]
    generated_texts = processor.batch_decode(
        output_ids,
        skip_special_tokens=True,
    )

    return generated_texts[0]


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


parser = argparse.ArgumentParser(description="Runs SmolVLM on a Heritage dataset of images.")
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



    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize processor and model
    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
    model = AutoModelForVision2Seq.from_pretrained(
        "HuggingFaceTB/SmolVLM-Instruct",
        torch_dtype=torch.bfloat16,
        _attn_implementation="eager",
    ).to(DEVICE)


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


        image1 = load_image(i)
        
        # run inference
        question1 = prompt[1]
        answer1 = ask_first_question(image1, question1)
        res1 = re.sub(r'\s+', ' ', answer1)
        # print(res1)
        
        question2 = prompt[2].replace("CITY", local_city)
        answer2 = ask_second_question(image1, question1, answer1, question2)
        res2 = re.sub(r'\s+', ' ', answer2)
        # print(res2)


        # save data
        content = f"{id_image}|{i}|{local_dataset}|{local_city}|{local_subject}|{res1}|{res2}\n";
        with open(f"./results/smolvlm_results_{lang}.csv", 'a') as file:
            file.write(content)

    

    