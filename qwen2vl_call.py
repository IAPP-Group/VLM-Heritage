import argparse
import os
import pandas as pd

from PIL import Image 
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import re


def get_message(image_path, quest, memory=None):
    message = []
    if memory is not None:
        message = memory
        message.append({"role": "user", "content": quest})
    else:
        message = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": image_path,
                            },
                            {"type": "text", "text": quest},
                        ],
                    }
        ]

    return message


def run_qwen2vl_model(model, processor, message):
    # Preparation for inference
    text = processor.apply_chat_template(
        message, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(message)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    
    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=256)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    res1 = re.sub(r'\s+', ' ', output_text[0])
    return res1


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


parser = argparse.ArgumentParser(description="Runs Qwen2-VL on a Heritage dataset of images.")
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

    # Load the model on the available device(s)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
    )

    # processer with min-max pixel size 
    min_pixels = 256 * 28 * 28
    max_pixels = 1280 * 28 * 28
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels
    )
    

    # collect data
    image_list = [f"./dataset/{i}" for i in os.listdir("dataset/")]

    # read dataset
    df = pd.read_csv("dataset.csv", sep=";")
    
    for i in image_list:
        
        id_image = os.path.basename(i).split("_")[0].replace("i","")
        
        local_city = df.at[int(id_image)-1, "city"].strip()
        local_subject = df.at[int(id_image)-1, "subject"].strip()

        if "wiki" in os.path.basename(i):
            local_dataset = "wiki"
        else:    
            local_dataset = os.path.basename(i).split("_")[1].split(".")[0]


        # 1st question
        message = get_message(i, prompt[1])
        res1 = run_qwen2vl_model(model, processor, message)

        # 2nd question
        history = message
        history.append({"role": "assistant", "content": res1})
        message2 = get_message(i, prompt[2].replace("CITY", local_city), memory=history)
        res2 = run_qwen2vl_model(model, processor, message2)
 

        # save data
        content = f"{id_image}|{i}|{local_dataset}|{local_city}|{local_subject}|{res1}|{res2}\n";
        print(content)
        
        with open(f"qwen2vl_results_{lang}.csv", 'a') as file:
            file.write(content)
        