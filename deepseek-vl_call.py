from transformers import AutoModelForCausalLM
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images

import argparse
import os
import pandas as pd
import torch
import re


def get_conversation(image_path, quest, memory=None):
    conversation = []
    if memory is not None:
        conversation = memory
        conversation.append({"role": "User", "content": quest})
        conversation.append({"role": "Assistant", "content": ""})
    else:
        conversation = [
            {
                "role": "User",
                "content": "<image_placeholder>{}".format(quest),
                "images": [image_path]
            },
            {
                "role": "Assistant",
                "content": ""
            }
        ]

    return conversation


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


    # load model
    model_path = "deepseek-ai/deepseek-vl-7b-base"
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer
    
    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
    

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

        # run DeepSeekVL inference

        # 1st question
        conversation = get_conversation(i, prompt[1])
        pil_images = load_pil_images(conversation)
        prepare_inputs = vl_chat_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True
        ).to(vl_gpt.device)
        
        # run image encoder to get the image embeddings
        inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
        
        # run the model to get the response
        outputs = vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True
        )
        res1 = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        res1 = re.sub(r'\s+', ' ', res1)
        # print(i, res1)

        # 2nd question
        history = conversation
        history.append({"role": "Assistant", "content": res1})
        conversation = get_conversation(i, prompt[2].replace("CITY", local_city), memory=history)
        
        pil_images = load_pil_images(conversation)
        prepare_inputs = vl_chat_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True
        ).to(vl_gpt.device)
        
        # run image encoder to get the image embeddings
        inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
        
        # run the model to get the response
        outputs = vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True
        )
        res2 = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        res2 = re.sub(r'\s+', ' ', res2)
        # print(i, res2)

        # save data
        content = f"{id_image}|{i}|{local_dataset}|{local_city}|{local_subject}|{res1}|{res2}\n";
        with open(f"deepseekvl_results_{lang}.csv", 'a') as file:
            file.write(content)
