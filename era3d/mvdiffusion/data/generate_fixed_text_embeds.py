from transformers import CLIPTokenizer, CLIPTextModel
import torch
import os

root = '/mnt/data/lipeng/'
pretrained_model_name_or_path =  'stabilityai/stable-diffusion-2-1-unclip'


weight_dtype = torch.float16
device = torch.device("cuda:0")
tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder='text_encoder')
text_encoder = text_encoder.to(device, dtype=weight_dtype)

def generate_mv_embeds():
    path = './fixed_prompt_embeds_8view'
    os.makedirs(path, exist_ok=True)
    views = ["front", "front_right", "right", "back_right", "back", " back_left", "left", "front_left"]
    # views = ["front", "front_right", "right", "back", "left", "front_left"]
    # views = ["front", "right", "back", "left"]
    clr_prompt = [f"a rendering image of 3D models, {view} view, color map." for view in views]
    normal_prompt = [f"a rendering image of 3D models, {view} view, normal map." for view in views]


    for id, text_prompt in enumerate([clr_prompt, normal_prompt]):
        print(text_prompt)
        text_inputs = tokenizer(text_prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").to(device)
        text_input_ids = text_inputs.input_ids
        untruncated_ids = tokenizer(text_prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids):
            removed_text = tokenizer.batch_decode(
                untruncated_ids[:, tokenizer.model_max_length - 1 : -1]
            )
        if hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None
        prompt_embeds = text_encoder(text_input_ids.to(device), attention_mask=attention_mask,)
        prompt_embeds = prompt_embeds[0].detach().cpu()
        print(prompt_embeds.shape)


        # print(prompt_embeds.dtype)
        if id == 0:
            torch.save(prompt_embeds, f'./{path}/clr_embeds.pt')
        else:
            torch.save(prompt_embeds, f'./{path}/normal_embeds.pt')
    print('done')
    

def generate_img_embeds():
    path = './fixed_prompt_embeds_persp2ortho'
    os.makedirs(path, exist_ok=True)
    text_prompt = ["a orthogonal renderining image of 3D models"]
    print(text_prompt)
    text_inputs = tokenizer(text_prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").to(device)
    text_input_ids = text_inputs.input_ids
    untruncated_ids = tokenizer(text_prompt, padding="longest", return_tensors="pt").input_ids
    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
        text_input_ids, untruncated_ids):
        removed_text = tokenizer.batch_decode(
            untruncated_ids[:, tokenizer.model_max_length - 1 : -1]
        )
    if hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask:
        attention_mask = text_inputs.attention_mask.to(device)
    else:
        attention_mask = None
    prompt_embeds = text_encoder(text_input_ids.to(device), attention_mask=attention_mask,)
    prompt_embeds = prompt_embeds[0].detach().cpu()
    print(prompt_embeds.shape)

    # print(prompt_embeds.dtype)
 
    torch.save(prompt_embeds, f'./{path}/embeds.pt')
    print('done')
    
generate_img_embeds()