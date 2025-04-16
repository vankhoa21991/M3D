import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import simple_slice_viewer as ssv
import SimpleITK as sikt
import os

device = torch.device('cuda') # 'cpu', 'cuda'
dtype = torch.bfloat16 # or bfloat16, float16, float32

model_name_or_path = 'GoodBaiBai88/M3D-LaMed-Llama-2-7B'
proj_out_num = 256

# Prepare your 3D medical image:
# 1. The image shape needs to be processed as 1*32*256*256, consider resize and other methods.
# 2. The image needs to be normalized to 0-1, consider Min-Max Normalization.
# 3. The image format needs to be converted to .npy 
# 4. Although we did not train on 2D images, in theory, the 2D image can be interpolated to the shape of 1*32*256*256 for input.
image_path = "./Data/data/examples/example_03.npy"

model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=dtype,
    device_map='auto',
    trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    model_max_length=512,
    padding_side="right",
    use_fast=False,
    trust_remote_code=True
)

model = model.to(device=device)

# question = "Can you provide a caption consists of findings for this medical image?"
question = "What is liver in this image? Please output the segmentation mask."
# question = "What is liver in this image? Please output the box."

image_tokens = "<im_patch>" * proj_out_num
input_txt = image_tokens + question
input_id = tokenizer(input_txt, return_tensors="pt")['input_ids'].to(device=device)

image_np = np.load(image_path)
image_pt = torch.from_numpy(image_np).unsqueeze(0).to(dtype=dtype, device=device)

# generation = model.generate(image_pt, input_id, max_new_tokens=256, do_sample=True, top_p=0.9, temperature=1.0)
generation, seg_logit = model.generate(image_pt, input_id, seg_enable=True, max_new_tokens=256, do_sample=True, top_p=0.9, temperature=1.0)

generated_texts = tokenizer.batch_decode(generation, skip_special_tokens=True)
seg_mask = (torch.sigmoid(seg_logit) > 0.5) * 1.0
seg_mask = seg_mask.squeeze(0).cpu().numpy()
print(seg_mask.shape, image_np.shape)
print(np.unique(seg_mask))

print('question', question)
print('generated_texts', generated_texts[0])

image = sikt.GetImageFromArray(image_np[0])
sikt.WriteImage(image, './Data/data/example_03.nii.gz')
# ssv.display(image)
seg = sikt.GetImageFromArray(seg_mask[0])
# ssv.display(seg)
sikt.WriteImage(seg, './Data/data/example_03_seg.nii.gz')
