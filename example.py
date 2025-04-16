import os
import numpy as np
import torch
import SimpleITK as sikt
from pathlib import Path
import nibabel as nib
from scipy.ndimage import zoom
from transformers import AutoTokenizer, AutoModelForCausalLM

def normalize_volume(volume, min_val=-400, max_val=400):
    """Normalize volume to [0, 1] range"""
    volume = np.clip(volume, min_val, max_val)
    return (volume - min_val) / (max_val - min_val)

def process_volume(volume_path, mask_path, target_shape=(32, 256, 256)):
    """Process a single volume-mask pair"""
    # Read images
    volume = sikt.ReadImage(str(volume_path))
    mask = sikt.ReadImage(str(mask_path))
    
    # Convert to numpy arrays
    volume_np = sikt.GetArrayFromImage(volume)
    mask_np = sikt.GetArrayFromImage(mask)
    
    # Get z-axis bounds from mask
    z_coords = np.where(np.any(mask_np > 0, axis=(1, 2)))[0]
    if len(z_coords) == 0:
        print(f"Warning: Empty mask in {mask_path}")
        return None
    
    # Calculate z-axis bounding box
    z_min = max(0, np.min(z_coords) - 5)  # 5 slices padding
    z_max = min(volume_np.shape[0], np.max(z_coords) + 5)
    
    print(f"Z-axis bounds: {z_min} to {z_max}")
    
    # Crop volume only along z-axis
    cropped_volume = volume_np[z_min:z_max, :, :]
    
    # Calculate zoom factors for resizing
    # Only resize z-axis to match target depth
    z_zoom = target_shape[0] / cropped_volume.shape[0]
    x_zoom = target_shape[1] / cropped_volume.shape[1]
    y_zoom = target_shape[2] / cropped_volume.shape[2]
    
    # Resize using zoom
    resized_volume = zoom(cropped_volume, (z_zoom, x_zoom, y_zoom), order=1)
    
    # Normalize
    normalized_volume = normalize_volume(resized_volume)
    
    print(f"Final shape: {normalized_volume.shape}")
    
    normalized_volume = np.expand_dims(normalized_volume, axis=0)
    
    # Create SimpleITK image from normalized volume
    normalized_image = sikt.GetImageFromArray(normalized_volume[0])
    
    return normalized_volume, normalized_image

def main():
    # Set up paths
    volumes_dir = Path("/mnt/datalake/DS-lake/vankhoa/Task150_liver_malignant_patient_balance_2/raw_splitted/imagesTs")  # Replace with your volumes directory
    masks_dir = Path("/mnt/datalake/DS-lake/vankhoa/Task150_liver_malignant_patient_balance_2/raw_splitted/organTs")      # Replace with your masks directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Model setup
    device = torch.device('cuda')
    dtype = torch.bfloat16
    model_name_or_path = '/home/vankhoa@median.cad/code/public/M3D/M3D-LaMed-Llama-2-7B'
    proj_out_num = 256
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        device_map='auto',
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        model_max_length=512,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True
    )
    model = model.to(device=device)
    
    # Process each volume-mask pair
    for volume_path in volumes_dir.glob("*.nii.gz"):
        # Find corresponding mask
        mask_path = masks_dir / volume_path.name.replace('_0000','')
        if not mask_path.exists():
            print(f"Warning: No mask found for {volume_path.name}")
            continue
        
        print(f"Processing {volume_path.name}...")
        
        # Process volume
        processed_volume, processed_image = process_volume(volume_path, mask_path)
        if processed_volume is None:
            continue
        
        # Save processed volume
        base_name = volume_path.stem.replace('.nii.gz', '')
        sikt.WriteImage(processed_image, str(output_dir / f"{base_name}_processed.nii.gz"))
        
        # Prepare input for model
        image_pt = torch.from_numpy(processed_volume).unsqueeze(0).to(dtype=dtype, device=device)
        
        # Generate caption
        question = "Is there any abnormality in this image?"
        image_tokens = "<im_patch>" * proj_out_num
        input_txt = image_tokens + question
        input_id = tokenizer(input_txt, return_tensors="pt")['input_ids'].to(device=device)
        
        # Generate output
        generation, seg_logit = model.generate(
            image_pt, 
            input_id, 
            seg_enable=True, 
            max_new_tokens=256, 
            do_sample=True, 
            top_p=0.9, 
            temperature=1.0
        )
        
        # Decode and save results
        generated_texts = tokenizer.batch_decode(generation, skip_special_tokens=True)
        seg_mask = (torch.sigmoid(seg_logit) > 0.5) * 1.0
        
        # Save results
        with open(output_dir / f"{base_name}_caption.txt", "w") as f:
            f.write(generated_texts[0])
        
        # Save segmentation mask
        seg_image = sikt.GetImageFromArray(seg_mask.cpu().numpy()[0])
        sikt.WriteImage(seg_image, str(output_dir / f"{base_name}_segmentation.nii.gz"))
        
        print(f"Processed {volume_path.name}")
        print(f"Caption: {generated_texts[0]}")

if __name__ == "__main__":
    main()
