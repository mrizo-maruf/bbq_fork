# use_clip_embeddings_to_lava.py
# Usage examples:
#  python use_clip_embeddings_to_lava.py --pkl CLIP_embeddings.pkl --img-dir instance_crops
#  python use_clip_embeddings_to_lava.py --pkl CLIP_embeddings.pkl --candidates candidates.txt
#
# Requirements:
#   pip install torch torchvision transformers pillow sentencepiece tqdm blip-client
# (If you can't install blip-client, the script will still run but won't auto-generate captions.)

import os
import argparse
import pickle
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

# transformers CLIP + BLIP
from transformers import CLIPProcessor, CLIPModel
from transformers import BlipProcessor, BlipForConditionalGeneration

def load_pkl(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def inspect_pkl(data):
    print("=== PKL CONTENTS ===")
    if isinstance(data, dict):
        for k,v in data.items():
            try:
                shape = getattr(v, "shape", None)
                print(f"- {k}: type={type(v)}, shape={shape}")
            except Exception:
                print(f"- {k}: type={type(v)}")
    else:
        print("Top-level object is", type(data))
    print("====================")

def ensure_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.cpu().numpy()
    return np.array(x)

def l2_normalize(x, axis=1, eps=1e-12):
    norms = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norms, eps)

def compute_text_embeddings_clip(texts, clip_model, clip_processor, device):
    # batch-safe
    all_embeds = []
    batch_size = 64
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = clip_processor(text=batch, images=None, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = clip_model.get_text_features(**inputs)
            emb = out.cpu().numpy()
        all_embeds.append(emb)
    return np.vstack(all_embeds)

def compute_image_embeddings_with_clip_from_dir(img_dir, clip_model, clip_processor, device):
    fnames = sorted([fn for fn in os.listdir(img_dir) if fn.lower().endswith((".jpg",".jpeg",".png"))])
    imgs = []
    for fn in fnames:
        im = Image.open(os.path.join(img_dir, fn)).convert("RGB")
        imgs.append(im)
    all_embeds = []
    batch_size = 32
    for i in range(0, len(imgs), batch_size):
        batch_imgs = imgs[i:i+batch_size]
        inputs = clip_processor(images=batch_imgs, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = clip_model.get_image_features(**inputs)
            emb = out.cpu().numpy()
        all_embeds.append(emb)
    return fnames, np.vstack(all_embeds)

def generate_blip_candidates(image_path, blip_processor, blip_model, device, num_beams=5, num_return_sequences=5):
    im = Image.open(image_path).convert("RGB")
    inputs = blip_processor(images=im, return_tensors="pt").to(device)
    out = blip_model.generate(**inputs, num_beams=num_beams, num_return_sequences=num_return_sequences, max_length=32)
    captions = [blip_processor.tokenizer.decode(g, skip_special_tokens=True).strip() for g in out]
    # dedupe order-preserving
    seen = set(); unique=[]
    for c in captions:
        if c not in seen:
            unique.append(c); seen.add(c)
    return unique

def main(args):
    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # load pkl
    data = load_pkl("/home/rizo/Downloads/12-9-scene-graph-aditya/clip_embeddings.pkl")
    inspect_pkl(data)

    # load CLIP model
    clip_model_name = None
    if isinstance(data, dict) and "model_name" in data:
        clip_model_name = data["model_name"]
        print("PKL metadata model_name:", clip_model_name)
    if args.clip_model is not None:
        clip_model_name = args.clip_model
    if clip_model_name is None:
        # clip_model_name = "openai/clip-vit-base-patch32"
        # print("No model info found in pkl. Defaulting to", clip_model_name)
        pass

    print("Loading CLIP model:", clip_model_name)
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

    # Decide what we have in the pkl
    img_embeds = None
    img_fnames = None
    text_embeds = None
    text_candidates = None

    # Heuristics for common keys
    if isinstance(data, dict):
        # common key names -- adapt if your file uses different names
        for key in data:
            kl = key.lower()
            if "image" in kl and ("emb" in kl or "feat" in kl):
                img_embeds = ensure_numpy(data[key])
            if "text" in kl and ("emb" in kl or "feat" in kl):
                text_embeds = ensure_numpy(data[key])
            if "filenames" in kl or "names" in kl or "image_ids" in kl:
                try:
                    img_fnames = list(data[key])
                except Exception:
                    pass
            if ("captions" in kl or "candidates" in kl or "texts" in kl) and text_candidates is None:
                try:
                    text_candidates = list(data[key])
                except Exception:
                    pass

    # If image embeddings are not inside, but user supplied an image directory, compute embeddings
    if img_embeds is None:
        if args.img_dir:
            print("No image embeddings in pkl; computing image embeddings from images in", args.img_dir)
            img_fnames, img_embeds = compute_image_embeddings_with_clip_from_dir(args.img_dir, clip_model, clip_processor, device)
        else:
            raise ValueError("No image embeddings found in .pkl and no --img-dir provided to compute them.")

    img_embeds = ensure_numpy(img_embeds)
    img_embeds = l2_normalize(img_embeds)

    # If user passed candidate captions file, load it
    if args.candidates:
        with open(args.candidates, "r", encoding="utf-8") as f:
            candidates = [l.strip() for l in f if l.strip()]
        text_candidates = candidates

    # If we already have text embeddings in pkl, use them
    if text_embeds is not None:
        text_embeds = ensure_numpy(text_embeds)
        text_embeds = l2_normalize(text_embeds)
        print("Using text embeddings from pkl with shape", text_embeds.shape)
        # but we *also* need corresponding candidate strings to write results; attempt to read if present
        if text_candidates is None:
            if isinstance(data, dict) and "text_strings" in data:
                text_candidates = list(data["text_strings"])
            else:
                print("Warning: text embeddings present but no text strings found in pkl. You may want to provide --candidates.")
    else:
        # Need to build candidate captions (either provided or auto-generate with BLIP)
        if text_candidates is None:
            # try BLIP generation per image (this is slower)
            print("No candidate captions provided; will generate BLIP candidates per image (slower).")
            blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
            per_image_candidates = []
            # Need image files to generate from -- require --img-dir
            if args.img_dir is None:
                raise ValueError("To auto-generate candidate captions with BLIP you must provide --img-dir")
            fnames = sorted([fn for fn in os.listdir(args.img_dir) if fn.lower().endswith((".jpg",".png",".jpeg"))])
            for fn in tqdm(fnames, desc="Generating BLIP candidates"):
                p = os.path.join(args.img_dir, fn)
                cands = generate_blip_candidates(p, blip_processor, blip_model, device,
                                                 num_beams=args.blip_beams, num_return_sequences=args.blip_n)
                per_image_candidates.append((fn, cands))
            # Next, collect unique candidate captions across images
            uniq = []
            for fn, c in per_image_candidates:
                for s in c:
                    if s not in uniq:
                        uniq.append(s)
            text_candidates = uniq
            # Save mapping for final choice per image (we'll rerank per-image using only its candidates)
            per_image_candidates_map = {fn:c for fn,c in per_image_candidates}
            print("Generated", len(text_candidates), "unique candidate captions (across images).")
        else:
            per_image_candidates_map = None

        # compute embeddings for text_candidates
        print("Computing text embeddings for", len(text_candidates), "candidates.")
        text_embeds = compute_text_embeddings_clip(text_candidates, clip_model, clip_processor, device)
        text_embeds = l2_normalize(text_embeds)

    # Now compute similarity and pick top candidate(s)
    print("Computing similarities and selecting top captions...")
    results = []
    if img_fnames is None:
        # try to get filenames from data or create placeholders
        if isinstance(data, dict) and "filenames" in data:
            img_fnames = list(data["filenames"])
        else:
            img_fnames = [f"object_{i+1}.jpg" for i in range(img_embeds.shape[0])]

    # If we have per-image candidate lists (BLIP per image)
    for idx, img_emb in enumerate(img_embeds):
        img_name = img_fnames[idx] if idx < len(img_fnames) else f"object_{idx+1}.jpg"
        if per_image_candidates_map is not None:
            cands = per_image_candidates_map.get(img_name, text_candidates)
            # compute their embeddings (we created text_embeds and text_candidates earlier as global uniq list)
            # create index map
            cand_idxs = [text_candidates.index(c) for c in cands]
            sub_text_embeds = text_embeds[cand_idxs]
            sims = (sub_text_embeds @ img_emb.reshape(-1,1)).squeeze()  # cos since normalized
            best = int(np.argmax(sims))
            chosen = cands[best]
            score = float(sims[best])
        else:
            sims = (text_embeds @ img_emb).squeeze()
            best = int(np.argmax(sims))
            chosen = text_candidates[best]
            score = float(sims[best])

        results.append((img_name, chosen, score))

    # Write LAVA file
    out_file = args.out if args.out else "lava_descriptions.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        for name, caption, score in results:
            # Format caption: capitalized and ensure trailing period
            txt = caption.strip()
            if len(txt) == 0:
                txt = "No description"
            if not txt.endswith("."):
                txt = txt + "."
            txt = txt[0].upper() + txt[1:]
            f.write(f"{name}: {txt}\n")
    print("Wrote", out_file)
    # Also print low-confidence ones
    if args.min_score is not None:
        low = [(n,c,s) for (n,c,s) in results if s < args.min_score]
        if low:
            print("\nLow-confidence results (score < {:.3f}):".format(args.min_score))
            for n,c,s in low:
                print(f"- {n}: {c} (score {s:.4f})")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pkl", required=False, help="Path to CLIP_embeddings.pkl")
    p.add_argument("--img-dir", default=None, help="Directory with instance crops (if needed)")
    p.add_argument("--candidates", default=None, help="Text file with one candidate caption per line (optional)")
    p.add_argument("--clip-model", default=None, help="CLIP model name (hf) to use for text embeddings (optional)")
    p.add_argument("--out", default=None, help="Output lava descriptions file name")
    p.add_argument("--min-score", type=float, default=0.12, help="Threshold to show low-confidence items")
    p.add_argument("--blip-beams", type=int, default=5)
    p.add_argument("--blip-n", type=int, default=5)
    args = p.parse_args()
    main(args)