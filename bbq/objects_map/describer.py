import os
import torch
import numpy as np
import open3d as o3d
from PIL import Image
from tqdm import tqdm
from loguru import logger
from bbq.models import LLaVaChat


def get_xyxy_from_mask(mask):
    non_zero_indices = np.nonzero(mask)
    if non_zero_indices[0].size == 0:
        return (0, 0, 0, 0)
    x_min = np.min(non_zero_indices[1])
    y_min = np.min(non_zero_indices[0])
    x_max = np.max(non_zero_indices[1])
    y_max = np.max(non_zero_indices[0])
    return (x_min, y_min, x_max, y_max)


def crop_image(image, mask, padding=30):
    image = np.array(image)
    x1, y1, x2, y2 = get_xyxy_from_mask(mask)

    if image.shape[:2] != mask.shape:
        logger.critical(
            "Shape mismatch: Image shape {} != Mask shape {}".format(image.shape, mask.shape)
        )
        raise RuntimeError

    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image.shape[1], x2 + padding)
    y2 = min(image.shape[0], y2 + padding)

    # Crop and convert back to PIL
    image_crop = Image.fromarray(image[y1:y2, x1:x2])
    return image_crop


def describe_objects(objects, colors, output_dir="output/crops"):
    os.makedirs(output_dir, exist_ok=True)
    captions_file = os.path.join(output_dir, "descriptions.txt")

    chat = LLaVaChat()
    logger.info("LLaVA chat is initialized.")

    query_base = """Describe visible object in front of you, 
    paying close attention to its spatial dimensions and visual attributes."""

    with open(captions_file, "w") as f:
        for idx, object_ in tqdm(enumerate(objects), total=len(objects)):
            bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
                o3d.utility.Vector3dVector(object_['pcd'].points))

            # Prepare image crop
            image = Image.open(colors[object_["color_image_idx"]]).convert("RGB")
            mask = object_["mask"]
            image = image.resize((mask.shape[1], mask.shape[0]), Image.LANCZOS)
            image_crop = crop_image(image, mask)

            # Save crop
            img_name = f"object_{idx+1}.jpg"
            img_path = os.path.join(output_dir, img_name)
            image_crop.save(img_path)

            # Get caption
            image_features = [image_crop]
            image_sizes = [image.size for image in image_features]
            image_features = chat.preprocess_image(image_features)
            image_tensor = [image.to("cuda", dtype=torch.float16) for image in image_features]

            query_tail = """
            The object is one we usually see in indoor scenes. 
            It signature must be short and sparse, describe appearance, geometry, material. Don't describe background.
            Fit your description in four or five words.
            Examples: 
            a closed wooden door with a glass panel;
            a pillow with a floral pattern;
            a wooden table;
            a gray wall.
            """
            query = query_base + "\n" + query_tail
            text = chat(query=query, image_features=image_tensor, image_sizes=image_sizes)
            description = text.replace("<s>", "").replace("</s>", "").strip()

            # Write to file
            f.write(f"{img_name}: {description}\n")

            logger.info(f"Saved {img_name} with description: {description}")
