import numpy as np
import cv2 
import os
import itertools
import tqdm
from pathlib import Path
import h5py
import pycolmap


from lightglue import ALIKED, LightGlue
from lightglue.utils import load_image, rbd

import torch
import torchvision
from torchvision.models import vgg16
from torch.nn.functional import cosine_similarity
from torchvision.transforms import v2


from database import *
from h5py_file import *



def embed_image(data_transform: v2.Compose,
                model: torchvision.models,
                path: Path | str):
    "Output: Torch.Tensor[1, 4096]"
    
    image = load_image(path)
    inputs = data_transforms(image).to(device).unsqueeze(dim = 0)
    output = model(inputs)
    
    return output



def image_similarity(
    paths: list[Path] | list[str],
    lower_border: float = 0.3,
    upper_border: float = 0.99,
    device: torch.device = 'cpu') -> list[tuple[int, int]]:
    
    data_transforms = v2.Compose([
     v2.ToDtype(torch.float64, scale=True),    
     v2.Resize((224, 224)),
     v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
     ])
    
    model = vgg16(weights='DEFAULT').to(device)
    model.classifier = model.classifier[:-1]
    
    list_of_pairs = []
    temp_i = 1
    n_images = len(paths)
    image_1 = embed_image(data_transforms, model, paths[0])

    with torch.inference_mode():

        for i in tqdm(range(n_images-1), desc="Checking image similarities"):
            if temp_i != i:
                temp_i = i
                image_1 = embed_image(data_transforms, model, paths[temp_i])
           
            for j in range(i+1, n_images):
            
                image_2 = embed_image(data_transforms, model, paths[j])
                CosSim = cosine_similarity(image_1, image_2)
                if (CosSim < upper_border) and (CosSim > lower_border): 
                    list_of_pairs.append((i, j))

    return list_of_pairs


def detect_keypoints(
    paths: list[Path],
    feature_dir: Path,
    max_num_keypoints: int = 4096, 
    detection_threshold: float = 0.01,
    device: torch.device = 'cpu'
    
) -> None:
    #dtype = torch.float32 # ALIKED has issues with float16
    
    extractor = ALIKED(max_num_keypoints=max_num_keypoints, 
                           detection_threshold=detection_threshold).eval().to(device)
    
    feature_dir.mkdir(parents=True, exist_ok=True)

    
    
    with h5py.File(feature_dir / "keypoints.h5", mode="w") as f_keypoints, \
         h5py.File(feature_dir / "descriptors.h5", mode="w") as f_descriptors:
        masters_dict = {}
        
        for path in tqdm(paths, desc="Computing keypoints"):
            key = path.name
            
            image = load_image(path).to(device)
            
            feats = extractor.extract(image)
            kps, ds = feats['keypoints'].squeeze(), feats['descriptors'].squeeze()

            kps, ds = kps.detach().cpu().numpy(), ds.detach().cpu().numpy()
            
            f_keypoints[key] = kps
            f_descriptors[key] = ds
            
            masters_dict[key] = feats
            
        torch.save(masters_dict, 'features.pt')
        
        
             
def keypoint_distances(
    paths: list[Path],
    index_pairs: list[tuple[int, int]],
    feature_dir: Path,
    min_matches: int = 15,
    verbose: bool = False,
    device: torch.device = 'cpu',
    n_layers: int = 9,
    filter_threshold: float = 0.01,
    depth_confidence: float = 0.95,
    width_confidence: float = 0.95
) -> None:

    
    matcher = LightGlue(features='aliked', 
                        n_layers = n_layers,
                        filter_threshold = filter_threshold,
                        depth_confidence = depth_confidence,
                        width_confidence = width_confidence).eval().to(device)

    features = torch.load('features.pt')
    
    with h5py.File(feature_dir / "matches.h5", mode="w") as f_matches:
            for idx1, idx2 in tqdm(index_pairs, desc="Computing keypoing distances"):
                
                key1, key2 = paths[idx1].name, paths[idx2].name

                feats0, feats1  = features[key1], features[key2]
                matches = matcher({'image0': feats0, 
                                   'image1': feats1})

                
                feats0, feats1, matches = [rbd(x) for x in 
                                             [feats0, feats1, matches]]
                matches = matches['matches'].detach().cpu().numpy()
                # We have matches to consider
                n_matches = np.shape(matches)[0]
                if n_matches:
                    if verbose:
                        print(f"{key1}-{key2}: {n_matches} matches")
                    # Store the matches in the group of one image
                    if n_matches >= min_matches:
                        group  = f_matches.require_group(key1)
                        group.create_dataset(key2, data=matches)
                        


def import_into_colmap(
    path: Path,
    feature_dir: Path,
    database_path: str = "colmap.db",
) -> None:
    """Adds keypoints into colmap"""
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    single_camera = False
    fname_to_id = add_keypoints(db, feature_dir, path, "", "simple-pinhole", single_camera)
    add_matches(
        db,
        feature_dir,
        fname_to_id,
    )
    db.commit()
