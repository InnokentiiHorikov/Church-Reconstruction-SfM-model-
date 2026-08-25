import numpy as np
import cv2 
import os
import itertools
import tqdm
from pathlib import Path
import h5py
import pycolmap


from database import *
from h5py_file import *


def detect_keypoints(
    paths: list[Path],
    feature_dir: Path,
    num_features: int = 4096,
    resize_to: int = 1024,
    #device: torch.device = torch.device("cpu"),
) -> None:
    """Detects the keypoints in a list of images with ALIKED
    
    Stores them in feature_dir/keypoints.h5 and feature_dir/descriptors.h5
    to be used later with LightGlue
    """
    #dtype = torch.float32 # ALIKED has issues with float16
    
    extractor = cv2.SIFT_create()
    
    feature_dir.mkdir(parents=True, exist_ok=True)

    
    with h5py.File(feature_dir / "keypoints.h5", mode="w") as f_keypoints, \
         h5py.File(feature_dir / "descriptors.h5", mode="w") as f_descriptors:
        
        for path in tqdm(paths, desc="Computing keypoints"):
            key = path.name
            
            image = cv2.imread(path)
            
            kp, ds = extractor.detectAndCompute(image, None)
            '''
            kp = [{
                        'pt': p.pt,
                        'size': p.size,
                        'angle': p.angle,
                        'response': p.response,
                        'octave': p.octave,
                        'class_id': p.class_id
                        } for p in kp]
            '''
            kp = np.array([kps.pt for kps in kp])
            f_keypoints[key] = kp
            f_descriptors[key] = ds

    
def keypoint_distances(
    paths: list[Path],
    index_pairs: list[tuple[int, int]],
    feature_dir: Path,
    min_matches: int = 15,
    verbose: bool = True,
    #device: torch.device = torch.device("cpu"),
) -> None:

    matcher = cv2.BFMatcher()
    
    with h5py.File(feature_dir / "keypoints.h5", mode="r") as f_keypoints, \
         h5py.File(feature_dir / "descriptors.h5", mode="r") as f_descriptors, \
         h5py.File(feature_dir / "matches.h5", mode="w") as f_matches:
        
            for idx1, idx2 in tqdm(index_pairs, desc="Computing keypoing distances"):
                key1, key2 = paths[idx1].name, paths[idx2].name

                matches = matcher.knnMatch(f_descriptors[key1][...], 
                                   f_descriptors[key2][...], k=2)
    
                queryIdx, trainIdx = [], []
            
                for m,n in matches:
                    if m.distance < 0.85*n.distance:
                        queryIdx.append(m.queryIdx), trainIdx.append(m.trainIdx)
                    
                indices = np.column_stack((queryIdx, trainIdx))
                # We have matches to consider
                n_matches = len(trainIdx)
                
                if n_matches:
                    if verbose:
                        print(f"{key1}-{key2}: {n_matches} matches")
                    # Store the matches in the group of one image
                    if n_matches >= min_matches:
                        group  = f_matches.require_group(key1)
                        group.create_dataset(key2, data=indices)


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
