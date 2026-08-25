from visualize import *
from functions import *


def main() -> None:
  
  path = '/kaggle/input/competitions/image-matching-challenge-2024/test/church/images'
  feature_dir = Path("./sample_test_features")
  images_list = list(Path(path).glob("*.png"))

  #SIFT
  detect_keypoints(images_list, feature_dir)
  all_pairs = list(itertools.combinations([i for i in range(len(images_list))], 2))
  #BFMatcher
  keypoint_distances(images_list, all_pairs, feature_dir)

  database_path = "colmap.db"
  images_dir = images_list[0].parent
  
  import_into_colmap(
      images_dir, 
      feature_dir, 
      database_path
  )
  # This does RANSAC
  pycolmap.match_exhaustive(database_path)
  
  mapper_options = pycolmap.IncrementalPipelineOptions()
  mapper_options.min_model_size = 8
  mapper_options.max_num_models = 10

  #Incremental SfM
  maps = pycolmap.incremental_mapping(
      database_path=database_path, 
      image_path=images_dir,
      output_path=Path.cwd() / "incremental_pipeline_outputs"
  )
  #Visualizing a result
  visualize(maps[0])

if __name__ == '__main__':
    main()
