from visualize import *
from functions import *


def main() -> None:
  
  path = './images'
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
  
  #Incremental SfM
  mapper_options = pycolmap.IncrementalPipelineOptions()
  mapper_options.min_model_size = 2
  mapper_options.max_num_models = 3
  mapper_options.mapper.abs_pose_max_error = 4.0
  mapper_options.mapper.filter_max_reproj_error = 4.0
  mapper_options.mapper.abs_pose_min_num_inliers = 50
  mapper_options.mapper.init_min_tri_angle = 3.0
  mapper_options.mapper.init_min_num_inliers = 100
  
  
  maps = pycolmap.incremental_mapping(
      database_path=database_path, 
      image_path=images_dir,
      output_path=Path.cwd() / "incremental_pipeline_outputs",
      options = mapper_options
  )
  #Visualizing a result
  visualize(maps[0])

if __name__ == '__main__':
    main()
