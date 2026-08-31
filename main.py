from visualize import *
from functions import *


def main() -> None:
  
  path = './images'
  feature_dir = Path("./sample_test_features")
  images_list = list(Path(path).glob("*.png"))
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  #ALIKED 
  detect_keypoints(images_list, feature_dir)
  #VGG16
  all_pairs = image_similarity(images_list, device = device)
  #LightGlue
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
  mapper_options.mapper.abs_pose_max_error = 7.0
  mapper_options.mapper.filter_max_reproj_error = 6.0
  mapper_options.mapper.abs_pose_min_num_inliers = 40
  mapper_options.mapper.init_min_tri_angle = 6.0
  mapper_options.mapper.init_min_num_inliers = 50
  mapper_options.triangulation.ignore_two_view_tracks = False
  
  
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
