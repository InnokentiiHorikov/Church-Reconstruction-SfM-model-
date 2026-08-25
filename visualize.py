import pycolmap 


def visualize(maps: pycolmap.Reconstruction) -> None:

  xyz = []
  rgb = []
  
  for point3D_id, point3D in maps.points3D.items():
      xyz.append(point3D.xyz)
      # PyCOLMAP stores colors as 0-255 integers; Open3D requires 0.0-1.0 floats
      rgb.append(point3D.color / 255.0)
      
  xyz_np = np.array(xyz)
  rgb_np = np.array(rgb)
  
  fig = go.Figure(data=[go.Scatter3d(
      x=xyz_np[:, 0], y=xyz_np[:, 1], z=xyz_np[:, 2],
      mode='markers',
      marker=dict(
          size=2,
          color=['rgb({},{},{})'.format(int(r*255), int(g*255), int(b*255)) for r, g, b in rgb_np],
          opacity=0.8
      )
  )])
  
  fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
  fig.show()
