from torchvision.io.video import read_video
import torch.nn as nn
import torch
## Import video swin transformer
from torchvision.models.video import swin3d_b, Swin3D_B_Weights
from glob import glob
import numpy as np
import os
"""
ref: https://web.stanford.edu/~nanbhas/blog/forward-hooks-pytorch/

"""

## Import videoresnet
# from torchvision.models.video import r3d_18, R3D_18_Weights
# a dict to store the activations

def VideoEncode(video_path):

  activation = {}
  def getActivation(name):
    # the hook signature
    def hook(model, input, output):
      activation[name] = output.detach()
    return hook

  # if save_path == 'None':

  save_path = "/".join(video_path.split('/')[:-1])+'/VST_feature.npy'

  # Check if the file exists
  if os.path.exists(save_path):
      print("The file VST_feature.npy exists in the specified location.")
      return "None"
  else:
      print("The file VST_feature.npy does not exist in the specified location.")

      vid, _, _ = read_video(video_path, output_format="TCHW",pts_unit="sec")

      if vid.shape[0] > 100:
        print("\nThe video has more than 128 frames !!")

        vid = vid[:100]  # optionally shorten duration

      # For vision swin transformer
      device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

      weights = Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1

      model = swin3d_b(weights=weights)
      model = model.to(device)
      model.eval()

      # Step 2: Initialize the inference transforms
      preprocess = weights.transforms()


      # Step 3: Apply inference preprocessing transforms
      batch = preprocess(vid).unsqueeze(0).to(device)

      # [1,3,32,224,224] --> [#of batch, #of channel(rgb), #of frames, height,width ]

      # unsqueeze(0) adds a new dimension here on 1st place which is 1 or #of batch
      print("\n Batch shape we got: ",batch.shape)

      # # print(model)

      print("Layers we have for the model")
      print([n for n, _ in model.named_children()])


      # # h1 = model.norm.register_forward_hook(getActivation('norm'))
      h2 = model.avgpool.register_forward_hook(getActivation('avgpool'))
      # # h3 = model.head.register_forward_hook(getActivation('head'))



      out = model(batch)

      # # print(activation)


      # # print("\n Shape from norm: ",activation['norm'].shape)
      print("\n Shape from avgpool: ",activation['avgpool'].shape)
      # # print("\n Shape from head: ",activation['head'].shape)

      feature_vector_tensor = activation['avgpool'].squeeze(-1).squeeze(-1).squeeze(-1)

      feature_vector_np = feature_vector_tensor.detach().cpu().numpy()

      print('\n')
      print(feature_vector_np)
      print(type(feature_vector_np))
      print(feature_vector_np.shape)

      np.save(save_path,feature_vector_np)

      # # detach the hooks
      # # h1.remove()
      h2.remove()
    # # h3.remove()

      return feature_vector_np

if __name__ == '__main__':

    video_path = glob("/root/DataPrep/new/Action-Recognition/VSTData/*/*/*.avi")
    

    for video in video_path:


        ## Encode Video
        print("...............")
        print("\n\nFor Video: ",video)

        save_path = "/".join(video.split('/')[:-1])+'/VST_feature.npy'

        ### Encode Video
        encoded_featureVector= VideoEncode(video)
        # print(encoded_featureVector.shape)