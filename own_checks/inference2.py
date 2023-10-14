import sys
sys.path.append("/media/sanbingyouyong/Ray/Projects/Research/ProceduralModeling/ASU/GeoCode/geocode/")

import torch
from torch.utils.data import Dataset
import yaml
from models.vgg import vgg11_bn  # Import your VGG model class
from models.decoder import DecodersNet
from geocode.geocode_util import InputType  # Import the InputType enum from your code

from geocode.geocode_model import Model

from common.file_util import load_obj, get_recipe_yml_obj
from geocode.geocode_util import InputType, get_inputs_to_eval, calc_prediction_vector_size
from common.param_descriptors import ParamDescriptors

from PIL import Image
from torchvision.transforms import transforms

from data.dataset_sketch import DatasetSketch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from pathlib import Path



def resize_and_convert(img_path: str) -> None:
    # Open an image
    image = Image.open(img_path)
    # Resize the image (e.g., to 300x300 pixels)
    resized_image = image.resize((224, 224))
    # Convert the image to grayscale
    grayscale_image = resized_image.convert('L')
    # Save the grayscale image
    grayscale_image.save(img_path)
    print(f"PIL saved img to {img_path}")

class SingleImageDataset(Dataset):
    def __init__(self, file_name, sketch_camera_angle, sketch, targets, shape):
        self.file_name = file_name
        self.sketch_camera_angle = sketch_camera_angle
        self.sketch = sketch
        self.targets = targets
        self.shape = shape

    def __len__(self):
        return 1  # We have only one image.

    def __getitem__(self, idx):
        return self.file_name, self.sketch_camera_angle, self.sketch, self.targets, self.shape


# Load the saved model checkpoint (replace 'path_to_checkpoint' with the actual path)
checkpoint_path = './models/exp_geocode_chair/last.ckpt'

recipe_yml_obj = get_recipe_yml_obj("./datasets/ChairDataset/recipe.yml")
inputs_to_eval = get_inputs_to_eval(recipe_yml_obj)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
camera_angles_to_process = recipe_yml_obj['camera_angles_train'] + recipe_yml_obj['camera_angles_test']
camera_angles_to_process = [f'{a}_{b}' for a, b in camera_angles_to_process]

param_descriptors = ParamDescriptors(recipe_yml_obj, inputs_to_eval, use_regression=False)
param_descriptors_map = param_descriptors.get_param_descriptors_map()
detailed_vec_size = calc_prediction_vector_size(param_descriptors_map)
print(f"Prediction vector length is set to [{sum(detailed_vec_size)}]")


batch_size = 1
# test_dataloaders = []
# test_dataloaders_types = []
# test_dataset_sketch = DatasetSketch(inputs_to_eval, param_descriptors_map,
#                                     camera_angles_to_process, False,
#                                     "./blends/temp_dataset/", "test")
# test_dataloader_sketch = DataLoader(test_dataset_sketch, batch_size=batch_size, shuffle=False,
#                                     num_workers=1, prefetch_factor=1)
# test_dataloaders.append(test_dataloader_sketch)
# test_dataloaders_types.append('sketch')

input_image = "./blends/annotation_image.png"
resize_and_convert(input_image)

# Process the input image (sketch) as needed
input_data = Image.open(input_image)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])
# create place holder / empty tensors
fake_targets = torch.zeros((1, sum(detailed_vec_size)))
fake_shape = torch.zeros((1, 1))
# input_data = preprocess(input_data).unsqueeze(0)  # Add batch dimension
input_data = preprocess(input_data)
single_image_dataset = SingleImageDataset("annotation_image_out.png", camera_angles_to_process[0], input_data, fake_targets, fake_shape)
dataloader = DataLoader(single_image_dataset, batch_size=batch_size, shuffle=False)


pl_model = Model.load_from_checkpoint(  checkpoint_path, batch_size=1,
                                        param_descriptors=param_descriptors, results_dir=Path("./blends/outs"),
                                        test_dir=Path("./datasets/temp"), models_dir=Path("./models"),
                                        test_dataloaders_types=["sketch"], test_input_type=[InputType.sketch],
                                        exp_name="temp")
# pl_model.forward()
trainer = pl.Trainer(gpus=1)
trainer.test(model=pl_model, dataloaders=dataloader, ckpt_path=checkpoint_path)
# trainer.predict(model=pl_model, dataloaders=dataloader, ckpt_path=checkpoint_path)
