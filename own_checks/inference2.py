import sys
sys.path.append("/media/sanbingyouyong/Ray/Projects/Research/ProceduralModeling/ASU/GeoCode/geocode/")

import torch
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
test_dataloaders = []
test_dataloaders_types = []
test_dataset_sketch = DatasetSketch(inputs_to_eval, param_descriptors_map,
                                    camera_angles_to_process, False,
                                    "./blends/temp_dataset/", "test")
test_dataloader_sketch = DataLoader(test_dataset_sketch, batch_size=batch_size, shuffle=False,
                                    num_workers=1, prefetch_factor=1)
# print(len(test_dataset_sketch))
# raise
test_dataloaders.append(test_dataloader_sketch)
test_dataloaders_types.append('sketch')

pl_model = Model.load_from_checkpoint(  checkpoint_path, batch_size=1,
                                        param_descriptors=param_descriptors, results_dir="/blends/outs",
                                        test_dir="./datasets/temp", models_dir="./models",
                                        test_dataloaders_types=test_dataloaders_types, test_input_type="sketch",
                                        exp_name="temp")

trainer = pl.Trainer(gpus=1)
trainer.test(model=pl_model, dataloaders=test_dataloaders, ckpt_path=checkpoint_path)
