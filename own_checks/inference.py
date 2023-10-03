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

# Load the saved model checkpoint (replace 'path_to_checkpoint' with the actual path)
checkpoint_path = './models/exp_geocode_chair/last.ckpt'

recipe_yml_obj = get_recipe_yml_obj("./datasets/ChairDataset/recipe.yml")
inputs_to_eval = get_inputs_to_eval(recipe_yml_obj)
param_descriptors = ParamDescriptors(recipe_yml_obj, inputs_to_eval, use_regression=False)
model = Model.load_from_checkpoint( checkpoint_path, batch_size=1,
                                    param_descriptors=param_descriptors, results_dir="./blends/outs",
                                    test_dir="./datasets/temp", models_dir="./models",
                                    test_dataloaders_types=["sketch"], test_input_type="sketch",
                                    exp_name="temp")

# Set the model to evaluation mode
model.eval()

# Load and preprocess your single input sketch image
# Replace 'input_image' with your actual input data (e.g., image loading and preprocessing)
input_image = "./blends/annotation_image.png"

# Process the input image (sketch) as needed
input_data = Image.open(input_image)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
input_data = preprocess(input_data).unsqueeze(0)  # Add batch dimension

# Pass the input data through the VGG model
sketch_emb = model.vgg(input_data)

# Decode and get predictions for the sketch
with torch.no_grad():
    pred_sketch = model.decoders_net.decode(sketch_emb)

# Convert predictions to a map or other suitable format
pred_map_sketch = model.param_descriptors.convert_prediction_vector_to_map(pred_sketch.cpu())

# Save the prediction as a YAML file (replace 'output_path' with the desired path)
output_path = './blends/inference_out.yml'
with open(output_path, 'w') as yaml_file:
    yaml.dump(pred_map_sketch, yaml_file)

# The prediction is now saved in 'output_path' as a YAML file
