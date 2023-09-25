import torch
import yaml
from models.vgg import vgg11_bn  # Import your VGG model class
from models.decoder import DecodersNet
from geocode.geocode_util import InputType  # Import the InputType enum from your code

from geocode.geocode_model import Model

# Load the saved model checkpoint (replace 'path_to_checkpoint' with the actual path)
checkpoint_path = 'path_to_checkpoint/model.ckpt'
model = Model.load_from_checkpoint(checkpoint_path)

# Set the model to evaluation mode
model.eval()

# Load and preprocess your single input sketch image
# Replace 'input_image' with your actual input data (e.g., image loading and preprocessing)
input_image = ...

# Process the input image (sketch) as needed
input_data = preprocess_sketch(input_image)  # Replace with your preprocessing code
input_data = torch.from_numpy(input_data).unsqueeze(0)  # Add batch dimension

# Pass the input data through the VGG model
sketch_emb = model.vgg(input_data)

# Decode and get predictions for the sketch
with torch.no_grad():
    pred_sketch = model.decoders_net.decode(sketch_emb)

# Convert predictions to a map or other suitable format
pred_map_sketch = model.param_descriptors.convert_prediction_vector_to_map(pred_sketch.cpu())

# Save the prediction as a YAML file (replace 'output_path' with the desired path)
output_path = 'output_prediction_sketch.yml'
with open(output_path, 'w') as yaml_file:
    yaml.dump(pred_map_sketch, yaml_file)

# The prediction is now saved in 'output_path' as a YAML file
