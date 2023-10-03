import sys
sys.path.append("/media/sanbingyouyong/Ray/Projects/Research/ProceduralModeling/ASU/GeoCode/geocode/")
# sys.path.append("/home/sanbingyouyong/miniconda3/envs/geocode3.10m/lib/python3.10/site-packages/geocode-0.0.0-py3.10.egg")

import torch
import yaml
from models.vgg import vgg11_bn  # Import your VGG model class
from models.decoder import DecodersNet
from geocode.geocode_util import InputType  # Import the InputType enum from your code

from geocode.geocode_model import Model

from PIL import Image


print("yes")