from pathlib import Path
from common.bpy_util import select_shape, get_geometric_nodes_modifier
from common.input_param_map import get_input_param_map, load_shape_from_yml
from common.file_util import get_recipe_yml_obj


# recipe_file_path = "./datasets/ChairLess/recipe.yml"
# recipe_file_path = "./datasets/SingleImg/recipe.yml"
# recipe_file_path = "./datasets/SingleImgVase/recipe.yml"
recipe_file_path = "./datasets/SingleImgTable/recipe.yml"
# shape_yml_path = "./blends/inference_out.yml"
# shape_yml_path = "./datasets/ChairDataset/test/results_exp_geocode_chair/yml_predictions_sketch/chair_back_frame_mid_y_offset_pct_0_0000_0000_-30.0_55.0_pred_sketch.yml"
# shape_yml_path = "./datasets/ChairDataset/test/results_exp_geocode_chair/yml_gt/chair_back_frame_mid_y_offset_pct_0_0000_0000_gt.yml"
# shape_yml_path = "./blends/outs/yml_predictions_sketch/annotation_image_out.png_-30.0_35.0_pred_sketch.yml"
# shape_yml_path = "./datasets/ChairLess/test/results_exp_geocode_chair/yml_predictions_sketch/chair_back_frame_mid_y_offset_pct_0_0000_0000_-30.0_15.0_pred_sketch.yml"
# shape_yml_path = "./datasets/SingleImg/test/results_exp_geocode_chair/yml_predictions_sketch/single_img_-30.0_15.0_pred_sketch.yml"
#shape_yml_path = "./datasets/SingleImg/test/results_exp_geocode_chair/yml_gt/single_img_gt.yml"

# vase
#shape_yml_path = "./datasets/SingleImgVase/test/yml_gt/single_img.yml"
shape_yml_path = "./datasets/SingleImgVase/test/results_exp_geocode_vase/yml_predictions_sketch/single_img_-30.0_15.0_pred_sketch.yml"

# table
shape_yml_path = "./datasets/SingleImgTable/test/results_exp_geocode_table/yml_predictions_sketch/single_img_-30.0_15.0_pred_sketch.yml"


# check if ymls exists
recipe_yml_yes = Path(recipe_file_path).is_file()
print(f"recipe yml exists? {recipe_yml_yes}")
shape_yml_yes = Path(shape_yml_path).is_file()
print(f"shape yml exists? {shape_yml_yes}")


obj = select_shape()
# print(f"Selected shape: {obj.name}")
mod = get_geometric_nodes_modifier(obj)
recipe_yml = get_recipe_yml_obj(recipe_file_path)  # this might be where denorm happens
input_params_map = get_input_param_map(mod, recipe_yml)
# print(f"Input params map: {input_params_map}")
load_shape_from_yml(shape_yml_path, input_params_map)