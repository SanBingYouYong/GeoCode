from common.bpy_util import select_shape, get_geometric_nodes_modifier
from common.input_param_map import get_input_param_map, load_shape_from_yml
from common.file_util import get_recipe_yml_obj


recipe_file_path = "./datasets/ChairDataset/recipe.yml"
# shape_yml_path = "./blends/inference_out.yml"
shape_yml_path = "./datasets/ChairDataset/test/results_exp_geocode_chair/yml_predictions_sketch/chair_back_frame_mid_y_offset_pct_0_0000_0000_-30.0_55.0_pred_sketch.yml"
# shape_yml_path = "./datasets/ChairDataset/test/results_exp_geocode_chair/yml_gt/chair_back_frame_mid_y_offset_pct_0_0000_0000_gt.yml"
obj = select_shape()
mod = get_geometric_nodes_modifier(obj)
recipe_yml = get_recipe_yml_obj(recipe_file_path)  # this might be where denorm happens
input_params_map = get_input_param_map(mod, recipe_yml)
load_shape_from_yml(shape_yml_path, input_params_map)