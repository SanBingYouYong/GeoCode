from pathlib import Path
from common.bpy_util import select_shape, get_geometric_nodes_modifier
from common.input_param_map import get_input_param_map, load_shape_from_yml
from common.file_util import get_recipe_yml_obj

from own_checks.gc_domain import GCDomain


def param2obj_entrypoint(domain: GCDomain):
    if domain == GCDomain.CHAIR:
        param2obj_actual(
            recipe_file_path="./datasets/SingleImgChair/recipe.yml",
            shape_yml_path="./datasets/SingleImgChair/test/results_exp_geocode_chair/yml_predictions_sketch/single_img_-30.0_15.0_pred_sketch.yml"
        )
    elif domain == GCDomain.TABLE:
        param2obj_actual(
            recipe_file_path="./datasets/SingleImgTable/recipe.yml",
            shape_yml_path="./datasets/SingleImgTable/test/results_exp_geocode_table/yml_predictions_sketch/single_img_-30.0_15.0_pred_sketch.yml"
        )
    elif domain == GCDomain.VASE:
        param2obj_actual(
            recipe_file_path="./datasets/SingleImgVase/recipe.yml",
            shape_yml_path="./datasets/SingleImgVase/test/results_exp_geocode_vase/yml_predictions_sketch/single_img_-30.0_15.0_pred_sketch.yml"
        )
    else:
        raise Exception(f"Unknown domain [{domain}]")


def param2obj_actual(recipe_file_path: str, shape_yml_path: str):
    obj = select_shape()
    mod = get_geometric_nodes_modifier(obj)
    recipe_yml = get_recipe_yml_obj(recipe_file_path)  # this might be where denorm happens
    input_params_map = get_input_param_map(mod, recipe_yml)
    load_shape_from_yml(shape_yml_path, input_params_map)
    # update the object in the viewport
    obj.data.update()
