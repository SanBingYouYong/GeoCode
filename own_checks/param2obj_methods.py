from pathlib import Path
import traceback

import yaml
from common.bpy_util import select_shape, get_geometric_nodes_modifier, refresh_obj_in_viewport
from common.input_param_map import get_input_param_map
from common.file_util import get_recipe_yml_obj

from own_checks.gc_domain import GCDomain

import bpy


def param2obj_entrypoint(domain: GCDomain, obj: bpy.types.Object):
    if domain == GCDomain.CHAIR:
        select_obj(obj)
        param2obj_actual(
            recipe_file_path="./datasets/SingleImgChair/recipe.yml",
            shape_yml_path="./datasets/SingleImgChair/test/results_exp_geocode_chair/yml_predictions_sketch/single_img_-30.0_15.0_pred_sketch.yml",
            obj=obj
        )
    elif domain == GCDomain.TABLE:
        select_obj(obj)
        param2obj_actual(
            recipe_file_path="./datasets/SingleImgTable/recipe.yml",
            shape_yml_path="./datasets/SingleImgTable/test/results_exp_geocode_table/yml_predictions_sketch/single_img_-30.0_15.0_pred_sketch.yml",
            obj=obj
        )
    elif domain == GCDomain.VASE:
        select_obj(obj)
        param2obj_actual(
            recipe_file_path="./datasets/SingleImgVase/recipe.yml",
            shape_yml_path="./datasets/SingleImgVase/test/results_exp_geocode_vase/yml_predictions_sketch/single_img_-30.0_15.0_pred_sketch.yml",
            obj=obj
        )
    else:
        raise Exception(f"Unknown domain [{domain}]")

# taken from common/bpy_util.py
def select_objs(*objs):
    bpy.ops.object.select_all(action='DESELECT')
    for i, obj in enumerate(objs):
        if i == 0:
            bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

# taken from common/bpy_util.py
def select_obj(obj):
    select_objs(obj)

def yml_to_shape(shape_yml_obj, input_params_map, obj, ignore_sanity_check=False):
    try:
        # get the geometric nodes modifier fo the object
        gnodes_mod = get_geometric_nodes_modifier(obj)

        # loops through all the inputs in the geometric node group
        for input in gnodes_mod.node_group.inputs:
            param_name = str(input.name)
            if param_name not in shape_yml_obj:
                continue
            param_val = shape_yml_obj[param_name]
            if hasattr(param_val, '__iter__'):
                # vector handling
                for axis_idx, axis in enumerate(['x', 'y', 'z']):
                    val = param_val[axis]
                    val = round(val, 4)
                    param_name_with_axis = f'{param_name} {axis}'
                    gnodes_mod[input.identifier][axis_idx] = val if abs(val + 1.0) > 0.1 else input_params_map[param_name_with_axis].possible_values[0].item()
                    assert gnodes_mod[input.identifier][axis_idx] >= 0.0
            else:
                param_val = round(param_val, 4)
                if not ignore_sanity_check:
                    err_msg = f'param_name [{param_name}] param_val [{param_val}] possible_values {input_params_map[param_name].possible_values}'
                    assert param_val == -1 or (param_val in input_params_map[param_name].possible_values), err_msg
                gnodes_mod[input.identifier] = param_val if (abs(param_val + 1.0) > 0.1) else (input_params_map[param_name].possible_values[0].item())
                # we assume that all input values are non-negative
                assert gnodes_mod[input.identifier] >= 0.0

        refresh_obj_in_viewport(obj)
    except Exception as e:
        print(repr(e))
        print(traceback.format_exc())


def load_shape_from_yml(yml_file_path, input_params_map, obj, ignore_sanity_check=False):
    with open(yml_file_path, 'r') as f:
        yml_obj = yaml.load(f, Loader=yaml.FullLoader)
    yml_to_shape(yml_obj, input_params_map, obj, ignore_sanity_check=ignore_sanity_check)



def param2obj_actual(recipe_file_path: str, shape_yml_path: str, obj: bpy.types.Object):
    mod = get_geometric_nodes_modifier(obj)
    recipe_yml = get_recipe_yml_obj(recipe_file_path)  # this might be where denorm happens
    input_params_map = get_input_param_map(mod, recipe_yml)
    load_shape_from_yml(shape_yml_path, input_params_map, obj)
    # update the object in the viewport
    obj.data.update()
    # bpy.context.view_layer.update()
