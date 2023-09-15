import bpy
import os
import yaml

def get_node_tree_inputs():
    obj = bpy.context.active_object

    modifiers=[]
    for modifier in obj.modifiers:
        if modifier.type == "NODES":
            modifiers.append(modifier)
            print(f"Found Geometry Node modifier {modifier.name}")

    modifier:bpy.types.Modifier = modifiers[0]
    param_names = []
    for input in modifier.node_group.inputs:
        # print(f"Input {input.identifier} is named {input.name}")
        param_names.append(
            input.name
        )
    return param_names

def get_model_outputs():
    target_yml = "./datasets/ChairDataset/test/results_exp_geocode_chair/yml_predictions_sketch/chair_back_frame_mid_y_offset_pct_0_0000_0000_-30.0_15.0_pred_sketch.yml"
    cwd = os.getcwd()
    full_path = os.path.join(cwd, target_yml)

    with open(full_path, 'r') as yml:
        try:
            data = yaml.safe_load(yml)
            param_names = []
            for key in data:
            # print("Entry name:", key)
                param_names.append(key)
        except yaml.YAMLError as e:
            print("Error reading YAML file:", e)
    return param_names


input_params = get_node_tree_inputs()
output_params = get_model_outputs()
print(len(input_params))
print(len(output_params))
