import typing
import bpy
import os
import json
import sys
import importlib

from bpy.types import Context

# from own_checks.inference.inference3 import gc_single_image_inference

# from pathlib import Path
# def import_parents(level=1):
#     global __package__
#     file = Path(__file__).resolve()
#     parent, top = file.parent, file.parents[level]
#     sys.path.append(str(top))
#     try:
#         sys.path.remove(str(parent))
#     except ValueError:
#         pass
#     __package__ = '.'.join(parent.parts[len(top.parts):])
#     importlib.import_module(__package__)
# import_parents(level=1)

from PIL import Image

from common.bpy_util import select_shape, get_geometric_nodes_modifier
from common.input_param_map import get_input_param_map, load_shape_from_yml
from common.file_util import get_recipe_yml_obj

import torch
import shutil
from pathlib import Path
import pytorch_lightning as pl
from subprocess import Popen, PIPE
from data.dataset_sketch import DatasetSketch
from geocode.barplot_util import gen_and_save_barplot
from common.param_descriptors import ParamDescriptors
from geocode.geocode_util import (
    InputType,
    get_inputs_to_eval,
    calc_prediction_vector_size,
)
from geocode.geocode_model import Model
from torch.utils.data import DataLoader
from common.file_util import get_recipe_yml_obj


# from inference3
SINGLE_IMAGE_DATASET_DIR = "./datasets/SingleImg"
PHASE = "test"
EXP_NAME = "exp_geocode_chair"
MODELS_DIR = "./models"
BLENDER_DIR = os.path.expandvars("$BLENDER32")
BLEND_FILE = "./blends/procedural_chair.blend"


def gc_single_image_inference(
    single_img_dataset_dir=SINGLE_IMAGE_DATASET_DIR,
    phase=PHASE,
    exp_name=EXP_NAME,
    models_dir=MODELS_DIR,
):
    sys.path.append(
        "/media/sanbingyouyong/Ray/Projects/Research/ProceduralModeling/ASU/GeoCode/geocode/"
    )
    recipe_file_path = Path(single_img_dataset_dir, "recipe.yml")
    if not recipe_file_path.is_file():
        raise Exception(f"No 'recipe.yml' file found in path [{recipe_file_path}]")
    recipe_yml_obj = get_recipe_yml_obj(str(recipe_file_path))

    inputs_to_eval = get_inputs_to_eval(recipe_yml_obj)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    camera_angles_to_process = (
        recipe_yml_obj["camera_angles_train"] + recipe_yml_obj["camera_angles_test"]
    )
    camera_angles_to_process = [f"{a}_{b}" for a, b in camera_angles_to_process]

    param_descriptors = ParamDescriptors(
        recipe_yml_obj, inputs_to_eval, use_regression=False
    )
    param_descriptors_map = param_descriptors.get_param_descriptors_map()
    detailed_vec_size = calc_prediction_vector_size(param_descriptors_map)
    print(f"Prediction vector length is set to [{sum(detailed_vec_size)}]")

    # setup required dirs
    required_dirs = [
        "barplot",
        "yml_gt",
        "yml_predictions_pc",
        "yml_predictions_sketch",
        "obj_gt",
        "obj_predictions_pc",
        "obj_predictions_sketch",
        "render_gt",
        "render_predictions_pc",
        "render_predictions_sketch",
        "sketch_gt",
    ]
    test_dir = Path(single_img_dataset_dir, phase)
    test_dir_obj_gt = test_dir.joinpath("obj_gt")
    results_dir = test_dir.joinpath(f"results_{exp_name}")
    results_dir.mkdir(exist_ok=True)
    for dir in required_dirs:
        results_dir.joinpath(dir).mkdir(exist_ok=True)

    # save the recipe to the results directory
    shutil.copy(recipe_file_path, results_dir.joinpath("recipe.yml"))

    # find the best checkpoint (the one with the highest epoch number out of the saved checkpoints)
    exp_dir = Path(models_dir, exp_name)
    best_model_and_highest_epoch = None
    highest_epoch = 0
    for ckpt_file in exp_dir.glob("*.ckpt"):
        file_name = ckpt_file.name
        if "epoch" not in file_name:
            continue
        epoch_start_idx = file_name.find("epoch") + len("epoch")
        epoch = int(file_name[epoch_start_idx : epoch_start_idx + 3])
        if epoch > highest_epoch:
            best_model_and_highest_epoch = ckpt_file
            highest_epoch = epoch
    print(f"Best model with highest epoch is [{best_model_and_highest_epoch}]")

    batch_size = 1
    test_dataloaders = []
    test_dataloaders_types = []
    test_dataset_sketch = DatasetSketch(
        inputs_to_eval,
        param_descriptors_map,
        camera_angles_to_process,
        False,
        single_img_dataset_dir,
        phase,
    )
    test_dataloader_sketch = DataLoader(
        test_dataset_sketch,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        prefetch_factor=2,
    )
    test_dataloaders.append(test_dataloader_sketch)
    test_dataloaders_types.append("sketch")

    pl_model = Model.load_from_checkpoint(
        str(best_model_and_highest_epoch),
        batch_size=1,
        param_descriptors=param_descriptors,
        results_dir=results_dir,
        test_dir=test_dir,
        models_dir=models_dir,
        test_dataloaders_types=test_dataloaders_types,
        test_input_type=[InputType.sketch],
        exp_name=exp_name,
    )

    trainer = pl.Trainer(gpus=1)
    trainer.test(
        model=pl_model,
        dataloaders=test_dataloaders,
        ckpt_path=best_model_and_highest_epoch,
    )

    # report average inference time
    avg_inference_time = pl_model.inference_time / pl_model.num_inferred_samples
    print(
        f"Average inference time for [{pl_model.num_inferred_samples}] samples is [{avg_inference_time:.3f}]"
    )

    # save the validation and test bar-plots as image
    barplot_target_dir = results_dir.joinpath("barplot")
    for barplot_type in ["val", "test"]:
        barplot_json_path = Path(
            models_dir, exp_name, f"{barplot_type}_barplot_top_1.json"
        )
        if not barplot_json_path.is_file():
            print(f"Could not find barplot [{barplot_json_path}] skipping copy")
            continue
        barplot_target_image_path = barplot_target_dir.joinpath(
            f"{barplot_type}_barplot.png"
        )
        title = "Validation Accuracy" if barplot_type == "val" else "Test Accuracy"
        gen_and_save_barplot(
            barplot_json_path,
            title,
            barplot_target_image_path=barplot_target_image_path,
        )
        shutil.copy(
            barplot_json_path, barplot_target_dir.joinpath(barplot_json_path.name)
        )


def resize_and_convert(img_path: str) -> None:
    # Open an image
    image = Image.open(img_path)
    # Resize the image (e.g., to 300x300 pixels)
    resized_image = image.resize((224, 224))
    # convert to grayscale and save as RGB
    binarized_image = resized_image.convert("L")
    # invert the image
    binarized_image = Image.eval(binarized_image, lambda x: 255 - x)
    # convert back to RGB
    binarized_image = binarized_image.convert("RGB")
    binarized_image.save(img_path)
    print(f"PIL saved img to {img_path}")


class CaptureAnnotationOperator(bpy.types.Operator):
    bl_idname = "object.capture_annotation_operator"
    bl_label = "Capture Annotation & GeoCode"

    def execute(self, context: Context):
        print("You've called Capture Annotation & GeoCode Inference.")

        # hide "procedural shape"
        bpy.data.objects["procedural shape"].hide_viewport = True

        scene: bpy.types.Scene = context.scene
        img_path = "./datasets/SingleImg/test/sketches/single_img_-30.0_15.0.png"
        bpy.context.scene.render.filepath = img_path
        bpy.ops.render.opengl(write_still=True)
        resize_and_convert(img_path)

        gc_single_image_inference()
        recipe_file_path = "./datasets/SingleImg/recipe.yml"
        shape_yml_path = "./datasets/SingleImg/test/results_exp_geocode_chair/yml_predictions_sketch/single_img_-30.0_15.0_pred_sketch.yml"
        obj = select_shape()
        mod = get_geometric_nodes_modifier(obj)
        recipe_yml = get_recipe_yml_obj(
            recipe_file_path
        )  # this might be where denorm happens
        input_params_map = get_input_param_map(mod, recipe_yml)
        load_shape_from_yml(shape_yml_path, input_params_map)

        # bring it back in
        bpy.data.objects["procedural shape"].hide_viewport = False
        return {"FINISHED"}


class ClearAllAnnotationOperator(bpy.types.Operator):
    bl_idname = "object.clear_all_annotation_operator"
    bl_label = "Clear All Annotation"

    def execute(self, context: Context):
        print("You've called Clear All Annotation.")
        scene = context.scene
        for annotation_layer in scene.grease_pencil.layers:
            annotation_layer.clear()
        return {"FINISHED"}


class GeoCodeInterfacePanel(bpy.types.Panel):
    bl_label = "GeoCode Interface Panel"
    bl_idname = "SIUI_PT_GeoCodeInterfacePanel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Tool"

    def draw(self, context: Context):
        layout = self.layout
        scene = context.scene
        # add elements to ui
        # layout.prop(scene, "annotation_image_path")
        layout.prop(scene, "slider_value", text="View Angle")
        layout.operator("object.capture_annotation_operator")
        layout.operator("object.clear_all_annotation_operator")


def update_slider_value(self, context):
    print("slider value updated")
    scene = context.scene
    print(scene.slider_value)
    # update z rotation of "CameraTrack" object accordingly
    bpy.data.objects["CameraTrack"].rotation_euler[2] = (
        scene.slider_value * 3.1415926 / 180.0
    )
    print("updated camera rotation")


def register():
    # bpy.types.Scene.annotation_image_path: bpy.types.StringProperty = bpy.props.StringProperty(
    #     name="Annotation Image Path Property",
    #     subtype='FILE_PATH',
    #     default="//datasets//SingleImg//test//sketches//single_img_-30.0_15.0.png"
    # )

    bpy.types.Scene.slider_value = bpy.props.FloatProperty(
        name="View Angle", default=0.0, min=0.0, max=90.0, update=update_slider_value
    )

    bpy.utils.register_class(CaptureAnnotationOperator)
    bpy.utils.register_class(ClearAllAnnotationOperator)
    bpy.utils.register_class(GeoCodeInterfacePanel)


def unregister():
    # del bpy.types.Scene.annotation_image_path

    bpy.utils.unregister_class(CaptureAnnotationOperator)
    bpy.utils.unregister_class(ClearAllAnnotationOperator)
    bpy.utils.unregister_class(GeoCodeInterfacePanel)


if __name__ == "__main__":
    register()
