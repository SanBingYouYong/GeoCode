import typing
import bpy
import os
import json
import sys
import importlib

from bpy.types import Context

from PIL import Image

sys.path.append("/media/sanbingyouyong/Ray/Projects/Research/ProceduralModeling/ASU/GeoCode/geocode/")

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

import logging

from own_checks.inference_methods import GCDomain, gc_single_image_inference_entrypoint
from own_checks.param2obj_methods import param2obj_entrypoint
from own_checks.gc_domain import GCDomain


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

        # get domain
        scene = context.scene
        domain = scene.geocode_domain_options
        img_path = None
        obj = None
        if domain == "CHAIR":
            domain = GCDomain.CHAIR
            img_path = "./datasets/SingleImgChair/test/sketches/single_img_-30.0_15.0.png"
            obj = bpy.data.objects["procedural chair"]
        elif domain == "TABLE":
            domain = GCDomain.TABLE
            img_path = "./datasets/SingleImgTable/test/sketches/single_img_-30.0_15.0.png"
            obj = bpy.data.objects["procedural table"]
        elif domain == "VASE":
            domain = GCDomain.VASE
            img_path = "./datasets/SingleImgVase/test/sketches/single_img_-30.0_15.0.png"
            obj = bpy.data.objects["procedural vase"]
        else:
            raise Exception(f"Unknown domain [{domain}]")

        # hide "procedural shape"
        obj.hide_viewport = True
        # hide camera (and its background image)
        bpy.data.objects["Camera"].hide_viewport = True

        bpy.context.scene.render.filepath = img_path
        bpy.ops.render.opengl(write_still=True)
        resize_and_convert(img_path)

        gc_single_image_inference_entrypoint(domain)
        param2obj_entrypoint(domain, obj)

        # bring it back in
        obj.hide_viewport = False
        bpy.data.objects["Camera"].hide_viewport = False
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

class ClearBackgroundImageOperator(bpy.types.Operator):
    bl_idname = "object.clear_background_image_operator"
    bl_label = "Clear Background Image"

    def execute(self, context: Context):
        print("You've called Clear Background Image.")
        context.scene.background_image_path = ""  # Clear the background image path
        update_camera_background_image(context)  # Update the camera background image
        return {"FINISHED"}

class ToggleCameraViewOperator(bpy.types.Operator):
    bl_idname = "object.toggle_camera_view_operator"
    bl_label = "Toggle Camera View"

    def execute(self, context: Context):
        print("You've called Toggle Camera View.")
        bpy.ops.view3d.view_camera()
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

        box = layout.box()
        box.label(text="GeoCode")
        box.prop(scene, "geocode_domain_options", text="GeoCode Domain")
        box.operator("object.capture_annotation_operator")

        box = layout.box()
        box.label(text="View")
        # Toggle to show/hide the current shape
        box.prop(scene, "show_current_shape", text="Show Current Shape")
        box.prop(scene, "slider_value", text="View Angle")
        # switch to camera view
        box.operator("object.toggle_camera_view_operator", text="Toggle Camera View")
        box.operator("object.clear_all_annotation_operator")

        box = layout.box()
        box.label(text="Background Image")
        box.prop(scene, "show_background_image", text="Show Background Image")
        # Background image path input
        box.prop(scene, "background_image_path", text="Background Image")
        # Background image opacity slider
        box.prop(scene, "background_image_opacity", text="Opacity")
        # Clear background image button
        box.operator("object.clear_background_image_operator", text="Clear Background Image")


def update_slider_value(self, context):
    print("slider value updated")
    scene = context.scene
    print(scene.slider_value)
    # update z rotation of "CameraTrack" object accordingly
    bpy.data.objects["CameraTrack"].rotation_euler[2] = (
        scene.slider_value * 3.1415926 / 180.0
    )
    print("updated camera rotation")

def update_geocode_domain_options(self, context):
    selected_domain = context.scene.geocode_domain_options

    # Hide all objects initially
    for obj in bpy.data.objects:
        obj.hide_viewport = True

    # Show the corresponding object based on the selected domain
    object_name = f"procedural {selected_domain.lower()}"
    if object_name in bpy.data.objects:
        bpy.data.objects[object_name].hide_viewport = False
    
    # Still show camera for background image
    bpy.data.objects["Camera"].hide_viewport = False

def update_camera_background_image(context):
    camera = context.space_data.camera
    if camera is not None:
        # Set the background image for the camera
        camera.data.background_images.clear()
        if context.scene.background_image_path != "":
            camera.data.show_background_images = True
            bg_img = camera.data.background_images.new()
            bg_img.image = bpy.data.images.load(context.scene.background_image_path)
            bg_img.alpha = context.scene.background_image_opacity
        else:
            camera.data.show_background_images = False

def update_camera_background_opacity(context):
    camera = context.space_data.camera
    if camera is not None:
        # Update the background image opacity for the camera
        for bg_img in camera.data.background_images:
            bg_img.alpha = context.scene.background_image_opacity


def update_background_image_path(self, context):
    # Update the camera background image when the path is changed
    update_camera_background_image(context)

def update_background_image_opacity(self, context):
    # Update the camera background image opacity when the opacity is changed
    update_camera_background_opacity(context)

def update_show_current_shape(self, context):
    # Update the visibility of the current shape
    scene = context.scene
    selected_domain = scene.geocode_domain_options
    object_name = f"procedural {selected_domain.lower()}"
    if object_name in bpy.data.objects:
        bpy.data.objects[object_name].hide_viewport = not scene.show_current_shape

def update_show_background_image(self, context):
    # Update whether camera background image is shown
    camera = context.space_data.camera
    if camera is not None:
        camera.data.show_background_images = context.scene.show_background_image




def register():
    # bpy.types.Scene.annotation_image_path: bpy.types.StringProperty = bpy.props.StringProperty(
    #     name="Annotation Image Path Property",
    #     subtype='FILE_PATH',
    #     default="//datasets//SingleImg//test//sketches//single_img_-30.0_15.0.png"
    # )

    bpy.types.Scene.slider_value = bpy.props.FloatProperty(
        name="View Angle", default=0.0, min=0.0, max=90.0, update=update_slider_value
    )
    bpy.types.Scene.geocode_domain_options = bpy.props.EnumProperty(
        items=[
            ('CHAIR', 'Chair', 'Select Chair'),
            ('TABLE', 'Table', 'Select Table'),
            ('VASE', 'Vase', 'Select Vase')
            # Add more options as needed
        ],
        name='GeoCode Domain',
        description='Select the GeoCode Domain',
        default='CHAIR',
        update=update_geocode_domain_options
    )
    bpy.types.Scene.background_image_path = bpy.props.StringProperty(
        name="Background Image Path",
        subtype='FILE_PATH',
        default="",
        description="Path to the background image for reference in camera view",
        update=update_background_image_path
    )

    bpy.types.Scene.background_image_opacity = bpy.props.FloatProperty(
        name="Background Image Opacity",
        default=1.0,
        min=0.0,
        max=1.0,
        description="Opacity of the background image in camera view",
        update=update_background_image_opacity
    )
    bpy.types.Scene.show_current_shape = bpy.props.BoolProperty(
        name="Show Current Shape",
        default=True,
        description="Toggle to show/hide the current shape in the viewport",
        update=update_show_current_shape
    )
    bpy.types.Scene.show_background_image = bpy.props.BoolProperty(
        name="Show Background Image",
        default=True,
        description="Toggle to show/hide the background image in the viewport",
        update=update_show_background_image
    )




    bpy.utils.register_class(CaptureAnnotationOperator)
    bpy.utils.register_class(ClearAllAnnotationOperator)
    bpy.utils.register_class(GeoCodeInterfacePanel)
    bpy.utils.register_class(ClearBackgroundImageOperator)
    bpy.utils.register_class(ToggleCameraViewOperator)


def unregister():
    # del bpy.types.Scene.annotation_image_path

    bpy.utils.unregister_class(CaptureAnnotationOperator)
    bpy.utils.unregister_class(ClearAllAnnotationOperator)
    bpy.utils.unregister_class(GeoCodeInterfacePanel)
    bpy.utils.unregister_class(ClearBackgroundImageOperator)
    bpy.utils.unregister_class(ToggleCameraViewOperator)


if __name__ == "__main__":
    register()
