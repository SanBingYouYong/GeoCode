import typing
import bpy
import os
import json
import sys
import importlib

from bpy.types import Context


class ClearAllAnnotationsOperator(bpy.types.Operator):
    bl_idname = "object.clear_all_annotations_operator"
    bl_label = "Clear All Annotations"

    def execute(self, context: Context):
        # Get the active scene
        scene = context.scene

        # Clear all annotations
        for annotation_layer in scene.grease_pencil.layers:
            annotation_layer.clear()
        
        return {"FINISHED"}



class AnnotationToModelOperator(bpy.types.Operator):
    bl_idname = "object.annotation_to_model_operator"
    bl_label = "Annotation To Model"

    _ctype_table = {
        "0": "BuildingMass", 
        "1": "Roof", 
        "2": "Window", 
        "3": "Ledge"
    }
    
    def execute(self, context):
        '''
        save annotation image [x]
        cv2 binarize []
        pytorch network inference []
        gs spawn []
        '''
        scene = context.scene
        path = bpy.path.abspath(scene.annotations_img_save_path)
        fullpath = os.path.join(path, scene.annotations_filename)
        json_info_path = self._render_and_save(context, scene, fullpath)
        processed_img_path = self._process_saved_img(context, json_info_path)
        params = self._network_inference(context, scene, processed_img_path)
        # TODO: print params or save params to a dedicated file? for inspection
        obj = self._spawn_object(context, scene, params)  # TODO: what to do with this object reference? the need to store these info? for later clean up? 
        return {'FINISHED'}
    
    def _spawn_object(self, context, scene, params: list) -> bpy.types.Object:
        '''
        Uses gs_dedicated_renderer.DRUtils.spawn to spawn objects with predicted params. 
        Returns the object (reference). 
        '''
        dir = os.path.dirname(bpy.data.filepath)
        if not dir in sys.path:
            sys.path.append(dir)
        from scripts.gs_based import gs_dedicated_renderer
        # forced reload
        importlib.reload(gs_dedicated_renderer)
        img_path = params.pop(0)
        # _, img_name = os.path.split(img_path)
        ctype = params.pop(0)
        subtype = params.pop(0)
        img_name = self._ctype_table[ctype] + "-" + subtype
        comp, obj = gs_dedicated_renderer.DRUtils.spawn(type=ctype, subtype=subtype, params=params, obj_name=img_name)
        return obj

    def _network_inference(self, context, scene, image_path: str):
        '''
        Uses NNInferenceDrive to carry out inference. Classification + Regression
        Returns parameter for spawning. 
        '''
        dir = os.path.dirname(bpy.data.filepath)
        if not dir in sys.path:
            sys.path.append(dir)
        from scripts.network import nn_inference
        # forced reload
        importlib.reload(nn_inference)
        inferencor = nn_inference.NNInferenceDrive()
        # print(f"type info: {type(scene.selected_component_type_index)}")
        type_info = scene.selected_component_type_index
        # TODO: from now on, the process cannot be tested unless training is complete and models are put into proper places
        predicted_class = inferencor.classify(image_path=image_path, typeinfo=type_info)
        # predicted_class = 3
        # TODO: check predicted_class's type: need it to be int
        return inferencor.regress(image_path=image_path, typeinfo=type_info, subtype=int(predicted_class))
    
    def _process_saved_img(self, context, json_path: str):
        '''
        Uses UIExternal to binarize the image. 
        Adds current working dir to sys.path. 
        Returns absolute path to processed image. 
        '''
        dir = os.path.dirname(bpy.data.filepath)
        if not dir in sys.path:
            sys.path.append(dir)
        from scripts.interface import ui_external
        # forced reload
        importlib.reload(ui_external)
        ui = ui_external.UIExternal(json_path)
        ui.binarize_image_opencv()
        return ui.image_path

    def _render_and_save(self, context, scene, fullpath: str) -> str:
        '''
        Renders and saves image, also saves relevant info in json file. 
        Returns path to json file, in absolute path. 
        '''
        context.scene.render.filepath = fullpath
        # find viewport overlay options
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        overlay = space.overlay
                        break
        # render and save image
        overlay.show_axis_x = False
        overlay.show_axis_y = False
        overlay.show_floor = False
        bpy.ops.render.opengl(write_still=True)
        overlay.show_axis_x = True
        overlay.show_axis_y = True
        overlay.show_floor = True
        # save image info file
        info_path = bpy.path.abspath(scene.annotations_info_save_path)
        json_data = {
            "img_path": fullpath, 
            "type_info": scene.selected_component_type_index
        }
        with open(info_path, mode='w', encoding='utf-8') as file:
            json.dump(json_data, file, indent=4)
        return info_path
    
    # def _render_and_save(self):


class ComponentTypeChoicePanel(bpy.types.Panel):
    bl_label = "Component Type Choice Panel"
    bl_idname = "SIUI_PT_ComponentTypeChoicePanel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Tool'

    def draw(self, context):
        layout = self.layout
        scene = bpy.context.scene
        
        layout.label(text="Select a component type:")
        layout.prop(scene, "selected_component_type_index", expand=True)

class AnnotationToModelPanel(bpy.types.Panel):
    bl_label = "Annotation To Model Panel"
    bl_idname = "SIUI_PT_AnnotationToModelPanel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Tool'

    def draw(self, context):
        layout = self.layout
        scene = bpy.context.scene
        # TODO: add models path
        layout.prop(scene, "annotations_info_save_path")
        layout.prop(scene, "annotations_img_save_path")
        layout.prop(scene, "annotations_filename")
        layout.operator("object.annotation_to_model_operator")
        layout.operator("object.clear_all_annotations_operator")

def register():
    bpy.types.Scene.annotations_info_save_path = bpy.props.StringProperty(
        name="Annotations Info Save Path", 
        subtype='FILE_PATH',
        default="//data/addon_out/annotations_info.json"
    )
    
    bpy.types.Scene.annotations_img_save_path = bpy.props.StringProperty(
        name="Image Save Path",
        subtype='FILE_PATH',
        default="//data/addon_out/"
    )
    
    bpy.types.Scene.annotations_filename = bpy.props.StringProperty(
        name="Image Filename",
        default="annotations_image.png"
    )
    
    bpy.types.Scene.selected_component_type_index = bpy.props.EnumProperty(
        items=[("0", "Building Mass", "Building Mass"),
               ("1", "Roof", "Roof"),
               ("2", "Window", "Window"),
               ("3", "Ledge", "Ledge")],
        name="Selected Option",
        default="0"
    )
    
    bpy.utils.register_class(AnnotationToModelOperator)
    bpy.utils.register_class(ComponentTypeChoicePanel)
    bpy.utils.register_class(AnnotationToModelPanel)
    bpy.utils.register_class(ClearAllAnnotationsOperator)

def unregister():
    del bpy.types.Scene.annotations_info_save_path
    del bpy.types.Scene.annotations_img_save_path
    del bpy.types.Scene.annotations_filename
    del bpy.types.Scene.selected_component_type_index
    
    bpy.utils.unregister_class(AnnotationToModelOperator)
    bpy.utils.unregister_class(ComponentTypeChoicePanel)
    bpy.utils.unregister_class(AnnotationToModelPanel)
    bpy.utils.unregister_class(ClearAllAnnotationsOperator)

if __name__ == "__main__":
    register()
