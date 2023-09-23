import typing
import bpy
import os
import json
import sys
import importlib

from bpy.types import Context

'''
Template for getting started with a Blender interface quickly. 
Bind methods to those `execute` methods. 
Remember to replace Blender's default python directory. 
    e.g. remove built-in python folder, start blender with conda env activated (python path overwrite). 
        or modify default python path manually. 
'''


class SomeFunctionOperator(bpy.types.Operator):
    bl_idname = "object.some_function_operator"
    bl_label = "Some Function"

    def execute(self, context: Context):
        print("You've called some function.")
        return {"FINISHED"}


class SomeOtherFunctionOperator(bpy.types.Operator):
    bl_idname = "object.some_other_function_operator"
    bl_label = "Some Other Function"
    
    def execute(self, context: Context):
        print("You've called some other function.")
        return {'FINISHED'}


class SomeInterfacePanel(bpy.types.Panel):
    bl_label = "Some Interface Panel"
    bl_idname = "SIUI_PT_SomeInterfacePanel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Tool'

    def draw(self, context: Context):
        layout = self.layout
        scene = context.scene
        # add elements to ui
        layout.prop(scene, "some_filepath_property")
        layout.prop(scene, "some_string_property")
        layout.operator("object.some_function_operator")
        layout.operator("object.some_other_function_operator")

def register():
    bpy.types.Scene.some_filepath_property = bpy.props.StringProperty(
        name="Some Filepath Property", 
        subtype='FILE_PATH',
        default="//double/slash/as/relative/path"
    )
    
    bpy.types.Scene.some_string_property = bpy.props.StringProperty(
        name="Some String Property",
        default="some string"
    )
    
    bpy.utils.register_class(SomeFunctionOperator)
    bpy.utils.register_class(SomeOtherFunctionOperator)
    bpy.utils.register_class(SomeInterfacePanel)

def unregister():
    del bpy.types.Scene.some_filepath_property
    del bpy.types.Scene.some_string_property
    
    bpy.utils.unregister_class(SomeFunctionOperator)
    bpy.utils.unregister_class(SomeOtherFunctionOperator)
    bpy.utils.unregister_class(SomeInterfacePanel)

if __name__ == "__main__":
    register()
