from own_checks.inference_methods import gc_single_image_inference_entrypoint
from own_checks.gc_domain import GCDomain
from own_checks.param2obj_methods import param2obj_entrypoint

import bpy


if __name__ == "__main__":
    gc_single_image_inference_entrypoint(GCDomain.VASE)
    param2obj_entrypoint(GCDomain.VASE, bpy.context.active_object)