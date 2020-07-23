import bpy
from math import *
from mathutils import *
import os
import random


pts = "~/home/smandal/" #base location 
inp = pts+'pix3d/'
#all custom texture file should be in folder called texture
for r, d, f in os.walk(inp):
    for file in f:

        if '.obj' in file:
            texture = ['blue.png','red.png','brown.png','wood_solid_03-l-color.jpg','metal_paintedl-color.png',
            'metal_swirlendcap-l-color.jpg','metal-painted-l-color.png','pink.png',
            'metal-painted-l-color.png','mockaroon-YqUeLG7fMr4-unsplash.jpg','nicole-wilcox-ExCeuFAH5ak-unsplash.jpg',
            'wood solid 14-l-color.png','wood_planks_painted_01-m-color.jpg','wood_solid_03-l-color.jpg',
            'black.png','green.png','grey.png','f1.jpg','f2.jpg','f3.jpg']
            bpy.ops.object.delete()
            # print(r,d,file)
            towrite = r+'/'+file[:-4]+'-synthetic_img' #maybe changed output location
            file_loc = r+'/'+file
            print(towrite,file_loc)
            imported_object = bpy.ops.import_scene.obj(filepath=file_loc)
            obj_object = bpy.context.selected_objects[0]

            target = obj_object#bpy.data.objects['Cube']
            mat = bpy.data.materials.new(name="new")
            mat.use_nodes = True
            cam = bpy.data.objects['Camera']
            t_loc_x = target.location.x
            t_loc_y = target.location.y
            cam_loc_x = cam.location.x
            cam_loc_y = cam.location.y

            #dist = sqrt((t_loc_x-cam_loc_x)**2+(t_loc_y-cam_loc_y)**2)
            dist = (target.location.xy-cam.location.xy).length
            #ugly fix to get the initial angle right
            init_angle  = (1-2*bool((cam_loc_y-t_loc_y)<0))*acos((cam_loc_x-t_loc_x)/dist)-2*pi*bool((cam_loc_y-t_loc_y)<0)

            num_steps = 55 #how many rotation steps
            for x in range(num_steps):
                if x<34: continue
                rr = random.randint(0,len(texture)-1)
                bsdf = mat.node_tree.nodes["Principled BSDF"]
                texImage = mat.node_tree.nodes.new('ShaderNodeTexImage')
                texImage.image = bpy.data.images.load(pts+"texture/"+texture[rr])
                mat.node_tree.links.new(bsdf.inputs['Base Color'], texImage.outputs['Color'])

                # Assign it to object
                if target.data.materials:
                    target.data.materials[0] = mat
                else:
                    target.data.materials.append(mat)

                alpha = init_angle + (x+1)*2*pi/num_steps
                cam.rotation_euler[2] = pi/2+alpha
                cam.location.x = t_loc_x+cos(alpha)*dist
                cam.location.y = t_loc_y+sin(alpha)*dist
                file = os.path.join(towrite, str(x))
                bpy.context.scene.render.image_settings.color_mode ='RGBA'
                bpy.context.scene.render.image_settings.file_format = 'PNG'
                bpy.context.scene.render.filepath = file
                bpy.ops.view3d.camera_to_view_selected()
                bpy.ops.render.render( write_still=True )
#            break
#            for o in bpy.context.scene.objects:
#                if o.type == 'MESH':
#                    o.select = True
#                else:
#                    o.select = False
#            bpy.ops.object.select_all(action='DESELECT')
            bpy.ops.object.delete()
#            bpy.context.selected_objects.remove(obj_object)
#            for scene in bpy.data.scenes:
#                for obj in scene.objects:
#                    scene.objects.unlink(obj)
#            obj_object.delete()
#            imported_object.delete()
#            exit(0)


