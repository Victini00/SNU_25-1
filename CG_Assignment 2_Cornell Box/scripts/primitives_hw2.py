import pyglet
from pyglet import window, app, shapes
from pyglet.math import Mat4, Vec3, Vec4
import math
from pyglet.gl import *

import shader

class CustomGroup(pyglet.graphics.Group):
    '''
    To draw multiple 3D shapes in Pyglet, you should make a group for an object.
    '''
    def __init__(self, transform_mat: Mat4, order, type, group, rotation_matrix = Mat4.from_translation(vector=Vec3(x=0, y=0, z=0))):
        super().__init__(order)

        '''
        Create shader program for each shape
        '''
        self.shader_program = shader.create_program(
            shader.vertex_source_default, shader.fragment_source_default
        )

        # shape.ransform_mat
        self.type = type
        self.group = group
        self.rotation_matrix = rotation_matrix

        self.transform_mat = transform_mat
        self.indexed_vertices_list = None
        self.shader_program.use()

    def set_state(self):
        self.shader_program.use()
        model = self.transform_mat
        self.shader_program['model'] = model

    def unset_state(self):
        self.shader_program.stop()

    def __eq__(self, other):
        return (self.__class__ is other.__class__ and
                self.order == other.order and
                self.parent == other.parent)
    
    def __hash__(self):
        return hash((self.order)) 
    

class Cube:
    '''
    default structure of cube
    '''
    def __init__(self, scale_x=1.0, scale_y=1.0, scale_z=1.0):
        # RenderWindow.update[0~23]
        # indexed_vertices_list.vertices
        self.vertices = [
            -0.5*scale_x, -0.5*scale_y, 0.5*scale_z,
            0.5*scale_x, -0.5*scale_y, 0.5*scale_z,
            0.5*scale_x, 0.5*scale_y, 0.5*scale_z,
            -0.5*scale_x, 0.5*scale_y, 0.5*scale_z,
            -0.5*scale_x, -0.5*scale_y, -0.5*scale_z,
            0.5*scale_x, -0.5*scale_y, -0.5*scale_z,
            0.5*scale_x,0.5*scale_y,-0.5*scale_z,
            -0.5*scale_x,0.5*scale_y,-0.5*scale_z]
        # self.vertices = [scale[idx%3] * x for idx, x in enumerate(self.vertices)]
    
        self.indices = [0, 1, 2, 2, 3, 0,
                    4, 7, 6, 6, 5, 4,
                    4, 5, 1, 1, 0, 4,
                    6, 7, 3, 3, 2, 6,
                    5, 6, 2, 2, 1, 5,
                    7, 4, 0, 0, 3, 7]
        
        self.colors = []
    
        for _ in range(8):
            self.colors.extend([160, 120, 90, 256]) # brown

class Face:
    def __init__(self, scale_x=1.0, scale_y=1.0, colors = "white"):

        self.face_color = []

        self.vertices = [
            0.5*scale_x, 0.5*scale_y, 0,
            -0.5*scale_x, 0.5*scale_y, 0,
            -0.5*scale_x, -0.5*scale_y, 0,
            0.5*scale_x, -0.5*scale_y, 0]
    
        self.indices = [0,1,2,2,3,0,
                        2,1,0,0,3,2]
    
        self.colors = []

        if colors == "white": self.face_color = [255, 255, 255, 0]
        elif colors == "red": self.face_color = [255, 0, 0, 0]
        elif colors == "green": self.face_color = [0, 255, 0, 0]
        else: pass

        for _ in range(4):
            self.colors.extend(self.face_color)

class Face_with_Hole:
    def __init__(self, scale_x=1.0, scale_y=1.0, hole_x=0.5, hole_y=0.5, colors = "white"):

        self.face_color = []

        self.vertices = [
            0.5*scale_x, 0.5*scale_y, 0,
            -0.5*scale_x, 0.5*scale_y, 0,
            -0.5*scale_x, -0.5*scale_y, 0,
            0.5*scale_x, -0.5*scale_y, 0, # Face
            0.5*hole_x, 0.5*hole_y, 0,
            -0.5*hole_x, 0.5*hole_y, 0,
            -0.5*hole_x, -0.5*hole_y, 0,
            0.5*hole_x, -0.5*hole_y, 0 # Hole
            ]
    
        self.indices = [4,0,1,1,5,4,
                        5,1,2,2,6,5,
                        6,2,3,3,7,6,
                        7,3,0,0,4,7,
                        1,0,4,4,5,1,
                        2,1,5,5,6,2,
                        3,2,6,6,7,3,
                        0,3,7,7,4,0
                        ]
    
        self.colors = []

        if colors == "white": self.face_color = [255, 255, 255, 0]
        elif colors == "red": self.face_color = [255, 0, 0, 0]
        elif colors == "green": self.face_color = [0, 255, 0, 0]
        else: pass

        for _ in range(8):
            self.colors.extend(self.face_color)

class Hollow_Cylinder:
    def __init__(self, outer_radius=1, inner_radius=0.5, height=1, slices=32):

        self.vertices = []
        self.indices = []
        self.colors = []

        for i in range(slices+1):
            angle = 2 * i * math.pi / slices
            outer_x = outer_radius * math.cos(angle)
            outer_z = outer_radius * math.sin(angle)

            inner_x = inner_radius * math.cos(angle)
            inner_z = inner_radius * math.sin(angle)

            self.vertices.extend([outer_x, height/2, outer_z])
            self.vertices.extend([inner_x, height/2, inner_z])
            self.vertices.extend([outer_x, -height/2, outer_z])
            self.vertices.extend([inner_x, -height/2, inner_z])
            
            for _ in range(4):
                self.colors.extend([245, 245, 220, 256]) # beige

        for i in range(slices): # 위아래면 
            index1 = i * 4
            index2 = (i + 1) * 4
            index3 = index1 + 1
            index4 = index2 + 1

            self.indices.extend([index1, index2, index4, 
                                 index4, index3, index1]) # 위
            
            self.indices.extend([index4+2, index2+2, index1+2, 
                                 index1+2, index3+2, index4+2]) # 아래 
            
        for i in range(slices): # 안, 바깥 면
            index1 = i * 4
            index2 = (i + 1) * 4
            index3 = index1 + 2
            index4 = index2 + 2

            self.indices.extend([index1+1, index2+1, index4+1, 
                                 index4+1, index3+1, index1+1]) # 안
            
            self.indices.extend([index4, index2, index1, 
                                 index1, index3, index4]) # 바깥 
        
