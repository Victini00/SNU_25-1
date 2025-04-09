import pyglet
from pyglet import window, app, shapes
from pyglet.window import mouse,key

from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.gl import GL_TRIANGLES
from pyglet.math import Mat4, Vec3, Vec4
from pyglet.gl import *

import shader
import math
import time
from primitives import CustomGroup

class RenderWindow(pyglet.window.Window):
    '''
    inherits pyglet.window.Window which is the default render window of Pyglet
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch = pyglet.graphics.Batch()
        '''
        View (camera) parameters
        '''
        self.cam_eye = Vec3(0,7,7)
        self.cam_target = Vec3(0,2,0)
        self.cam_vup = Vec3(0,1,0)
        self.view_mat = None
        '''
        Projection parameters
        '''
        self.z_near = 0.1
        self.z_far = 100
        self.fov = 100
        self.proj_mat = None

        self.shapes = []
        self.setup()

        self.animate = False

        '''
        Propeller settings
        '''
        self.vertical_movement_state = {}
        self.vertical_movement_limits = {
            "Propeller": {"min": 0, "max": 2},
            "Head": {"min": 0, "max": 2}
        }

    def setup(self) -> None:
        self.set_minimum_size(width = 400, height = 300)
        self.set_mouse_visible(True)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)

        # 1. Create a view matrix
        self.view_mat = Mat4.look_at(
            self.cam_eye, target=self.cam_target, up=self.cam_vup)
        
        # 2. Create a projection matrix 
        self.proj_mat = Mat4.perspective_projection(
            aspect = self.width/self.height, 
            z_near=self.z_near, 
            z_far=self.z_far, 
            fov = self.fov)

    def on_draw(self) -> None:
        self.clear()
        self.batch.draw()


    def update(self,dt) -> None:
        view_proj = self.proj_mat @ self.view_mat
        for shape in self.shapes:            
            '''
            Update position/orientation in the scene. In the current setting, 
            shapes created later rotate faster while positions are not changed.
            '''
            if self.animate:
                if shape.group == "car":
                    if shape.type == "Wheel":
                        rotate_angle = 0.7 * dt 
                        rotate_axis = Vec3(0, 1, 0)
                        rotate_mat = Mat4.from_rotation(angle=rotate_angle, vector=rotate_axis)
                        
                        shape.transform_mat = shape.transform_mat @ rotate_mat
                    
                    if shape.type == "Propeller":
                        rotate_angle = 5 * dt
                        rotate_axis = Vec3(1, 0, 0)

                        rotate_mat = Mat4.from_rotation(angle=rotate_angle, vector=rotate_axis)
                        shape.transform_mat = shape.transform_mat @ rotate_mat

                    if shape.type == "Head":
                        rotate_angle = 5 * dt
                        rotate_axis = Vec3(0, 1, 0)

                        rotate_mat = Mat4.from_rotation(angle=rotate_angle, vector=rotate_axis)
                        shape.transform_mat = shape.transform_mat @ rotate_mat

                    if shape.type == "Propeller" or shape.type == "Head":
                        y_offset = 2* dt * math.sin(time.time() * math.pi * 2)  # 1초 주기로 왕복
                        wing_translation = Mat4.from_translation(vector=Vec3(x=0, y=y_offset, z=0))
                        shape.transform_mat = wing_translation @ shape.transform_mat

                global_translation = Mat4.from_translation(vector=Vec3(x=-0.7 * dt, y=0, z=0))
                shape.transform_mat = global_translation @ shape.transform_mat

                

            '''
            Update view and projection matrix. There exist only one view and projection matrix 
            in the program, so we just assign the same matrices for all the shapes
            '''
            shape.shader_program['view_proj'] = view_proj


    def on_resize(self, width, height):
        glViewport(0, 0, *self.get_framebuffer_size())
        self.proj_mat = Mat4.perspective_projection(
            aspect = width/height, z_near=self.z_near, z_far=self.z_far, fov = self.fov)
        return pyglet.event.EVENT_HANDLED

    def add_shape(self, transform, vertice, indice, color, type, group, rotation_matrix=Mat4.from_translation(vector=Vec3(x=0, y=0, z=0))):
        
        '''
        Assign a group for each shape
        '''
        shape = CustomGroup(transform, len(self.shapes), type, group, rotation_matrix=rotation_matrix)
        shape.indexed_vertices_list = shape.shader_program.vertex_list_indexed(len(vertice)//3, GL_TRIANGLES,
                        batch = self.batch,
                        group = shape,
                        indices = indice,
                        vertices = ('f', vertice),
                        colors = ('Bn', color))
        self.shapes.append(shape)

    def transform_group(self, group_name, global_transform):
        """
        특정 그룹의 모든 shape에 글로벌 변환을 적용합니다.
    
        :param group_name: 변환할 그룹의 이름
        :param global_transform: 적용할 변환 행렬 (Mat4)
        """
        for shape in self.shapes:
            if shape.group == group_name:
                shape.transform_mat = global_transform @ shape.transform_mat
         
    def run(self):
        pyglet.clock.schedule_interval(self.update, 1/60)
        pyglet.app.run()

    