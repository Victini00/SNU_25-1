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
    
        self.colors = [
            255, 255, 0, 255, # 노랑
            255, 255, 0, 255, 
            255, 255, 0, 255, 
            255, 255, 0, 255, 
            255, 255, 0, 255, 
            255, 255, 0, 255, 
            255, 255, 0, 255, 
            255, 255, 0, 255
        ]


class Cylinder:
    def __init__(self, radius=1, height=1, slices=32, option="Wheel"):

        self.vertices = []
        self.indices = []
        self.colors = []

        for i in range(slices + 1):
            angle = 2 * i * math.pi / slices
            x = radius * math.cos(angle)
            z = radius * math.sin(angle)

            self.vertices.extend([x, height/2, z])

            # 위아래 원원
            if option == "Head":
                self.colors.extend([64,128,0,255])
            elif option == "Wheel":
                if i%8 == 0: self.colors.extend([0,0,0,255])
                else: self.colors.extend([255,255,255,255])
            else:
                self.colors.extend([255,0,0,255])

            self.vertices.extend([x, -height/2, z])
            if option == "Head":
                self.colors.extend([64,128,0,255])
            elif option == "Wheel":
                if i%8 == 0: self.colors.extend([0,0,0,255])
                else: self.colors.extend([255,255,255,255])
            else:
                self.colors.extend([255,0,0,255])

        # 옆면
        for i in range(slices):
            index1 = i * 2
            index2 = (i + 1) * 2
            index3 = index1 + 1
            index4 = index2 + 1

            self.indices.extend([index1, index2, index3, 
                                 index2, index4, index3])

        top_center = len(self.vertices) // 3
        bottom_center = top_center + 1

        # 중심
        self.vertices.extend([0, height/2, 0])  
        if option == "Wheel": self.colors.extend([0,0,0,255])
        else: self.colors.extend([255,0,0,255])

        self.vertices.extend([0, -height/2, 0]) 
        if option == "Wheel": self.colors.extend([0,0,0,255])
        else: self.colors.extend([255,0,0,255])


        for i in range(slices):
            index1 = i * 2
            index2 = ((i + 1) % slices) * 2

            # 윗면은 시계 방향
            self.indices.extend([top_center, index2, index1])
            
            # 아랫면은 시계 방향
            self.indices.extend([bottom_center, index1 + 1, index2 + 1])

class Sphere:
    '''
    default structure of sphere
    '''
    def __init__(self, stacks, slices, scale=1.0):
        num_triangles = 2 * slices * (stacks - 1)

        self.vertices = []
        self.indices = []
        self.colors = ()

        for i in range(stacks):
            phi0 = 0.5 * math.pi - (i * math.pi) / stacks
            phi1 = 0.5 * math.pi - ((i + 1) * math.pi) / stacks
            coord_v0 = 1.0 - float(i) / stacks
            coord_v1 = 1.0 - float(i + 1) / stacks

            y0 = scale * math.sin(phi0)
            r0 = scale * math.cos(phi0)
            y1 = scale * math.sin(phi1)
            r1 = scale * math.cos(phi1)
            y2 = y1
            y3 = y0

            for j in range(slices):
                theta0 = (j * 2 * math.pi) / slices
                theta1 = ((j + 1) * 2 * math.pi) / slices
                coord_u0 = float(j) / slices
                coord_u1 = float(j + 1) / slices

                x0 = r0 * math.cos(theta0)
                z0 = r0 * math.sin(-theta0)
                u0 = coord_u0
                v0 = coord_v0
                x1 = r1 * math.cos(theta0)
                z1 = r1 * math.sin(-theta0)
                u1 = coord_u0
                v1 = coord_v1
                x2 = r1 * math.cos(theta1)
                z2 = r1 * math.sin(-theta1)
                u2 = coord_u1
                v2 = coord_v1
                x3 = r0 * math.cos(theta1)
                z3 = r0 * math.sin(-theta1)
                u3 = coord_u1
                v3 = coord_v0

                if (i != stacks - 1):
                    self.vertices.append(x0)
                    self.vertices.append(y0)
                    self.vertices.append(z0)

                    self.vertices.append(x1)
                    self.vertices.append(y1)
                    self.vertices.append(z1)

                    self.vertices.append(x2)
                    self.vertices.append(y2)
                    self.vertices.append(z2)
                    
                    self.colors += (int(math.cos(phi0) * 255),int(math.cos(theta0) * 255),int(math.sin(phi0)*255),255)
                    self.colors += (int(math.cos(phi0) * 255),int(math.cos(theta0) * 255),int(math.sin(phi0)*255),255)
                    self.colors += (int(math.cos(phi0) * 255),int(math.cos(theta0) * 255),int(math.sin(phi0)*255),255)
                
                if (i != 0):
                    self.vertices.append(x2)
                    self.vertices.append(y2)
                    self.vertices.append(z2)

                    self.vertices.append(x3)
                    self.vertices.append(y3)
                    self.vertices.append(z3)

                    self.vertices.append(x0)
                    self.vertices.append(y0)
                    self.vertices.append(z0)
                    
                    self.colors += (int(math.cos(phi0) * 255),int(math.cos(theta0) * 255),int(math.sin(phi0)*255),255)
                    self.colors += (int(math.cos(phi0) * 255),int(math.cos(theta0) * 255),int(math.sin(phi0)*255),255)
                    self.colors += (int(math.cos(phi0) * 255),int(math.cos(theta0) * 255),int(math.sin(phi0)*255),255)

        for i in range(num_triangles*3):
            self.indices.append(i)


class Propeller:
    def __init__(self, radius=1, height=1, slices=30):
        self.vertices = []
        self.indices = []
        self.colors = []

        # 위치 보정값 - 정삼각형 아랫 부분을 y축 중점으로.
        coordinate_correction = (math.sqrt(3) / 2) * 2 * radius

        triangle_side = 2 * radius
        triangle_height = (math.sqrt(3) / 2) * triangle_side

        # 반원
        for i in range(slices + 1):
            angle = math.pi * i / slices  

            x = radius * math.cos(angle)
            z = radius * math.sin(angle)

            # 위아래 원

            self.vertices.extend([x, height/2, z + coordinate_correction])
            self.colors.extend([127, 255, 0, 255])
            self.vertices.extend([x, -height/2, z + coordinate_correction])
            self.colors.extend([127, 255, 0, 255])

        for i in range(slices + 1):
            index1 = i * 2
            index2 = (i + 1) * 2
            index3 = index1 + 1
            index4 = index2 + 1

            self.indices.extend([index1, index2, index3, 
                                 index2, index4, index3])

        top_vertices_start = len(self.vertices) // 3

        # 반원 중심점
        self.vertices.extend([0, height/2, coordinate_correction])
        self.colors.extend([127, 255, 0, 255])

        # 1. 왼쪽 
        self.vertices.extend([-radius, height/2, coordinate_correction])
        self.colors.extend([127, 255, 0, 255])

        # 2.오른쪽 
        self.vertices.extend([radius, height/2, coordinate_correction])
        self.colors.extend([127, 255, 0, 255])

        # 3.위쪽 
        self.vertices.extend([0, height/2, coordinate_correction-triangle_height])
        self.colors.extend([127, 255, 0, 255])

        # 윗면
        for i in range(slices + 1):
            index1 = i * 2
            index2 = (i + 1) * 2
            self.indices.extend([top_vertices_start, index2, index1])

        # 정삼각형 부분
        self.indices.extend([
            top_vertices_start + 1, 
            top_vertices_start + 2, 
            top_vertices_start + 3  
        ])

        # 여기부터 아랫면면
        bottom_vertices_start = len(self.vertices) // 3

        # 반원 중심점
        self.vertices.extend([0, -height/2, coordinate_correction])
        self.colors.extend([127, 255, 0, 255])
        self.vertices.extend([-radius, -height/2, coordinate_correction])
        self.colors.extend([127, 255, 0, 255])
        self.vertices.extend([radius, -height/2, coordinate_correction])
        self.colors.extend([127, 255, 0, 255])
        self.vertices.extend([0, -height/2, coordinate_correction-triangle_height])
        self.colors.extend([127, 255, 0, 255])

        # 아랫면
        for i in range(slices + 1):
            index1 = i * 2 + 1
            index2 = (i + 1) * 2 + 1
            self.indices.extend([bottom_vertices_start, index1, index2])

        # 정삼각형 부분
        self.indices.extend([
            bottom_vertices_start + 1,     
            bottom_vertices_start + 3,  
            bottom_vertices_start + 2
        ])

        # 정삼각형 부분 ~ 옆면
        self.indices.extend([
            top_vertices_start + 2,     
            bottom_vertices_start + 2,  
            top_vertices_start + 3, 
        
            bottom_vertices_start + 2,     
            bottom_vertices_start + 3,  
            top_vertices_start + 3, 
        
            top_vertices_start + 3,     
            bottom_vertices_start + 3,  
            top_vertices_start + 1, 
        
            bottom_vertices_start + 3,     
            bottom_vertices_start + 1,  
            top_vertices_start + 1 
        ])