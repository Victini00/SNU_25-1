import pyglet
import sys
from pyglet.math import Mat4, Vec3
import math

sys.path.append("./scripts")
from render import RenderWindow
from primitives import Cube,Sphere,Cylinder,Propeller
from control import Control


if __name__ == '__main__':
    width = 1280
    height = 720

    # Render window.
    renderer = RenderWindow(width, height, "Hello Pyglet", resizable = True)   
    renderer.set_location(200, 200)

    # Keyboard/Mouse control. Not implemented yet.
    controller = Control(renderer)

    propeller_radius = 0.8

    translate_wheel1 = Mat4.from_translation(vector=Vec3(x=-2, y=0, z=1.5))
    translate_wheel2 = Mat4.from_translation(vector=Vec3(x=-2, y=0, z=-1.5))
    translate_wheel3 = Mat4.from_translation(vector=Vec3(x=2, y=0, z=1.5))
    translate_wheel4 = Mat4.from_translation(vector=Vec3(x=2, y=0, z=-1.5))

    translate_body = Mat4.from_translation(vector=Vec3(x=0, y=0.5, z=0))
    translate_head = Mat4.from_translation(vector=Vec3(x=0, y=3, z=0))
    translate_propeller = Mat4.from_translation(vector=Vec3(x=0, y=4, z=0))

    translate_center_cylinder = Mat4.from_translation(vector=Vec3(x=0, y=1.5, z=0))

    rotation_wheel = Mat4.from_rotation(angle=math.pi/2, vector=Vec3(1,0,0))

    rotation_propeller_1 = Mat4.from_rotation(angle=math.pi/2, vector=Vec3(0,0,1))

    rotation_propeller_2 = Mat4.from_rotation(angle=math.pi/2, vector=Vec3(0,0,-1))
    rotation_propeller_reverse = Mat4.from_rotation(angle=math.pi, vector=Vec3(0,1,0))
    

    
    # scale_vec = Vec3(x=1, y=1, z=1)
    # sphere = Sphere(30,30)

    wheel = Cylinder()
    body = Cube(scale_x=6, scale_y=2, scale_z=3)
    head = Cylinder(radius = 0.2, height = 3, option="Head")
    propeller = Propeller(radius = propeller_radius, height = 0.3)
    center_cylinder = Cylinder(radius = 1, height= 1, option="Center_Cylinder")


    # body
    renderer.add_shape(translate_body, body.vertices, body.indices, body.colors, type="Body", group="car")

    # wheel
    renderer.add_shape(translate_wheel1@rotation_wheel, wheel.vertices, wheel.indices, wheel.colors, type="Wheel", group="car", rotation_matrix = rotation_wheel)
    renderer.add_shape(translate_wheel2@rotation_wheel, wheel.vertices, wheel.indices, wheel.colors, type="Wheel", group="car", rotation_matrix = rotation_wheel)
    renderer.add_shape(translate_wheel3@rotation_wheel, wheel.vertices, wheel.indices, wheel.colors, type="Wheel", group="car", rotation_matrix = rotation_wheel)
    renderer.add_shape(translate_wheel4@rotation_wheel, wheel.vertices, wheel.indices, wheel.colors, type="Wheel", group="car", rotation_matrix = rotation_wheel)

    # head
    renderer.add_shape(translate_head, head.vertices, head.indices, head.colors, type="Head", group="car")
    
    # propeller
    renderer.add_shape(translate_propeller@rotation_propeller_1, propeller.vertices, propeller.indices, propeller.colors, type="Propeller", group="car", rotation_matrix = rotation_propeller_1)
    renderer.add_shape(translate_propeller@rotation_propeller_2@rotation_propeller_reverse, propeller.vertices, propeller.indices, propeller.colors, type="Propeller", group="car", rotation_matrix = rotation_propeller_2@rotation_propeller_reverse)

    # center_cylinder
    renderer.add_shape(translate_center_cylinder, center_cylinder.vertices, center_cylinder.indices, center_cylinder.colors, type="Center_Cylinder", group="car")

    #draw shapes
    renderer.run()
