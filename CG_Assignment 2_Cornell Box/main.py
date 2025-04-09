import pyglet
from pyglet.math import Mat4, Vec3
import sys, math
sys.path.append("./scripts")

from render_hw2 import RenderWindow
from primitives_hw2 import Cube, Face, Face_with_Hole, Hollow_Cylinder
from control import Control

import camera

if __name__ == '__main__':
    width = 1280
    height = 720

    mouseRotatePressed = False
    mouseMovePressed   = False
    mouseDollyPressed   = False

    # Render window.
    renderer = RenderWindow(width, height, "Cornell Box", resizable = True)   
    renderer.set_location(200, 200)

    # Keyboard/Mouse control. Not implemented yet.
    controller = Control(renderer)

    # Cornell Box

    Cornell_width   = 32
    Cornell_depth   = 19.5
    Cornell_height  = 25

    Cornell_hole_x = 9.5
    Cornell_hole_y = 4.5

    Cornell_back_trans      = Mat4.from_translation(vector=Vec3(x=0, y=0, z=-(Cornell_depth/2)))
    Cornell_up_trans        = Mat4.from_translation(vector=Vec3(x=0, y=Cornell_height/2, z=0))
    Cornell_down_trans      = Mat4.from_translation(vector=Vec3(x=0, y=-Cornell_height/2, z=0))
    Cornell_left_trans      = Mat4.from_translation(vector=Vec3(x=-Cornell_width/2, y=0, z=0))
    Cornell_right_trans     = Mat4.from_translation(vector=Vec3(x=Cornell_width/2, y=0, z=0))

    Cornell_back_rot        = Mat4.from_rotation(angle=0, vector=Vec3(0,0,1))
    Cornell_up_rot          = Mat4.from_rotation(angle=math.pi/2, vector=Vec3(1,0,0))
    Cornell_down_rot        = Mat4.from_rotation(angle=-math.pi/2, vector=Vec3(1,0,0))
    Cornell_left_rot        = Mat4.from_rotation(angle=math.pi/2, vector=Vec3(0,1,0))
    Cornell_right_rot       = Mat4.from_rotation(angle=-math.pi/2, vector=Vec3(0,1,0))
    
    Cornell_back   = Face(scale_x=Cornell_width, scale_y=Cornell_height, colors="white")
    Cornell_up     = Face_with_Hole(scale_x=Cornell_width, scale_y=Cornell_depth, hole_x=Cornell_hole_x, hole_y=Cornell_hole_y, colors="white")
    Cornell_down   = Face(scale_x=Cornell_width, scale_y=Cornell_depth, colors="white")
    Cornell_left   = Face(scale_x=Cornell_depth, scale_y=Cornell_height, colors="red")
    Cornell_right  = Face(scale_x=Cornell_depth, scale_y=Cornell_height, colors="green")

    # Tape

    Tape_outer_radius = 9/2
    Tape_inner_radius = 7.5/2
    Tape_height = 5

    Tape_x = 5
    Tape_z = 0

    Tape_trans      = Mat4.from_translation(vector=Vec3(x=Tape_x, y=(-Cornell_height/2 + Tape_height/2 + 0.01), z=Tape_z))
    Tape_rot       = Mat4.from_rotation(angle=0, vector=Vec3(0,1,0))
    Tape  = Hollow_Cylinder(outer_radius=Tape_outer_radius, inner_radius=Tape_inner_radius, height=Tape_height)

    # Box

    Box_width = 8
    Box_depth = 5.5
    Box_height = 12

    Box_x = -5
    Box_z = 0

    Box_trans = Mat4.from_translation(vector=Vec3(x=Box_x, y=(-Cornell_height/2 + Box_height/2 + 0.01), z=Box_z))
    Box_rot       = Mat4.from_rotation(angle=0, vector=Vec3(0,1,0))
    Box  = Cube(scale_x=Box_width, scale_y=Box_height, scale_z=Box_depth)

    #####################################################################################################################
    # Cornell Box
    
    renderer.add_shape(Cornell_back_trans@Cornell_back_rot, Cornell_back.vertices, Cornell_back.indices, Cornell_back.colors, type="Cornell Box", group="Cornell Box")
    renderer.add_shape(Cornell_up_trans@Cornell_up_rot, Cornell_up.vertices, Cornell_up.indices, Cornell_up.colors, type="Cornell Box", group="Cornell Box")
    renderer.add_shape(Cornell_down_trans@Cornell_down_rot, Cornell_down.vertices, Cornell_down.indices, Cornell_down.colors, type="Cornell Box", group="Cornell Box")
    renderer.add_shape(Cornell_left_trans@Cornell_left_rot, Cornell_left.vertices, Cornell_left.indices, Cornell_left.colors, type="Cornell Box", group="Cornell Box")
    renderer.add_shape(Cornell_right_trans@Cornell_right_rot, Cornell_right.vertices, Cornell_right.indices, Cornell_right.colors, type="Cornell Box", group="Cornell Box")
    

    # Tape
    renderer.add_shape(Tape_trans@Tape_rot, Tape.vertices, Tape.indices, Tape.colors, type="Tape", group="Cornell Box")

    # Box
    renderer.add_shape(Box_trans@Box_rot, Box.vertices, Box.indices, Box.colors, type="Box", group="Cornell Box")


    @renderer.event
    def on_resize(width, height):
        camera.resize(renderer, width, height )
        return pyglet.event.EVENT_HANDLED

    @renderer.event
    def on_draw():
        renderer.clear()
        camera.apply(renderer)
        renderer.batch.draw()

    @renderer.event
    def on_key_press( key, mods ):	
        if key==pyglet.window.key.Q:
            pyglet.app.exit()
            
    @renderer.event
    def on_mouse_release( x, y, button, mods ):
        global mouseRotatePressed, mouseMovePressed, mouseDollyPressed

        mouseMovePressed   = False
        mouseRotatePressed = False
        mouseDollyPressed   = False

    @renderer.event
    def on_mouse_press( x, y, button, mods ):
        global mouseRotatePressed, mouseMovePressed, mouseDollyPressed

        if button & pyglet.window.mouse.LEFT and mods & pyglet.window.key.MOD_SHIFT:
            mouseMovePressed   = True
            mouseRotatePressed = False
            mouseDollyPressed   = False
        elif button & pyglet.window.mouse.LEFT and mods & pyglet.window.key.MOD_CTRL:
            mouseMovePressed   = False
            mouseRotatePressed = False
            mouseDollyPressed   = True
        elif button & pyglet.window.mouse.LEFT:
            camera.beginRotate(x, y)
            mouseMovePressed   = False
            mouseRotatePressed = True
            mouseDollyPressed   = False

    @renderer.event
    def on_mouse_drag( x, y, dx, dy, button, mods ):	
        if mouseRotatePressed:
            camera.rotate(x, y)
        elif mouseMovePressed:
            camera.move(dx/width, dy/height, 0.0)
        elif mouseDollyPressed:
            camera.zoom(dy/height)	

    #draw shapes
    renderer.run()
