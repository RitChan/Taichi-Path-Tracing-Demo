import numpy as np
import taichi as ti

from rendering import Camera, RenderingKernel, Canvas, Lights

if __name__ == '__main__':
    ti.init(arch=ti.cuda, default_fp=ti.f32, default_ip=ti.i32)

    WIDTH, HEIGHT = 700, 700
    triangles = [[-0.5, 0, -1], [0.5, 0, -1], [-0.5, 0.5, -2], [-0.5, 0.5, -2], [0.5, 0, -1], [0.5, 0.5, -2]]
    lights = Lights([
        Lights.make_light_data(origin=[-0.5, 0.5, -2], dir_x=[1, 0, 0], dir_y=[0, 0, 1], power=5, color=(1, 1, 1))
    ])
    camera = Camera(eye=[0, 0, 0], forward=[0, 0, -1], up=[0, 1, 0], aspect_ratio=WIDTH / HEIGHT, fov_degree=60)
    kernel = RenderingKernel(triangles, [0, 0], lights, camera, Canvas(WIDTH, HEIGHT))

    kernel.canvas.clear(ti.Vector([0, 0, 0], dt=ti.float32))
    # kernel.render(1)
    iter_count = 1
    gui = ti.GUI("Triangle", res=(WIDTH, HEIGHT))
    while gui.running:
        kernel.render(iter_count)
        gui.set_image(kernel.canvas.buffer)
        gui.show()
        iter_count += 1
        # print(f"\r                     \rIter={iter_count}", end="")
