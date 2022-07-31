import numpy as np
import taichi as ti

from rendering import Camera, RenderingKernel, Canvas, Light

if __name__ == '__main__':
    ti.init(arch=ti.cuda, default_fp=ti.f32, default_ip=ti.i32)

    WIDTH, HEIGHT = 300, 300
    triangles = [[-0.5, -0.5, -1], [0.5, -0.5, -1], [0, 0.5, -1]]
    camera = Camera(eye=[0, 0, 0], forward=[0, 0, -1], up=[0, 1, 0], aspect_ratio=WIDTH / HEIGHT, fov_degree=60)
    kernel = RenderingKernel(triangles, [0], [], camera, Canvas(WIDTH, HEIGHT))
    kernel.canvas.clear(ti.Vector([0, 0, 0], dt=ti.float32))
    kernel.render()
    gui = ti.GUI("Triangle", res=(WIDTH, HEIGHT))
    while gui.running:
        gui.set_image(kernel.canvas.buffer)
        gui.show()
