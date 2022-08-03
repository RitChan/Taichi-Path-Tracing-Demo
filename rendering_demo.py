import numpy as np
import taichi as ti

from rendering import Camera, RenderingKernel, Canvas, Lights, Triangles

if __name__ == '__main__':
    ti.init(arch=ti.cuda, default_fp=ti.f32, default_ip=ti.i32)

    WIDTH, HEIGHT = 700, 700
    BOX_POINTS = np.array([
        [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5],
    ], dtype="f4")
    BOX_SCALE = 0.8
    BOX_ROT = np.array((
        (0.38235995173454285, 0.06055793538689613, 0.3164389431476593),
        (0.14269095659255981, 0.4084676206111908, -0.25058630108833313),
        (-0.28886011242866516, 0.28193429112434387, 0.29508116841316223)
    ), dtype="f4")
    BOX_TRANSLATE = np.array([0, 0, -2.5])
    BOX_POINTS *= BOX_SCALE
    BOX_POINTS = (BOX_ROT @ BOX_POINTS.T).T + BOX_TRANSLATE
    triangles = Triangles([
        # Left
        Triangles.create([-1, -1, 0], [1, -1, 0], [1, -1, -6], color=(0.8, 0.8, 0.8)),
        Triangles.create([-1, -1, 0], [1, -1, -6], [-1, -1, -6], color=(0.8, 0.8, 0.8)),
        # Right
        Triangles.create([-1, -3, 0], [-1, -3, -6], [-1, 3, -6], color=(1.0, 0.0, 0.0)),
        Triangles.create([-1, -3, 0], [-1, 3, -6], [-1, 3, 0], color=(1.0, 0.0, 0.0)),
        # Bottom
        Triangles.create([1, -3, 0], [1, 3, -6], [1, -3, -6], color=(0.0, 0.0, 1.0)),
        Triangles.create([1, -3, 0], [1, 3, 0], [1, 3, -6], color=(0.0, 0.0, 1.0)),
        # Back
        Triangles.create([-1, -1, -4], [1, -1, -4], [1, 3, -4], color=(1, 1, 1)),
        Triangles.create([-1, -1, -4], [1, 3, -4], [-1, 3, -4], color=(1, 1, 1)),
        # Top
        Triangles.create([-1, 1.01, 0], [-1, 1.01, -4], [1, 1.01, -4], color=(1, 1, 1)),
        Triangles.create([-1, 1.01, 0], [1, 1.01, -4], [1, 1.01, 0], color=(1, 1, 1)),
        # Box
        Triangles.create(BOX_POINTS[0], BOX_POINTS[3], BOX_POINTS[1], color=(1, 1, 1)),
        Triangles.create(BOX_POINTS[1], BOX_POINTS[3], BOX_POINTS[2], color=(1, 1, 1)),
        Triangles.create(BOX_POINTS[4], BOX_POINTS[5], BOX_POINTS[7], color=(1, 1, 1)),
        Triangles.create(BOX_POINTS[5], BOX_POINTS[6], BOX_POINTS[7], color=(1, 1, 1)),
        Triangles.create(BOX_POINTS[0], BOX_POINTS[1], BOX_POINTS[5], color=(1, 1, 1)),
        Triangles.create(BOX_POINTS[0], BOX_POINTS[5], BOX_POINTS[4], color=(1, 1, 1)),
        Triangles.create(BOX_POINTS[3], BOX_POINTS[7], BOX_POINTS[6], color=(1, 1, 1)),
        Triangles.create(BOX_POINTS[3], BOX_POINTS[6], BOX_POINTS[2], color=(1, 1, 1)),
        Triangles.create(BOX_POINTS[1], BOX_POINTS[2], BOX_POINTS[6], color=(1, 1, 1)),
        Triangles.create(BOX_POINTS[1], BOX_POINTS[6], BOX_POINTS[5], color=(1, 1, 1)),
        Triangles.create(BOX_POINTS[0], BOX_POINTS[4], BOX_POINTS[3], color=(1, 1, 1)),
        Triangles.create(BOX_POINTS[3], BOX_POINTS[4], BOX_POINTS[7], color=(1, 1, 1)),
    ])
    lights = Lights([
        Lights.create(origin=[-0.25, 1, -3], dir_x=[0.5, 0, 0], dir_y=[0, 0, 0.5], power=10, color=(1, 1, 1))
    ])
    camera = Camera(eye=[0, 0, 0], forward=[0, 0, -1], up=[0, 1, 0], aspect_ratio=WIDTH / HEIGHT, fov_degree=60)
    kernel = RenderingKernel(triangles, lights, camera, Canvas(WIDTH, HEIGHT))

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
