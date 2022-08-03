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
    BOX_POINTS *= BOX_SCALE

    BOX1_ROT = np.array((
        (0.38235995173454285, 0.06055793538689613, 0.3164389431476593),
        (0.14269095659255981, 0.4084676206111908, -0.25058630108833313),
        (-0.28886011242866516, 0.28193429112434387, 0.29508116841316223)
    ), dtype="f4")
    BOX1_TRANSLATE = np.array([-0.3, -0.1, -2.5])
    BOX1 = (BOX1_ROT @ BOX_POINTS.T).T + BOX1_TRANSLATE

    BOX2_ROT = np.array((
        (0.3295649588108063, 0.31123486161231995, 0.21099717915058136),
        (0.18337589502334595, 0.11190993338823318, -0.451496958732605),
        (-0.32826852798461914, 0.37497878074645996, -0.04038276523351669)
    ), dtype="f4")
    BOX2_TRANSLATE = np.array([0.2, -0.4, -2.5])
    BOX2 = (BOX2_ROT @ BOX_POINTS.T).T + BOX2_TRANSLATE

    triangles = Triangles([
        # Bottom
        Triangles.create([-1, -1, 0], [1, -1, 0], [1, -1, -6], color=(1, 1, 1)),
        Triangles.create([-1, -1, 0], [1, -1, -6], [-1, -1, -6], color=(1, 1, 1)),
        # Left
        Triangles.create([-1, -3, 0], [-1, -3, -6], [-1, 3, -6], color=(223 / 255, 99 / 255, 99 / 255)),
        Triangles.create([-1, -3, 0], [-1, 3, -6], [-1, 3, 0], color=(223 / 255, 99 / 255, 99 / 255)),
        # Right
        Triangles.create([1, -3, 0], [1, 3, -6], [1, -3, -6], color=(91 / 255, 99 / 168, 79 / 255)),
        Triangles.create([1, -3, 0], [1, 3, 0], [1, 3, -6], color=(91 / 255, 99 / 168, 79 / 255)),
        # Back
        Triangles.create([-1, -1, -4], [1, -1, -4], [1, 3, -4], color=(1, 1, 1)),
        Triangles.create([-1, -1, -4], [1, 3, -4], [-1, 3, -4], color=(1, 1, 1)),
        # Top
        Triangles.create([-1, 1.01, 0], [-1, 1.01, -4], [1, 1.01, -4], color=(1, 1, 1)),
        Triangles.create([-1, 1.01, 0], [1, 1.01, -4], [1, 1.01, 0], color=(1, 1, 1)),
        # Box 1
        Triangles.create(BOX1[0], BOX1[3], BOX1[1], color=(1, 1, 1)),
        Triangles.create(BOX1[1], BOX1[3], BOX1[2], color=(1, 1, 1)),
        Triangles.create(BOX1[4], BOX1[5], BOX1[7], color=(1, 1, 1)),
        Triangles.create(BOX1[5], BOX1[6], BOX1[7], color=(1, 1, 1)),
        Triangles.create(BOX1[0], BOX1[1], BOX1[5], color=(1, 1, 1)),
        Triangles.create(BOX1[0], BOX1[5], BOX1[4], color=(1, 1, 1)),
        Triangles.create(BOX1[3], BOX1[7], BOX1[6], color=(1, 1, 1)),
        Triangles.create(BOX1[3], BOX1[6], BOX1[2], color=(1, 1, 1)),
        Triangles.create(BOX1[1], BOX1[2], BOX1[6], color=(1, 1, 1)),
        Triangles.create(BOX1[1], BOX1[6], BOX1[5], color=(1, 1, 1)),
        Triangles.create(BOX1[0], BOX1[4], BOX1[3], color=(1, 1, 1)),
        Triangles.create(BOX1[3], BOX1[4], BOX1[7], color=(1, 1, 1)),
        # Box 2
        Triangles.create(BOX2[0], BOX2[3], BOX2[1], color=(1, 1, 1)),
        Triangles.create(BOX2[1], BOX2[3], BOX2[2], color=(1, 1, 1)),
        Triangles.create(BOX2[4], BOX2[5], BOX2[7], color=(1, 1, 1)),
        Triangles.create(BOX2[5], BOX2[6], BOX2[7], color=(1, 1, 1)),
        Triangles.create(BOX2[0], BOX2[1], BOX2[5], color=(1, 1, 1)),
        Triangles.create(BOX2[0], BOX2[5], BOX2[4], color=(1, 1, 1)),
        Triangles.create(BOX2[3], BOX2[7], BOX2[6], color=(1, 1, 1)),
        Triangles.create(BOX2[3], BOX2[6], BOX2[2], color=(1, 1, 1)),
        Triangles.create(BOX2[1], BOX2[2], BOX2[6], color=(1, 1, 1)),
        Triangles.create(BOX2[1], BOX2[6], BOX2[5], color=(1, 1, 1)),
        Triangles.create(BOX2[0], BOX2[4], BOX2[3], color=(1, 1, 1)),
        Triangles.create(BOX2[3], BOX2[4], BOX2[7], color=(1, 1, 1)),
    ])
    lights = Lights([
        Lights.create(origin=[-0.25, 1, -3], dir_x=[0.5, 0, 0], dir_y=[0, 0, 0.5], power=20, color=(1, 1, 1))
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
