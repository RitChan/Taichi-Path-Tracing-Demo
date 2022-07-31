import math
from typing import List, Union, Tuple, Sequence

import numpy as np
import taichi as ti
import taichi.types

# Self-defined types
Vector3f = ti.types.vector(3, ti.float32)
Ray = ti.types.struct(o=Vector3f, d=Vector3f)
HitRecord = ti.types.struct(hit=ti.int32,
                            t=ti.float32)  # currently, flag only indicates whether a ray hit a triangle or not


@ti.data_oriented
class RenderingKernel:
    def __init__(self,
                 triangles: Sequence[Sequence],
                 materials: Sequence,
                 lights: Sequence["Light"],
                 camera: "Camera",
                 canvas: "Canvas"):
        # Basic Parameters:
        # (1) triangles: nx3 matrix where n%3 == 0
        # (2) triangle material types
        # (3) lights
        # (4) camera
        # (5) canvas
        triangles = np.asarray(triangles, dtype="f4").reshape((-1, 3, 3))
        materials = np.asarray(materials, dtype="i4")
        assert len(materials) == len(triangles)
        self.triangles = ti.Matrix.field(n=3, m=3, dtype=ti.f32, shape=triangles.shape[0])
        self.triangles.from_numpy(triangles)
        self.materials = ti.field(dtype=ti.int32, shape=materials.shape)
        self.materials.from_numpy(materials)
        self.lights = lights
        self.camera = camera
        self.canvas = canvas

    @ti.kernel
    def render(self):
        for i, j in self.canvas.buffer:
            i_f = float(i) + 0.5
            j_f = float(j) + 0.5
            rel_x = i_f / float(self.canvas.width)
            rel_y = j_f / float(self.canvas.height)
            view_ray = self.camera.ray_cast(rel_x, rel_y)
            min_t = 1000.1
            hit = 0
            for k in range(self.triangles.shape[0]):
                v0 = self.triangles[k][0, :].transpose()
                v1 = self.triangles[k][1, :].transpose()
                v2 = self.triangles[k][2, :].transpose()
                n = (v1 - v0).cross(v2 - v0)
                hit_record = ray_triangle_intersection(view_ray.o, view_ray.d, v0, v1, v2, n, 0.1, 1000.0)
                if hit_record.hit == 1:
                    hit = 1
                    if hit_record.t < min_t:
                        min_t = hit_record.t
            if hit == 1:
                self.canvas.buffer[i, j] = ti.Vector([min_t, min_t, min_t], ti.f32)

    @ti.func
    def sample(self, x, k_o):
        pass


@ti.data_oriented
class Camera:
    def __init__(self,
                 eye: Sequence,
                 forward: Sequence,
                 up: Sequence,
                 aspect_ratio: float,
                 fov_degree: float = 60):
        forward = np.asarray(forward)
        forward = forward / np.linalg.norm(forward)
        up = np.asarray(up)
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        self.eye = ti.Matrix(eye, dt=ti.f32)
        self.forward = ti.Matrix(forward, dt=ti.f32)
        self.up = ti.Matrix(up, dt=ti.f32)
        self.right = ti.Matrix(right, dt=ti.f32)
        self.width_f = ti.float32 = 1
        self.height_f = ti.float32 = aspect_ratio
        self.half_w: ti.float32 = self.width_f / 2
        self.half_h: ti.float32 = self.height_f / 2
        self.z_near: ti.float32 = self.half_h / math.tan(math.radians(fov_degree / 2))

    @ti.func
    def ray_cast(self, rel_x: ti.f32, rel_y: ti.f32) -> Ray:
        p = self.eye + self.z_near * self.forward + \
            (rel_x * self.width_f - self.half_w) * self.right + \
            (rel_y * self.height_f - self.half_h) * self.up
        d = p - self.eye
        d = d * ti.rsqrt(d.transpose() @ d)[0]
        return Ray(o=self.eye, d=d)


@ti.data_oriented
class Light:
    def __init__(self):
        self.x = ti.Vector([1, 2, 3], dt=ti.i32)


@ti.data_oriented
class Canvas:
    def __init__(self, width, height):
        self.width: ti.i32 = width
        self.height: ti.i32 = height
        self.buffer = Vector3f.field(shape=(width, height))

    @ti.kernel
    def clear(self, color: Vector3f):
        for i, j in self.buffer:
            self.buffer[i, j] = color

    @ti.kernel
    def normalize(self, denominator: ti.float32):
        for i, j in self.buffer:
            self.buffer[i, j] = self.buffer[i, j] / denominator

    @ti.kernel
    def normalize_depth(self, a: ti.float32, b: ti.float32):
        """pixel = |pixel - a| / b"""
        for i, j in self.buffer:
            self.buffer[i, j] = ti.abs(self.buffer[i, j] - ti.Vector([a, a, a], ti.f32)) / b


@ti.func
def ray_triangle_intersection(o, d, v0, v1, v2, n, t0, t1):
    hit_record = HitRecord(hit=1, t=0)
    a = o - v0
    b = v1 - v0
    c = v2 - v0
    if ti.abs((a.transpose() @ n)[0]) < 1e-3:  # d与n平行, 我们认为没有交点
        hit_record.hit = 0
    if hit_record.hit == 1:
        M = ti.Matrix([
            [b[0], c[0], -d[0]],
            [b[1], c[1], -d[1]],
            [b[2], c[2], -d[2]]])
        x = M.inverse() @ a
        alpha = 1 - x[0] - x[1]
        if 0 < x[0] < 1 and 0 < x[1] < 1 and 0 < alpha < 1:
            hit_record.t = x[2]
        else:
            hit_record.hit = 0
    return hit_record
