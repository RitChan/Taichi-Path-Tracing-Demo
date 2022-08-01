import math
from typing import List, Union, Tuple, Sequence

import numpy as np
import taichi as ti
import taichi.types

# Zero Flag
ZERO = 0x00000000

# Self-defined Types
Vector3f = ti.types.vector(3, ti.float32)
Ray = ti.types.struct(o=Vector3f, d=Vector3f)
HitRecord = ti.types.struct(flag=ti.int32, t=ti.float32, object_idx=ti.int32, material=ti.int32)
PointSample = ti.types.struct(pos=Vector3f, prob=ti.float32)

# HitRecord Flags
HIT = 0x00000001
IS_LIGHT = 0x00000002

# Material Type
# TODO


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
            hit_record = self.ray_hit_nearest(view_ray, 0.01, 1000.0)
            if flag_query(hit_record.flag, HIT) == 1:
                # self.canvas.buffer[i, j] = ti.Vector([hit_record.t, hit_record.t, hit_record.t], ti.f32)
                if flag_query(hit_record.flag, IS_LIGHT) == 1:
                    light = self.lights[0]
                    self.canvas.buffer[i, j] = light.radiance(-view_ray.d) * light.color

    @ti.func
    def sample(self, x, k_o):
        pass

    @ti.func
    def direct_light_radiance_at(self, x, k_o, normal, material):
        result = float(0)
        k_o = k_o * ti.rsqrt(k_o.transpose() @ k_o)[0]
        for i in ti.static(range(len(self.lights))):
            light = self.lights[i]
            sample = light.sample()
            k_i = sample.pos - x
            dist_sqr = k_i.transpose() @ k_i
            dist = ti.sqrt(dist_sqr)
            k_i = k_i / dist
            shadow_ray = Ray(o=x, d=k_i)
            hit_record = self.ray_hit_nearest(shadow_ray, 0.1, dist + 1)
            if flag_query(hit_record.flag, HIT) and hit_record.t > dist - 1e-3:
                radiance = light.radiance(-k_i)
                cos_light = (light.normal.transpose() @ (-k_i))[0]
                cos_x = (normal.transpose() @ (-k_i))[0] * ti.rsqrt(normal.transpose() @ normal)[0]
                result = result + brdf(material, k_i, k_o) * radiance * cos_x * cos_light / (dist_sqr * sample.prob)
        return result

    @ti.func
    def ray_hit_nearest(self, ray, t0, t1):
        result = HitRecord(flag=ZERO, t=t1, object_idx=-1, material=-1)
        for i in range(self.triangles.shape[0]):
            v0 = self.triangles[i][0, :].transpose()
            v1 = self.triangles[i][1, :].transpose()
            v2 = self.triangles[i][2, :].transpose()
            n = (v1 - v0).cross(v2 - v0)
            hit_record = ray_triangle_intersection(ray.o, ray.d, v0, v1, v2, n, t0, t1)
            if flag_query(hit_record.flag, HIT) == 1 and hit_record.t < result.t:
                result.flag = flag_set(result.flag, HIT)
                result.t = hit_record.t
                result.object_idx = i
        for i in ti.static(range(len(self.lights))):
            hit_record = self.lights[i].ray_hit(ray, t0, t1)
            if flag_query(hit_record.flag, HIT) == 1 and hit_record.t < result.t:
                result.flag = flag_set(result.flag, HIT | IS_LIGHT)
                result.t = hit_record.t
                result.object_idx = i
        return result


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
    """Simple area light"""

    def __init__(self, origin: Sequence, dir_x: Sequence, dir_y: Sequence, power: float, color: Sequence = (1, 1, 1)):
        """
        :param origin: 左下角
        :param dir_x: 宽
        :param dir_y: 高

        normal = dir_x cross dir_y
        """
        self.origin = ti.Vector(origin, dt=ti.f32)
        self.dir_x = ti.Vector(dir_x, dt=ti.f32)
        self.dir_y = ti.Vector(dir_y, dt=ti.f32)
        self.normal = ti.Vector(np.cross(dir_x, dir_y) / np.linalg.norm(np.cross(dir_x, dir_y)), dt=ti.f32)
        self.area: ti.float32 = np.linalg.norm(self.dir_x) * np.linalg.norm(self.dir_y)
        self.irradiance: ti.f32 = power / self.area
        self.color = ti.Vector(color, dt=ti.f32)

    @ti.func
    def ray_hit(self, ray, t0, t1):
        result = HitRecord(flag=ZERO, t=0, object_idx=-1, material=ZERO)
        o_diff = ray.o - self.origin
        M = ti.Matrix([
            [self.dir_x[0], self.dir_y[0], -ray.d[0]],
            [self.dir_x[1], self.dir_y[1], -ray.d[1]],
            [self.dir_x[2], self.dir_y[2], -ray.d[2]]])
        x = M.inverse() @ o_diff
        if 0 < x[0] < 1 and 0 < x[1] < 1 and t0 < x[2] < t1:
            result.flag = flag_set(result.flag, HIT | IS_LIGHT)
            result.t = x[2]
        return result

    @ti.func
    def sample(self):
        n0 = ti.random(ti.f32)
        n1 = ti.random(ti.f32)
        pos = self.origin + n0 * self.dir_x + n1 * self.dir_y
        prob = 1 / self.area
        result = PointSample(pos=pos, prob=prob)
        return result

    @ti.func
    def probability(self, x) -> ti.f32:
        return 1 / self.area

    @ti.func
    def radiance(self, direction):
        cos = (self.normal.transpose() @ direction)[0] * ti.rsqrt(direction.transpose() @ direction)[0]
        sin = 1 - cos * cos
        result: ti.f32 = 0
        if cos > 1e-3:
            result = self.irradiance / (2 * float(3.141593) * sin)
        return result


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
    hit_record = HitRecord(flag=HIT, t=0, object_idx=-1, material=-1)
    a = o - v0
    b = v1 - v0
    c = v2 - v0
    if ti.abs((a.transpose() @ n)[0]) < 1e-3:  # d与n平行, 我们认为没有交点
        hit_record.flag = flag_clear(hit_record.flag, HIT)
    if flag_query(hit_record.flag, HIT) == 1:
        M = ti.Matrix([
            [b[0], c[0], -d[0]],
            [b[1], c[1], -d[1]],
            [b[2], c[2], -d[2]]])
        x = M.inverse() @ a
        alpha = 1 - x[0] - x[1]
        if 0 < x[0] < 1 and 0 < x[1] < 1 and 0 < alpha < 1 and t0 < x[2] < t1:
            hit_record.t = x[2]
        else:
            hit_record.flag = flag_clear(hit_record.flag, HIT)
    return hit_record


@ti.func
def brdf(material: ti.i32, k_i, k_o):
    """假设半球在y正半轴, 底部紧贴xz平面"""
    return float(1 / 3.1415926)  # perfect diffusing


@ti.func
def flag_set(status: ti.i32, flag_type: ti.i32):
    return status | flag_type


@ti.func
def flag_clear(status: ti.i32, flag_type: ti.i32):
    return status & (~flag_type)


@ti.func
def flag_query(status: ti.i32, flag_type: ti.i32) -> ti.i32:
    result: ti.i32 = 0
    if status & flag_type == flag_type:
        result = 1
    return result
