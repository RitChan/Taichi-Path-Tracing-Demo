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
                 triangles: "Triangles",
                 lights: "Lights",
                 camera: "Camera",
                 canvas: "Canvas"):
        # Basic Parameters:
        # (1) triangles: nx3 matrix where n%3 == 0
        # (2) triangle material types
        # (3) lights
        # (4) camera
        # (5) canvas
        self.triangles = triangles
        self.lights = lights
        self.camera = camera
        self.canvas = canvas

    @ti.kernel
    def render(self, iter_count: ti.i32):
        for i, j in self.canvas.buffer:
            iter_f = float(iter_count)
            i_f = float(i) + 0.5
            j_f = float(j) + 0.5
            rel_x = i_f / float(self.canvas.width)
            rel_y = j_f / float(self.canvas.height)
            view_ray = self.camera.ray_cast(rel_x, rel_y)
            hit_record = self.ray_hit_nearest(view_ray, 0.01, 1000.0)
            iter_f_inv = 1 / iter_f
            self.canvas.buffer[i, j] = self.canvas.buffer[i, j] * (iter_f - 1) * iter_f_inv
            if flag_query(hit_record.flag, HIT) == 1:
                # self.canvas.buffer[i, j] = ti.Vector([hit_record.t, hit_record.t, hit_record.t], ti.f32)
                L_s = ti.Vector([0, 0, 0], ti.f32)
                if flag_query(hit_record.flag, IS_LIGHT) == 1:
                    light_idx = hit_record.object_idx
                    L_s = self.lights.radiance(light_idx, -view_ray.d) * self.lights.get_vec3(light_idx, Lights.COLOR)
                else:
                    triangle_idx = hit_record.object_idx
                    x = view_ray.o + hit_record.t * view_ray.d
                    n = self.triangles.get_vec3(triangle_idx, Triangles.N)
                    L_s = self.direct_light_radiance_at(x, -view_ray.d, n, ZERO) * self.triangles.get_vec3(triangle_idx, Triangles.COLOR)
                self.canvas.buffer[i, j] = self.canvas.buffer[i, j] + iter_f_inv * L_s

    @ti.func
    def sample(self, x, k_o):
        pass

    @ti.func
    def direct_light_radiance_at(self, x, k_o, normal, material):
        result = float(0)
        k_o = k_o * ti.rsqrt(k_o.transpose() @ k_o)[0]
        for i in ti.static(range(self.lights.floats.shape[0])):
            sample = self.lights.sample(i)
            k_i = sample.pos - x
            dist_sqr = k_i.norm_sqr()
            dist = ti.sqrt(dist_sqr)
            k_i = k_i / dist
            shadow_ray = Ray(o=x, d=k_i)
            hit_record = self.ray_hit_nearest(shadow_ray, 0.01, dist + 1)
            if flag_query(hit_record.flag, HIT) == 1 and flag_query(hit_record.flag, IS_LIGHT) == 1 and hit_record.t > dist - 1e-3:
                radiance = self.lights.radiance(i, -k_i)
                cos_light = self.lights.get_vec3(i, Lights.NORMAL).dot(-k_i)
                cos_x = normal.dot(k_i) / normal.norm()
                L_f = brdf(material, k_i, k_o) * radiance * cos_x * cos_light / (dist_sqr * sample.prob)
                if L_f > 0:
                    result = result + L_f
        return result

    @ti.func
    def ray_hit_nearest(self, ray, t0, t1):
        result = HitRecord(flag=ZERO, t=t1, object_idx=-1, material=-1)
        for i in range(self.triangles.floats.shape[0]):
            v0 = self.triangles.get_vec3(i, Triangles.V0)
            v1 = self.triangles.get_vec3(i, Triangles.V1)
            v2 = self.triangles.get_vec3(i, Triangles.V2)
            n = self.triangles.get_vec3(i, Triangles.N)
            if ray.d.dot(n) < 0:
                hit_record = ray_triangle_intersection(ray.o, ray.d, v0, v1, v2, n, t0, t1)
                if flag_query(hit_record.flag, HIT) == 1 and hit_record.t < result.t:
                    result.flag = flag_set(result.flag, HIT)
                    result.t = hit_record.t
                    result.object_idx = i
        for i in ti.static(range(self.lights.floats.shape[0])):
            if ray.d.dot(self.lights.get_vec3(i, Lights.NORMAL)) < 0:
                hit_record = self.lights.ray_hit_light(i, ray, t0, t1)
                if flag_query(hit_record.flag, HIT) == 1 and hit_record.t < result.t:
                    result.flag = flag_set(result.flag, HIT | IS_LIGHT)
                    result.t = hit_record.t
                    result.object_idx = i
        return result


@ti.data_oriented
class Triangles:
    # floats
    V0 = 0
    V1 = 3
    V2 = 6
    N = 9
    COLOR = 12
    # integers
    MATERIAL = 0

    def __init__(self, floats: Sequence, integers: Sequence = None):
        floats = np.asarray(floats, dtype="f4").reshape((-1, 15))
        self.floats = ti.field(ti.f32, shape=floats.shape)
        self.floats.from_numpy(floats)
        if integers is None:
            integers = np.zeros(self.floats.shape[0], dtype="i4")
        integers = np.asarray(integers, dtype="i4").reshape(-1)
        self.integers = ti.field(ti.i32, shape=integers.shape)
        self.integers.from_numpy(integers)

    @staticmethod
    def create(v0, v1, v2, color=(1, 1, 1)):
        v0 = np.asarray(v0, dtype="f4")
        v1 = np.asarray(v1, dtype="f4")
        v2 = np.asarray(v2, dtype="f4")
        n = np.cross(v1 - v0, v2 - v0)
        n = n / np.linalg.norm(n)
        color = np.asarray(color, dtype="f4")
        return np.hstack((v0, v1, v2, n, color))

    @ti.func
    def get_vec3(self, triangle_idx: ti.i32, base: ti.i32) -> Vector3f:
        return ti.Vector([self.floats[triangle_idx, base], self.floats[triangle_idx, base + 1], self.floats[triangle_idx, base + 2]], dt=ti.f32)


@ti.data_oriented
class Lights:
    ORIGIN = 0
    DIR_X = 3
    DIR_Y = 6
    NORMAL = 9
    COLOR = 12
    AREA = 15
    IRRADIANCE = 16

    def __init__(self, float_properties: Sequence):
        float_properties = np.asarray(float_properties, dtype="f4")
        float_properties = float_properties.reshape((-1, 17))
        self.floats = ti.field(ti.f32, shape=float_properties.shape)
        self.floats.from_numpy(float_properties)

    @staticmethod
    def create(origin, dir_x, dir_y, power, color) -> np.ndarray:
        origin = np.asarray(origin, dtype="f4")
        dir_x = np.asarray(dir_x, dtype="f4")
        dir_y = np.asarray(dir_y, dtype="f4")
        normal = np.cross(dir_x, dir_y)
        area = np.linalg.norm(normal)
        normal /= area
        irradiance = power / area
        color = np.asarray(color, dtype="f4")
        return np.hstack([origin, dir_x, dir_y, normal, color, area, irradiance])

    @ti.func
    def ray_hit_light(self, light_idx: ti.i32, ray, t0: ti.f32, t1: ti.f32) -> HitRecord:
        result = HitRecord(flag=ZERO, t=0, object_idx=-1, material=ZERO)
        origin = self.get_vec3(light_idx, Lights.ORIGIN)
        dir_y = self.get_vec3(light_idx, Lights.DIR_Y)
        dir_x = self.get_vec3(light_idx, Lights.DIR_X)
        o_diff = ray.o - origin
        M = ti.Matrix.cols([dir_x, dir_y, -ray.d])
        x = M.inverse() @ o_diff
        if 0 < x[0] < 1 and 0 < x[1] < 1 and t0 < x[2] < t1:
            result.flag = flag_set(result.flag, HIT | IS_LIGHT)
            result.t = x[2]
        return result

    @ti.func
    def sample(self, light_idx: ti.i32) -> PointSample:
        origin = self.get_vec3(light_idx, Lights.ORIGIN)
        dir_y = self.get_vec3(light_idx, Lights.DIR_Y)
        dir_x = self.get_vec3(light_idx, Lights.DIR_X)
        n0 = ti.random(ti.f32)
        n1 = ti.random(ti.f32)
        pos = origin + n0 * dir_x + n1 * dir_y
        prob = 1 / self.floats[light_idx, Lights.AREA]
        result = PointSample(pos=pos, prob=prob)
        return result

    @ti.func
    def probability(self, light_idx: ti.i32, x) -> ti.f32:
        return 1 / self.area

    @ti.func
    def radiance(self, light_idx: ti.i32, direction):
        normal = self.get_vec3(light_idx, Lights.NORMAL)
        irradiance = self.floats[light_idx, Lights.IRRADIANCE]
        cos = normal.dot(direction) * ti.rsqrt(normal.norm_sqr())
        result: ti.f32 = 0
        if cos > 0:
            result = irradiance / float(3.141593)
        return result

    @ti.func
    def get_vec3(self, light_idx: ti.i32, base: ti.i32) -> Vector3f:
        return ti.Vector([self.floats[light_idx, base], self.floats[light_idx, base + 1], self.floats[light_idx, base + 2]])


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
