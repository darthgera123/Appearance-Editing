import enoki as ek
import mitsuba
mitsuba.set_variant('gpu_rgb')

from mitsuba.core import Vector0f, Vector2f, Vector3f, Float, Float32, Float64, Thread, xml, Spectrum, depolarize, RayDifferential3f, Frame3f, warp, Bitmap, Struct, UInt64, UInt32
from mitsuba.core import math as m_math
from mitsuba.core.xml import load_string, load_file
from mitsuba.render import BSDF, Emitter, BSDFContext, BSDFSample3f, SurfaceInteraction3f, ImageBlock, register_integrator, register_bsdf, MonteCarloIntegrator, SamplingIntegrator, has_flag, BSDFFlags, DirectionSample3f

import numpy as np
import trimesh

def sph_dir(theta, phi):
    """ Map spherical to Euclidean coordinates """
    st, ct = ek.sincos(theta)
    sp, cp = ek.sincos(phi)
    return Vector3f(cp*st, sp*st, ct)

def sph_convert(v):
    """ Map Euclidean to spherical coordinates """
    x2 = ek.pow(v.x, 2)
    y2 = ek.pow(v.y, 2)
    z2 = ek.pow(v.z, 2)

    r = ek.sqrt(x2+y2+z2)
    phi = ek.atan2(v.y, v.x)
    theta = ek.atan2(ek.sqrt(x2+y2), v.z)

    return r, theta, phi

class TransformIntegrator(SamplingIntegrator):

    def __init__(self, props):
        SamplingIntegrator.__init__(self, props)

    def sample(self, scene, sampler, ray, medium=None, active=True):
        result = Vector3f(0.0)

        si = scene.ray_intersect(ray, active)
        active = si.is_valid() & active

        col1 = si.to_world( Vector3f(1.0, 0.0, 0.0) )
        col2 = si.to_world( Vector3f(0.0, 1.0, 0.0) )
        col3 = si.to_world( Vector3f(0.0, 0.0, 1.0) )

        result.x = ek.select(active, si.uv.x, Float(0.0))
        result.y = ek.select(active, si.uv.y, Float(0.0))
        result.z = ek.select(active, Float(1.0), Float(0.0))

        return result, si.is_valid(), [Float(col1.x), Float(col2.x), Float(col3.x),
                                        Float(col1.y), Float(col2.y), Float(col3.y),
                                        Float(col1.z), Float(col2.z), Float(col3.z)]

    def aov_names(self):
        return ['col1.x','col1.z','col1.z','col2.x','col2.y','col2.z','col3.x','col3.y','col3.z']

    def to_string(self):
        return "TransformIntegrator[]"