import enoki as ek
import mitsuba
mitsuba.set_variant('gpu_rgb')

from mitsuba.core import Vector0f, Vector2f, Vector3f, Float, Float32, Float64, Thread, xml, Spectrum, depolarize, RayDifferential3f, Frame3f, warp, Bitmap, Struct, UInt64, UInt32
from mitsuba.core import math as m_math
from mitsuba.core.xml import load_string, load_file
from mitsuba.render import BSDF, Emitter, BSDFContext, BSDFSample3f, SurfaceInteraction3f, ImageBlock, register_integrator, register_bsdf, MonteCarloIntegrator, SamplingIntegrator, has_flag, BSDFFlags, DirectionSample3f

import numpy as np


def y_0_0_(colors, theta, phi):
    x = ek.sin(theta) * ek.cos(phi)
    y = ek.sin(theta) * ek.sin(phi)
    z = ek.cos(theta)

    K = ek.sqrt( 1.0/(ek.pi) ) * 1.0/2.0
    return K * colors

#-----------------------------------------------------------------------------------------------

def y_1_n1_(colors, theta, phi):
    x = ek.sin(theta) * ek.cos(phi)
    y = ek.sin(theta) * ek.sin(phi)
    z = ek.cos(theta)

    K = ek.sqrt( 3.0/(4*ek.pi) )
    return K * y * colors

def y_1_0_(colors, theta, phi):
    x = ek.sin(theta) * ek.cos(phi)
    y = ek.sin(theta) * ek.sin(phi)
    z = ek.cos(theta)

    K = ek.sqrt( 3.0/(4*ek.pi) )
    return K * z * colors

def y_1_p1_(colors, theta, phi):
    x = ek.sin(theta) * ek.cos(phi)
    y = ek.sin(theta) * ek.sin(phi)
    z = ek.cos(theta)

    K = ek.sqrt( 3.0/(4*ek.pi) )
    return K * x * colors

#-----------------------------------------------------------------------------------------------

def y_2_n2_(colors, theta, phi):
    x = ek.sin(theta) * ek.cos(phi)
    y = ek.sin(theta) * ek.sin(phi)
    z = ek.cos(theta)

    K = ek.sqrt( 15.0/(ek.pi) ) * 1.0/2.0
    return K * x * y * colors

def y_2_n1_(colors, theta, phi):
    x = ek.sin(theta) * ek.cos(phi)
    y = ek.sin(theta) * ek.sin(phi)
    z = ek.cos(theta)

    K = ek.sqrt( 15.0/(ek.pi) ) * 1.0/2.0
    return K * y * z * colors

def y_2_0_(colors, theta, phi):
    x = ek.sin(theta) * ek.cos(phi)
    y = ek.sin(theta) * ek.sin(phi)
    z = ek.cos(theta)

    K = ek.sqrt( 5.0/(16*ek.pi) )
    return K * ( -ek.pow(x, 2)-ek.pow(y, 2)+2*ek.pow(z, 2) ) * colors

def y_2_p1_(colors, theta, phi):
    x = ek.sin(theta) * ek.cos(phi)
    y = ek.sin(theta) * ek.sin(phi)
    z = ek.cos(theta)

    K = ek.sqrt( 15.0/(4*ek.pi) )
    return K * z * x * colors

def y_2_p2_(colors, theta, phi):
    x = ek.sin(theta) * ek.cos(phi)
    y = ek.sin(theta) * ek.sin(phi)
    z = ek.cos(theta)

    K = ek.sqrt( 15.0/(16*ek.pi) )
    return K * (ek.pow(x, 2)-ek.pow(y, 2)) * colors

#-----------------------------------------------------------------------------------------------

def y_3_n3_(colors, theta, phi):
    x = ek.sin(theta) * ek.cos(phi)
    y = ek.sin(theta) * ek.sin(phi)
    z = ek.cos(theta)

    K = ek.sqrt( 35.0/(2*ek.pi) ) * 1.0/4.0
    return K * y * ( 3*ek.pow(x, 2)-ek.pow(y, 2) ) * colors

def y_3_n2_(colors, theta, phi):
    x = ek.sin(theta) * ek.cos(phi)
    y = ek.sin(theta) * ek.sin(phi)
    z = ek.cos(theta)

    K = ek.sqrt( 105.0/(ek.pi) ) * 1.0/2.0
    return K * x * y * z * colors

def y_3_n1_(colors, theta, phi):
    x = ek.sin(theta) * ek.cos(phi)
    y = ek.sin(theta) * ek.sin(phi)
    z = ek.cos(theta)

    K = ek.sqrt( 21.0/(2*ek.pi) ) * 1.0/4.0
    return K * y * ( 4*ek.pow(z, 2)-ek.pow(x, 2)-ek.pow(y, 2) ) * colors

def y_3_0_(colors, theta, phi):
    x = ek.sin(theta) * ek.cos(phi)
    y = ek.sin(theta) * ek.sin(phi)
    z = ek.cos(theta)

    K = ek.sqrt( 7.0/(ek.pi) ) * 1.0/4.0
    return K * z * ( 2*ek.pow(z, 2)-3*ek.pow(x, 2)-3*ek.pow(y, 2) ) * colors

def y_3_p1_(colors, theta, phi):
    x = ek.sin(theta) * ek.cos(phi)
    y = ek.sin(theta) * ek.sin(phi)
    z = ek.cos(theta)

    K = ek.sqrt( 21.0/(2*ek.pi) ) * 1.0/4.0
    return K * x * ( 4*ek.pow(z, 2)-ek.pow(x, 2)-ek.pow(y, 2) ) * colors

def y_3_p2_(colors, theta, phi):
    x = ek.sin(theta) * ek.cos(phi)
    y = ek.sin(theta) * ek.sin(phi)
    z = ek.cos(theta)

    K = ek.sqrt( 105.0/(ek.pi) ) * 1.0/4.0
    return K * z * (ek.pow(x, 2)-ek.pow(y, 2)) * colors

def y_3_p3_(colors, theta, phi):
    x = ek.sin(theta) * ek.cos(phi)
    y = ek.sin(theta) * ek.sin(phi)
    z = ek.cos(theta)

    K = ek.sqrt( 35.0/(2*ek.pi) ) * 1.0/4.0
    return K * x * (ek.pow(x, 2)-3*ek.pow(y, 2)) * colors

#-----------------------------------------------------------------------------------------------

def y_4_n4_(colors, theta, phi):
    x = ek.sin(theta) * ek.cos(phi)
    y = ek.sin(theta) * ek.sin(phi)
    z = ek.cos(theta)

    K = ek.sqrt( 35.0/(ek.pi) ) * 3.0/4.0
    return K * x * y * ( ek.pow(x, 2)-ek.pow(y, 2) ) * colors

def y_4_n3_(colors, theta, phi):
    x = ek.sin(theta) * ek.cos(phi)
    y = ek.sin(theta) * ek.sin(phi)
    z = ek.cos(theta)

    K = ek.sqrt( 35.0/(2*ek.pi) ) * 3.0/4.0
    return K * y * z * ( 3*ek.pow(x, 2)-ek.pow(y, 2) ) * colors

def y_4_n2_(colors, theta, phi):
    x = ek.sin(theta) * ek.cos(phi)
    y = ek.sin(theta) * ek.sin(phi)
    z = ek.cos(theta)

    K = ek.sqrt( 5.0/(ek.pi) ) * 3.0/4.0
    return K * x * y * ( 7*ek.pow(z, 2)-1 ) * colors

def y_4_n1_(colors, theta, phi):
    x = ek.sin(theta) * ek.cos(phi)
    y = ek.sin(theta) * ek.sin(phi)
    z = ek.cos(theta)

    K = ek.sqrt( 5.0/(2*ek.pi) ) * 3.0/4.0
    return K * y * z * ( 7*ek.pow(z, 2)-3 ) * colors

def y_4_0_(colors, theta, phi):
    x = ek.sin(theta) * ek.cos(phi)
    y = ek.sin(theta) * ek.sin(phi)
    z = ek.cos(theta)

    K = ek.sqrt( 1.0/(ek.pi) ) * 3.0/16.0
    return K * ( 35*ek.pow(z, 4)-30*ek.pow(z, 2)+3 ) * colors

def y_4_p1_(colors, theta, phi):
    x = ek.sin(theta) * ek.cos(phi)
    y = ek.sin(theta) * ek.sin(phi)
    z = ek.cos(theta)

    K = ek.sqrt( 5.0/(2*ek.pi) ) * 3.0/4.0
    return K * x * z * ( 7*ek.pow(z, 2)-3 ) * colors

def y_4_p2_(colors, theta, phi):
    x = ek.sin(theta) * ek.cos(phi)
    y = ek.sin(theta) * ek.sin(phi)
    z = ek.cos(theta)

    K = ek.sqrt( 5.0/(ek.pi) ) * 3.0/8.0
    return K * ( ek.pow(x, 2)-ek.pow(y, 2) ) * ( 7*ek.pow(z, 2)-1 ) * colors

def y_4_p3_(colors, theta, phi):
    x = ek.sin(theta) * ek.cos(phi)
    y = ek.sin(theta) * ek.sin(phi)
    z = ek.cos(theta)

    K = ek.sqrt( 35.0/(2*ek.pi) ) * 3.0/4.0
    return K * x * z * ( ek.pow(x, 2)-3*ek.pow(y, 2) ) * colors

def y_4_p4_(colors, theta, phi):
    x = ek.sin(theta) * ek.cos(phi)
    y = ek.sin(theta) * ek.sin(phi)
    z = ek.cos(theta)

    K = ek.sqrt( 35.0/(ek.pi) ) * 3.0/16.0
    return K * ( ek.pow(x, 2)*( ek.pow(x, 2)-3*ek.pow(y, 2) ) - ek.pow(y, 2)*( 3*ek.pow(x, 2)-ek.pow(y, 2) ) ) * colors

#-----------------------------------------------------------------------------------------------

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

class EnvIntegrator(SamplingIntegrator):
    # Use 1 spp with this integrator

    def __init__(self, props):
        SamplingIntegrator.__init__(self, props)

    def sample(self, scene, sampler, ray, medium=None, active=True):
        result = Vector3f(0.0)
        si = scene.ray_intersect(ray, active)
        active = si.is_valid() & active

        bsdf = si.bsdf(ray)
        bsdf_ = bsdf.numpy()

        res = 10
        theta_o, phi_o = ek.meshgrid(
            ek.linspace(Float, 0,     ek.pi,     res),
            ek.linspace(Float, 0, 2 * ek.pi, 2 * res)
        )
        wo = sph_dir(theta_o, phi_o)
        wo_numpy = wo.numpy()
        num_samples = wo_numpy.shape[0]

        y_0_0 = Vector3f(0.0)

        y_1_n1 = Vector3f(0.0)
        y_1_0 = Vector3f(0.0)
        y_1_p1 = Vector3f(0.0)

        y_2_n2 = Vector3f(0.0)
        y_2_n1 = Vector3f(0.0)
        y_2_0 = Vector3f(0.0)
        y_2_p1 = Vector3f(0.0)
        y_2_p2 = Vector3f(0.0)

        y_3_n3 = Vector3f(0.0)
        y_3_n2 = Vector3f(0.0)
        y_3_n1 = Vector3f(0.0)
        y_3_0 = Vector3f(0.0)
        y_3_p1 = Vector3f(0.0)
        y_3_p2 = Vector3f(0.0)
        y_3_p3 = Vector3f(0.0)

        y_4_n4 = Vector3f(0.0)
        y_4_n3 = Vector3f(0.0)
        y_4_n2 = Vector3f(0.0)
        y_4_n1 = Vector3f(0.0)
        y_4_0 = Vector3f(0.0)
        y_4_p1 = Vector3f(0.0)
        y_4_p2 = Vector3f(0.0)
        y_4_p3 = Vector3f(0.0)
        y_4_p4 = Vector3f(0.0)

        for wo_ in wo_numpy:
            ctx = BSDFContext()

            wo_ = Vector3f( np.array([wo_]).repeat(bsdf_.shape[0], axis=0) )

            wo_world = si.to_world(wo_)
            _, theta_, phi_ = sph_convert(wo_world)

            si_new = scene.ray_intersect(si.spawn_ray(wo_world), active)
            emitter = si_new.emitter(scene, active)
            # active &= ek.neq(emitter, 0)
            emitter_val = Emitter.eval_vec(emitter, si_new, active)

            y_0_0 += y_0_0_(emitter_val, theta_, phi_)

            y_1_n1 += y_1_n1_(emitter_val, theta_, phi_)
            y_1_0 += y_1_0_(emitter_val, theta_, phi_)
            y_1_p1 += y_1_p1_(emitter_val, theta_, phi_)
            
            y_2_n2 += y_2_n2_(emitter_val, theta_, phi_)
            y_2_n1 += y_2_n1_(emitter_val, theta_, phi_)
            y_2_0 += y_2_0_(emitter_val, theta_, phi_)
            y_2_p1 += y_2_p1_(emitter_val, theta_, phi_)
            y_2_p2 += y_2_p2_(emitter_val, theta_, phi_)

            y_3_n3 += y_3_n3_(emitter_val, theta_, phi_)
            y_3_n2 += y_3_n2_(emitter_val, theta_, phi_)
            y_3_n1 += y_3_n1_(emitter_val, theta_, phi_)
            y_3_0 += y_3_0_(emitter_val, theta_, phi_)
            y_3_p1 += y_3_p1_(emitter_val, theta_, phi_)
            y_3_p2 += y_3_p2_(emitter_val, theta_, phi_)
            y_3_p3 += y_3_p3_(emitter_val, theta_, phi_)

            y_4_n4 += y_4_n4_(emitter_val, theta_, phi_)
            y_4_n3 += y_4_n3_(emitter_val, theta_, phi_)
            y_4_n2 += y_4_n2_(emitter_val, theta_, phi_)
            y_4_n1 += y_4_n1_(emitter_val, theta_, phi_)
            y_4_0 += y_4_0_(emitter_val, theta_, phi_)
            y_4_p1 += y_4_p1_(emitter_val, theta_, phi_)
            y_4_p2 += y_4_p2_(emitter_val, theta_, phi_)
            y_4_p3 += y_4_p3_(emitter_val, theta_, phi_)
            y_4_p4 += y_4_p4_(emitter_val, theta_, phi_)

        y_0_0 = y_0_0 * 4 * ek.pi / num_samples

        y_1_n1 = y_1_n1 * 4 * ek.pi / num_samples
        y_1_0 = y_1_0 * 4 * ek.pi / num_samples
        y_1_p1 = y_1_p1 * 4 * ek.pi / num_samples 

        y_2_n2 = y_2_n2 * 4 * ek.pi / num_samples
        y_2_n1 = y_2_n1 * 4 * ek.pi / num_samples
        y_2_0 = y_2_0 * 4 * ek.pi / num_samples
        y_2_p1 = y_2_p1 * 4 * ek.pi / num_samples
        y_2_p2 = y_2_p2 * 4 * ek.pi / num_samples

        y_3_n3 = y_3_n3 * 4 * ek.pi / num_samples
        y_3_n2 = y_3_n2 * 4 * ek.pi / num_samples
        y_3_n1 = y_3_n1 * 4 * ek.pi / num_samples
        y_3_0 = y_3_0 * 4 * ek.pi / num_samples
        y_3_p1 = y_3_p1 * 4 * ek.pi / num_samples
        y_3_p2 = y_3_p2 * 4 * ek.pi / num_samples
        y_3_p3 = y_3_p3 * 4 * ek.pi / num_samples

        y_4_n4 = y_4_n4 * 4 * ek.pi / num_samples
        y_4_n3 = y_4_n3 * 4 * ek.pi / num_samples
        y_4_n2 = y_4_n2 * 4 * ek.pi / num_samples
        y_4_n1 = y_4_n1 * 4 * ek.pi / num_samples
        y_4_0 = y_4_0 * 4 * ek.pi / num_samples
        y_4_p1 = y_4_p1 * 4 * ek.pi / num_samples
        y_4_p2 = y_4_p2 * 4 * ek.pi / num_samples
        y_4_p3 = y_4_p3 * 4 * ek.pi / num_samples
        y_4_p4 = y_4_p4 * 4 * ek.pi / num_samples

        return result, si.is_valid(), [ Float(y_0_0[0]), Float(y_0_0[1]), Float(y_0_0[2]),\
                                        Float(y_1_n1[0]), Float(y_1_n1[1]), Float(y_1_n1[2]),\
                                        Float(y_1_0[0]), Float(y_1_0[1]), Float(y_1_0[2]),\
                                        Float(y_1_p1[0]), Float(y_1_p1[1]), Float(y_1_p1[2]),\
                                        Float(y_2_n2[0]), Float(y_2_n2[1]), Float(y_2_n2[2]),\
                                        Float(y_2_n1[0]), Float(y_2_n1[1]), Float(y_2_n1[2]),\
                                        Float(y_2_0[0]), Float(y_2_0[1]), Float(y_2_0[2]),\
                                        Float(y_2_p1[0]), Float(y_2_p1[1]), Float(y_2_p1[2]),\
                                        Float(y_2_p2[0]), Float(y_2_p2[1]), Float(y_2_p2[2]),\
                                        Float(y_3_n3[0]), Float(y_3_n3[1]), Float(y_3_n3[2]),\
                                        Float(y_3_n2[0]), Float(y_3_n2[1]), Float(y_3_n2[2]),\
                                        Float(y_3_n1[0]), Float(y_3_n1[1]), Float(y_3_n1[2]),\
                                        Float(y_3_0[0]), Float(y_3_0[1]), Float(y_3_0[2]),\
                                        Float(y_3_p1[0]), Float(y_3_p1[1]), Float(y_3_p1[2]),\
                                        Float(y_3_p2[0]), Float(y_3_p2[1]), Float(y_3_p2[2]),\
                                        Float(y_3_p3[0]), Float(y_3_p3[1]), Float(y_3_p3[2]),\
                                        Float(y_4_n4[0]), Float(y_4_n4[1]), Float(y_4_n4[2]),\
                                        Float(y_4_n3[0]), Float(y_4_n3[1]), Float(y_4_n3[2]),\
                                        Float(y_4_n2[0]), Float(y_4_n2[1]), Float(y_4_n2[2]),\
                                        Float(y_4_n1[0]), Float(y_4_n1[1]), Float(y_4_n1[2]),\
                                        Float(y_4_0[0]), Float(y_4_0[1]), Float(y_4_0[2]),\
                                        Float(y_4_p1[0]), Float(y_4_p1[1]), Float(y_4_p1[2]),\
                                        Float(y_4_p2[0]), Float(y_4_p2[1]), Float(y_4_p2[2]),\
                                        Float(y_4_p3[0]), Float(y_4_p3[1]), Float(y_4_p3[2]),\
                                        Float(y_4_p4[0]), Float(y_4_p4[1]), Float(y_4_p4[2])]

        # return result, si.is_valid(), [ Float(y_0_0[0]), Float(y_0_0[1]), Float(y_0_0[2]),\
        #                                 Float(y_1_n1[0]), Float(y_1_n1[1]), Float(y_1_n1[2]),\
        #                                 Float(y_1_0[0]), Float(y_1_0[1]), Float(y_1_0[2]),\
        #                                 Float(y_1_p1[0]), Float(y_1_p1[1]), Float(y_1_p1[2]),\
        #                                 Float(y_2_n2[0]), Float(y_2_n2[1]), Float(y_2_n2[2]),\
        #                                 Float(y_2_n1[0]), Float(y_2_n1[1]), Float(y_2_n1[2]),\
        #                                 Float(y_2_0[0]), Float(y_2_0[1]), Float(y_2_0[2]),\
        #                                 Float(y_2_p1[0]), Float(y_2_p1[1]), Float(y_2_p1[2]),\
        #                                 Float(y_2_p2[0]), Float(y_2_p2[1]), Float(y_2_p2[2]),\
        #                                 Float(y_3_n3[0]), Float(y_3_n3[1]), Float(y_3_n3[2]),\
        #                                 Float(y_3_n2[0]), Float(y_3_n2[1]), Float(y_3_n2[2]),\
        #                                 Float(y_3_n1[0]), Float(y_3_n1[1]), Float(y_3_n1[2]),\
        #                                 Float(y_3_0[0]), Float(y_3_0[1]), Float(y_3_0[2]),\
        #                                 Float(y_3_p1[0]), Float(y_3_p1[1]), Float(y_3_p1[2]),\
        #                                 Float(y_3_p2[0]), Float(y_3_p2[1]), Float(y_3_p2[2]),\
        #                                 Float(y_3_p3[0]), Float(y_3_p3[1]), Float(y_3_p3[2])]

    def aov_names(self):
        names = []
        for i in range(0, 25):
        # for i in range(0, 16):
            for c in ['r', 'g', 'b']:
                names.append('sh_%s_%d' % (c, i))
        
        return names

    def to_string(self):
        return "EnvIntegrator[]"

class AuxIntegrator(SamplingIntegrator):
    # Use 1 spp with this integrator

    def __init__(self, props):
        SamplingIntegrator.__init__(self, props)

    def sample(self, scene, sampler, ray, medium=None, active=True):
        result = Vector3f(0.0)
        si = scene.ray_intersect(ray, active)

        emitter_vis = si.emitter(scene, active)
        emitter_val = ek.select(active, Emitter.eval_vec(emitter_vis, si, active), Vector3f(0.0))

        active = si.is_valid() & active

        bsdf = si.bsdf(ray)
        bsdf_ = bsdf.numpy()

        res = 10
        theta_o, phi_o = ek.meshgrid(
            ek.linspace(Float, 0,     ek.pi,     res),
            ek.linspace(Float, 0, 2 * ek.pi, 2 * res)
        )
        wo = sph_dir(theta_o, phi_o)
        wo_numpy = wo.numpy()
        num_samples = wo_numpy.shape[0]

        y_0_0 = Vector3f(0.0)

        y_1_n1 = Vector3f(0.0)
        y_1_0 = Vector3f(0.0)
        y_1_p1 = Vector3f(0.0)

        y_2_n2 = Vector3f(0.0)
        y_2_n1 = Vector3f(0.0)
        y_2_0 = Vector3f(0.0)
        y_2_p1 = Vector3f(0.0)
        y_2_p2 = Vector3f(0.0)

        # y_3_n3 = Vector3f(0.0)
        # y_3_n2 = Vector3f(0.0)
        # y_3_n1 = Vector3f(0.0)
        # y_3_0 = Vector3f(0.0)
        # y_3_p1 = Vector3f(0.0)
        # y_3_p2 = Vector3f(0.0)
        # y_3_p3 = Vector3f(0.0)

        # y_4_n4 = Vector3f(0.0)
        # y_4_n3 = Vector3f(0.0)
        # y_4_n2 = Vector3f(0.0)
        # y_4_n1 = Vector3f(0.0)
        # y_4_0 = Vector3f(0.0)
        # y_4_p1 = Vector3f(0.0)
        # y_4_p2 = Vector3f(0.0)
        # y_4_p3 = Vector3f(0.0)
        # y_4_p4 = Vector3f(0.0)

        for wo_ in wo_numpy:
            ctx = BSDFContext()

            wo_ = Vector3f( np.array([wo_]).repeat(bsdf_.shape[0], axis=0) )

            wo_world = si.to_world(wo_)
            _, theta_, phi_ = sph_convert(wo_world)

            bsdf_val = BSDF.eval_vec(bsdf, ctx, si, wo_)

            y_0_0 += y_0_0_(bsdf_val, theta_, phi_)

            y_1_n1 += y_1_n1_(bsdf_val, theta_, phi_)
            y_1_0 += y_1_0_(bsdf_val, theta_, phi_)
            y_1_p1 += y_1_p1_(bsdf_val, theta_, phi_)
            
            y_2_n2 += y_2_n2_(bsdf_val, theta_, phi_)
            y_2_n1 += y_2_n1_(bsdf_val, theta_, phi_)
            y_2_0 += y_2_0_(bsdf_val, theta_, phi_)
            y_2_p1 += y_2_p1_(bsdf_val, theta_, phi_)
            y_2_p2 += y_2_p2_(bsdf_val, theta_, phi_)

            # y_3_n3 += y_3_n3_(bsdf_val, theta_, phi_)
            # y_3_n2 += y_3_n2_(bsdf_val, theta_, phi_)
            # y_3_n1 += y_3_n1_(bsdf_val, theta_, phi_)
            # y_3_0 += y_3_0_(bsdf_val, theta_, phi_)
            # y_3_p1 += y_3_p1_(bsdf_val, theta_, phi_)
            # y_3_p2 += y_3_p2_(bsdf_val, theta_, phi_)
            # y_3_p3 += y_3_p3_(bsdf_val, theta_, phi_)

            # y_4_n4 += y_4_n4_(bsdf_val, theta_, phi_)
            # y_4_n3 += y_4_n3_(bsdf_val, theta_, phi_)
            # y_4_n2 += y_4_n2_(bsdf_val, theta_, phi_)
            # y_4_n1 += y_4_n1_(bsdf_val, theta_, phi_)
            # y_4_0 += y_4_0_(bsdf_val, theta_, phi_)
            # y_4_p1 += y_4_p1_(bsdf_val, theta_, phi_)
            # y_4_p2 += y_4_p2_(bsdf_val, theta_, phi_)
            # y_4_p3 += y_4_p3_(bsdf_val, theta_, phi_)
            # y_4_p4 += y_4_p4_(bsdf_val, theta_, phi_)
        
        y_0_0 = y_0_0 * 4 * ek.pi / num_samples

        y_1_n1 = y_1_n1 * 4 * ek.pi / num_samples
        y_1_0 = y_1_0 * 4 * ek.pi / num_samples
        y_1_p1 = y_1_p1 * 4 * ek.pi / num_samples 

        y_2_n2 = y_2_n2 * 4 * ek.pi / num_samples
        y_2_n1 = y_2_n1 * 4 * ek.pi / num_samples
        y_2_0 = y_2_0 * 4 * ek.pi / num_samples
        y_2_p1 = y_2_p1 * 4 * ek.pi / num_samples
        y_2_p2 = y_2_p2 * 4 * ek.pi / num_samples

        # y_3_n3 = y_3_n3 * 4 * ek.pi / num_samples
        # y_3_n2 = y_3_n2 * 4 * ek.pi / num_samples
        # y_3_n1 = y_3_n1 * 4 * ek.pi / num_samples
        # y_3_0 = y_3_0 * 4 * ek.pi / num_samples
        # y_3_p1 = y_3_p1 * 4 * ek.pi / num_samples
        # y_3_p2 = y_3_p2 * 4 * ek.pi / num_samples
        # y_3_p3 = y_3_p3 * 4 * ek.pi / num_samples

        # y_4_n4 = y_4_n4 * 4 * ek.pi / num_samples
        # y_4_n3 = y_4_n3 * 4 * ek.pi / num_samples
        # y_4_n2 = y_4_n2 * 4 * ek.pi / num_samples
        # y_4_n1 = y_4_n1 * 4 * ek.pi / num_samples
        # y_4_0 = y_4_0 * 4 * ek.pi / num_samples
        # y_4_p1 = y_4_p1 * 4 * ek.pi / num_samples
        # y_4_p2 = y_4_p2 * 4 * ek.pi / num_samples
        # y_4_p3 = y_4_p3 * 4 * ek.pi / num_samples
        # y_4_p4 = y_4_p4 * 4 * ek.pi / num_samples

        y_0_0 = ek.select(active, y_0_0, Vector3f(1.0))

        y_1_n1 = ek.select(active, y_1_n1, Vector3f(1.0))
        y_1_0 = ek.select(active, y_1_0, Vector3f(1.0))
        y_1_p1 = ek.select(active, y_1_p1, Vector3f(1.0)) 

        y_2_n2 = ek.select(active, y_2_n2, Vector3f(1.0))
        y_2_n1 = ek.select(active, y_2_n1, Vector3f(1.0))
        y_2_0 = ek.select(active, y_2_0, Vector3f(1.0))
        y_2_p1 = ek.select(active, y_2_p1, Vector3f(1.0))
        y_2_p2 = ek.select(active, y_2_p2, Vector3f(1.0))

        result.x = ek.select(active, si.uv.x, Float(0.0))
        result.y = ek.select(active, si.uv.y, Float(0.0))
        result.z = ek.select(active, Float(1.0), Float(0.0))

        return result, si.is_valid(), [ Float(emitter_val[0]), Float(emitter_val[1]), Float(emitter_val[2]),\
                                        Float(y_0_0[0]), Float(y_0_0[1]), Float(y_0_0[2]),\
                                        Float(y_1_n1[0]), Float(y_1_n1[1]), Float(y_1_n1[2]),\
                                        Float(y_1_0[0]), Float(y_1_0[1]), Float(y_1_0[2]),\
                                        Float(y_1_p1[0]), Float(y_1_p1[1]), Float(y_1_p1[2]),\
                                        Float(y_2_n2[0]), Float(y_2_n2[1]), Float(y_2_n2[2]),\
                                        Float(y_2_n1[0]), Float(y_2_n1[1]), Float(y_2_n1[2]),\
                                        Float(y_2_0[0]), Float(y_2_0[1]), Float(y_2_0[2]),\
                                        Float(y_2_p1[0]), Float(y_2_p1[1]), Float(y_2_p1[2]),\
                                        Float(y_2_p2[0]), Float(y_2_p2[1]), Float(y_2_p2[2])]

        # return result, si.is_valid(), [ Float(y_0_0[0]), Float(y_0_0[1]), Float(y_0_0[2]),\
        #                                 Float(y_1_n1[0]), Float(y_1_n1[1]), Float(y_1_n1[2]),\
        #                                 Float(y_1_0[0]), Float(y_1_0[1]), Float(y_1_0[2]),\
        #                                 Float(y_1_p1[0]), Float(y_1_p1[1]), Float(y_1_p1[2]),\
        #                                 Float(y_2_n2[0]), Float(y_2_n2[1]), Float(y_2_n2[2]),\
        #                                 Float(y_2_n1[0]), Float(y_2_n1[1]), Float(y_2_n1[2]),\
        #                                 Float(y_2_0[0]), Float(y_2_0[1]), Float(y_2_0[2]),\
        #                                 Float(y_2_p1[0]), Float(y_2_p1[1]), Float(y_2_p1[2]),\
        #                                 Float(y_2_p2[0]), Float(y_2_p2[1]), Float(y_2_p2[2]),\
        #                                 Float(y_3_n3[0]), Float(y_3_n3[1]), Float(y_3_n3[2]),\
        #                                 Float(y_3_n2[0]), Float(y_3_n2[1]), Float(y_3_n2[2]),\
        #                                 Float(y_3_n1[0]), Float(y_3_n1[1]), Float(y_3_n1[2]),\
        #                                 Float(y_3_0[0]), Float(y_3_0[1]), Float(y_3_0[2]),\
        #                                 Float(y_3_p1[0]), Float(y_3_p1[1]), Float(y_3_p1[2]),\
        #                                 Float(y_3_p2[0]), Float(y_3_p2[1]), Float(y_3_p2[2]),\
        #                                 Float(y_3_p3[0]), Float(y_3_p3[1]), Float(y_3_p3[2])]

    def aov_names(self):
        names = []
        for i in range(0, 25):
        # for i in range(0, 16):
            for c in ['r', 'g', 'b']:
                names.append('sh_%s_%d' % (c, i))
        
        return names

    def to_string(self):
        return "AuxIntegrator[]"