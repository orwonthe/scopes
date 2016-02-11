# Copyright(c) 2014 WillyMillsLLC
import unittest
import math

from optical.light import Position
from optical.light import Direction
from optical.light import Ray
from optical.light import vector_norm
from optical.light import vector_dot_product
from optical.light import WaveFront
from optical.instruments import ScopeMeasurements
from optical.instruments import ImageSensor
from optical.instruments import Aperture
from optical.instruments import SphericalMirror

class TestScopeMeasurements(unittest.TestCase):
    """
    Test ScopeMeasurements Class
    """
    def clean_state(self):
        self.scope_measurements = None
    
    def setUp(self):
        self.clean_state()
        self.scope_measurements = ScopeMeasurements()
        
    def tearDown(self):
        self.clean_state()

    def test_status_getter(self):
        self.scope_measurements['status'] = "happy"
        self.assertEqual("happy",  self.scope_measurements.status)
        
    def test_status_setter(self):
        self.scope_measurements.status = "happy"
        self.assertEqual("happy",  self.scope_measurements['status'])
        
    def test_adds_measurement(self):
        self.scope_measurements['key'] = 'value'
        self.assertTrue(self.scope_measurements.has_key('key'))
        self.assertEqual('value', self.scope_measurements['key'])
        
class TestImageSensor(unittest.TestCase):
    """
    Test ImageSensor class
    """
    def clean_state(self):
        self.imager = None
        self.width = None
        self.height = None
        self.area = None
        self.unit = None
    
    def setUp(self):
        self.clean_state()
        self.width = 8.0
        self.height = 6.0
        self.area = self.width * self.height
        self.unit = "smidgen"
        self.facing_direction = Direction.up
        self.horizontal_direction = Direction.right
        self.vertical_direction = Direction.forward
        self.position = Position.from_xyz((10.0, 20.0, 30))
        self.central_view = Ray(self.position, self.facing_direction)
        self.location_expectations = {
            (0.0, 0.0):(10.0, 20.0, 30.0),
            (1.0, 0.0):(14.0, 20.0, 30.0),
            (-1.0, 0.0):(6.0, 20.0, 30.0),
            (0.0, 1.0):(10.0, 23.0, 30.0),
            (0.0, -1.0):(10.0, 17.0, 30.0),
            (1.0, 1.0):(14.0, 23.0, 30.0),
            (1.0, -1.0):(14.0, 17.0, 30.0),
            (-1.0, 1.0):(6.0, 23.0, 30.0),
            (-1.0, -1.0):(6.0, 17.0, 30.0),
            }
    def tearDown(self):
        self.clean_state()
    
    def test_normal_image_sensor_construction(self):
        self.when_imager_is_constructed()
        self.then_imager_has_correct_dimensions()
    def when_imager_is_constructed(self):
        self.imager = ImageSensor(
            unit = self.unit,
            width = self.width,
            height = self.height,
            central_view = self.central_view,
            horizontal_direction = self.horizontal_direction,
            vertical_direction = self.vertical_direction,
            placement = self.central_view
            )
    def then_imager_has_correct_dimensions(self):
        self.assertEqual(self.width, self.imager.width)
        self.assertEqual(self.height, self.imager.height)
        self.assertEqual(self.area, self.imager.area)
        self.assertEqual(self.unit, self.imager.unit_of_distance)
        
    def test_imager_viewing_positions(self):
        self.when_imager_is_constructed()
        self.then_imager_correct_has_view_locations()
        
    def then_imager_correct_has_view_locations(self):
        for self.image_location, self.expected_sensor_location in self.location_expectations.items():
            self.then_sensor_has_correct_location()
    def then_sensor_has_correct_location(self):
        actual_location_ray = self.imager.view_at_location(self.image_location)
        actual_location_direction = actual_location_ray.direction
        self.assertTupleEqual(self.facing_direction.xyz, actual_location_direction.xyz)
        actual_positon = actual_location_ray.position
        self.assertTupleEqual(self.expected_sensor_location, actual_positon.xyz)

class TestAperture(unittest.TestCase):
    """
    Test Aperture class
    """
    def clean_state(self):
        self.aperture = None
        self.diameter = None
        self.aperture_origin = None
        self.aperture_direction = None
        self.orthogonal_directions = None
        self.placement = None
        self.ray_location = None
        self.orthogonal_direction = None
        self.scale_factor = None
        self.test_ray_distance = None
        self.test_ray = None
        self.initial_wavefront = None
        self.filtered_wavefront = None
    
    def setUp(self):
        self.clean_state()
        self.diameter = 0.1
        self.aperture_origin = Position.from_xyz((10.0, 20.0, 30.0))
        self.aperture_direction = Direction.from_xyz((1.0, 2.0, 3.0))
        self.placement = Ray(self.aperture_origin, self.aperture_direction.normalized())
        self.aperture = Aperture(placement = self.placement, diameter=self.diameter)
        self.initial_wavefront = WaveFront.unit_sphere
        self.orthogonal_directions = [
                self.aperture_direction.cross(Direction.right).normalized(),
                self.aperture_direction.cross(Direction.forward).normalized(),
                self.aperture_direction.cross(Direction.backward).normalized(),
        ]
        
    def tearDown(self):
        self.clean_state()
        
    def test_aperture_radius(self):
        self.assertAlmostEqual(self.diameter/2.0, self.aperture.radius)
    
    def test_aperture_correctly_measures_ray_distances(self):
        self.then_test_with_multiple_location_directions_and_scales()    
    def then_test_with_multiple_location_directions_and_scales(self):        
        for self.ray_location in [
                Position.from_xyz((44.0, 33.0, 2.0)),
                Position.from_xyz((-9.0, 3.0, -2.0)),
            ]:
            self.then_test_location_with_multiple_direction_and_scales()            
    def then_test_location_with_multiple_direction_and_scales(self):
        for self.orthogonal_direction in self.orthogonal_directions:
            self.then_test_location_and_direction_with_multiple_scales()    
    def then_test_location_and_direction_with_multiple_scales(self):
            for self.scale_factor in [1.0, 0.1, 0.01, 0.001, 1000.0]:
                self.then_test_location_direction_and_scale()
    def then_test_location_direction_and_scale(self):
        self.when_test_ray_is_constructed()
        self.when_test_ray_distance_is_measured()
        self.then_test_ray_distance_is_correct()
    def when_test_ray_is_constructed(self):
        scaled_normal = self.orthogonal_direction.scale(self.scale_factor)
        ray_endpoint = self.aperture_origin.add(scaled_normal)
        ray_direction = ray_endpoint.difference(self.ray_location).normalized()
        self.test_ray = Ray(self.ray_location, ray_direction)
    def when_test_ray_distance_is_measured(self):
        self.test_ray_distance = self.aperture.distance_to_ray(self.test_ray)
    def then_test_ray_distance_is_correct(self):        
        self.assertAlmostEqual(self.scale_factor, self.test_ray_distance)

    def test_volume_calculations(self):
        # Testing the test to be sure expected volume is correct.
        # The correct math using 1-cosine(angle) is inaccurate for small angles.
        # Instead a more complex tangent based formula is used.
        # This test makes sure the more complex formula is still correct.
        self.assertEqual(0.0 , self.volume_of_spherical_cone(0.0))
        for angle in [0.1, 0.2, 0.5, 1.0, 0.01]:
            cosine = math.cos(angle)
            tangent = math.tan(angle)
            expected_height_factor = 1.0 - cosine
            actual_height_factor = self.height_factor_of_spherical_cone(tangent)
            self.assertAlmostEqual(expected_height_factor, actual_height_factor)
    def volume_of_spherical_cone(self, tangent):
        height_factor = self.height_factor_of_spherical_cone(tangent)
        volume = height_factor * math.pi * 2.0 / 3.0
        return volume
    def height_factor_of_spherical_cone(self, tangent):
        tangent_squared = tangent * tangent
        secant_squared = 1 + tangent_squared
        secant = math.sqrt(secant_squared)
        alpha_squared = tangent_squared + 2.0 * (1.0 - secant)
        alpha = math.sqrt(alpha_squared)
        height_factor = alpha / secant
        return height_factor
    
    @unittest.skip("slow test because it does a lot of calculation")
    def test_aperture_filters_wavefront(self):
        self.when_wavefront_is_filtered()
        self.then_filter_wavefront_has_correct_properties()
    def when_wavefront_is_filtered(self):        
        self.filtered_wavefront = self.aperture.filter_wavefront(self.initial_wavefront)
    def then_filter_wavefront_has_correct_properties(self):
        distance_to_aperture = vector_norm(self.aperture_origin.xyz)
        tangent = self.aperture.radius / distance_to_aperture
        expected_volume = self.volume_of_spherical_cone(tangent)
        actual_volume = self.filtered_wavefront.volume
        self.assertAlmostEqual(expected_volume, actual_volume, 3)

class TestSphericalMirror(unittest.TestCase):
    """
    Test SphericalMirror class
    """
    def clean_state(self):
        self.mirror = None
        self.aperture = None
        self.diameter = None
        self.aperture_origin = None
        self.aperture_direction = None
        self.aperture_normal = None
        self.orthogonal_directions = None
        self.placement = None
        self.ray_location = None
        self.orthogonal_direction = None
        self.scale_factor = None
        self.test_ray_distance = None
        self.test_ray = None
        self.initial_wavefront = None
        self.filtered_wavefront = None
        self.test_directions = None
        self.test_direction = None
        self.test_point = None
    
    def setUp(self):
        self.clean_state()
        self.diameter = 10.0
        self.focal_length = 50.0
        self.aperture_origin = Position.from_xyz((10.0, 20.0, 30.0))
        self.aperture_direction = Direction.from_xyz((1.0, 2.0, 3.0))
        self.aperture_normal = self.aperture_direction.normalized()
        self.placement = Ray(self.aperture_origin, self.aperture_normal)
        self.orthogonal_directions = [
                self.aperture_direction.cross(Direction.right).normalized(),
                self.aperture_direction.cross(Direction.forward).normalized(),
                self.aperture_direction.cross(Direction.backward).normalized(),
        ]
        self.test_directions = [self.aperture_normal.add(
            orthogonal_direction.scale(0.1)).normalized() for orthogonal_direction in self.orthogonal_directions]
        
    def tearDown(self):
        self.clean_state()
                
    def test_positive_focal_length_is_concave(self):
        self.when_concave_mirror_is_constructed()
        self.then_mirror_is_concave()
    def when_concave_mirror_is_constructed(self):
        self.mirror = SphericalMirror(
            placement = self.placement, 
            diameter = self.diameter,
            focal_length = self.focal_length
        )
    def then_mirror_is_concave(self):
        self.assertFalse(self.mirror.is_convex)
        self.assertTrue(self.mirror.is_concave)
        
    def test_negative_focal_length_is_concave(self):
        self.when_convex_mirror_is_constructed()
        self.then_mirror_is_convex()
    def when_convex_mirror_is_constructed(self):
        self.mirror = SphericalMirror(
            placement = self.placement, 
            diameter = self.diameter,
            focal_length = -self.focal_length
        )
    def then_mirror_is_convex(self):
        self.assertTrue(self.mirror.is_convex)
        self.assertFalse(self.mirror.is_concave)
        
    def test_spherical_radius_is_correct(self):
        self.when_concave_mirror_is_constructed()
        self.then_spherical_radius_is_twice_focal_length()
        self.when_convex_mirror_is_constructed()
        self.then_spherical_radius_is_twice_focal_length()
    def then_spherical_radius_is_twice_focal_length(self):
        spherical_radius = self.mirror.sphere.radius
        self.assertAlmostEqual(2.0 * math.fabs(self.focal_length), spherical_radius)
        
    def test_aperture_circle_lies_on_sphere(self):
        self.when_concave_mirror_is_constructed()
        self.then_aperture_circle_lies_on_sphere()
        self.when_convex_mirror_is_constructed()
        self.then_aperture_circle_lies_on_sphere()
    def then_aperture_circle_lies_on_sphere(self):
        for self.orthogonal_direction in self.orthogonal_directions:
            self.when_aperture_circle_point_is_constructed()
            self.then_aperture_circle_point_lies_on_sphere()
    def when_aperture_circle_point_is_constructed(self):
        self.test_direction = self.orthogonal_direction.scale(self.diameter * 0.5)
        self.test_point = self.aperture_origin.add(self.test_direction)
    def then_aperture_circle_point_lies_on_sphere(self):
        actual_elevation = self.mirror.sphere.elevation(self.test_point)
        self.assertAlmostEqual(0.0, actual_elevation)
    
    def test_rays_from_center_of_sphere_reflect_back(self):
        self.when_concave_mirror_is_constructed()
        self.when_test_rays_are_positioned_at_sphere_center()
        self.then_test_rays_reflect_back()
    def when_test_rays_are_positioned_at_sphere_center(self):
        self.test_center_position = self.mirror.sphere.center_position
        self.when_test_rays_are_constructed()
    def when_test_rays_are_constructed(self):        
        self.test_rays = [Ray(self.test_center_position, test_direction) 
            for test_direction in self.test_directions]
    def then_test_rays_reflect_back(self):
        for self.test_ray in self.test_rays:
            self.when_test_ray_is_reflected()
            self.then_reflected_ray_is_opposite_of_test_ray()
    def when_test_ray_is_reflected(self):
        self.reflected_ray = self.mirror.reflect(self.test_ray)
    def then_reflected_ray_is_opposite_of_test_ray(self):
        dotProduct = vector_dot_product(self.test_ray.direction.xyz, self.reflected_ray.direction.xyz)
        self.assertAlmostEqual(-1.0, dotProduct)

    def test_rays_from_focal_point_of_sphere_reflect_parallel(self):
        self.when_concave_mirror_is_constructed()
        self.when_test_rays_are_positioned_at_focal_point()
        self.then_test_rays_reflect_parallel()
    def when_test_rays_are_positioned_at_focal_point(self):
        self.test_center_position = self.mirror.sphere.center_position.add(
                self.aperture_normal.scale(self.focal_length)
            )
        self.when_test_rays_are_constructed()
    def then_test_rays_reflect_parallel(self):
        for self.test_ray in self.test_rays:
            self.when_test_ray_is_reflected()
            self.then_reflected_ray_is_opposite_of_aperture_normal()
    def then_reflected_ray_is_opposite_of_aperture_normal(self):        
        dotProduct = vector_dot_product(self.aperture_normal.xyz, self.reflected_ray.direction.xyz)
        self.assertAlmostEqual(-1.0, dotProduct)
        
    
if __name__ == '__main__':
    unittest.main()

