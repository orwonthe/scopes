# Copyright(c) 2014 WillyMillsLLC
import unittest
from math import fabs
from math import pi
from math import sqrt

from numpy import matrix

from optical.light import Direction
from optical.light import FilterMonitor
from optical.light import Plane
from optical.light import Position
from optical.light import Projector
from optical.light import Ray
from optical.light import Sphere
from optical.light import SphericalTriangle
from optical.light import TriangleClassification
from optical.light import WaveFront
from optical.light import WaveStatistics
from optical.light import column_of_matrix_as_tuple
from optical.light import row_of_matrix_as_tuple
from optical.light import vector_add
from optical.light import vector_difference
from optical.light import vector_distance
from optical.light import vector_distance_squared
from optical.light import vector_dot_product
from optical.light import vector_interpolation
from optical.light import vector_norm
from optical.light import vector_norm_squared
from optical.light import vector_scale


class TestLightMethods(unittest.TestCase):
    """
    Test functions of light module
    """

    def clean_state(self):
        self.target_matrix = None

    def setUp(self):
        self.clean_state()

    def tearDown(self):
        self.clean_state()

    def test_vector_dot_product(self):
        self.assertEqual(123, vector_dot_product((1, 2, 3), (100, 10, 1)))

    def test_vector_norm_squared(self):
        self.assertAlmostEqual(50.0, vector_norm_squared((3.0, 4.0, 5.0)))

    def test_vector_norm(self):
        self.assertAlmostEqual(5.0, vector_norm((3.0, 4.0)))

    def test_vector_difference(self):
        self.assertTupleEqual((1.0, 2.0), vector_difference((11.0, 22.0), (10.0, 20.0)))

    def test_vector_add(self):
        self.assertTupleEqual((11.0, 22.0), vector_add((1.0, 2.0), (10.0, 20.0)))

    def test_vector_scale(self):
        self.assertTupleEqual((110.0, 230.0), vector_scale((11.0, 23.0), 10.0))

    def test_vector_interpolation(self):
        self.assertTupleEqual((15.0, 35.0), vector_interpolation((10.0, 20.0), (20.0, 50.0)))

    def test_vector_distance(self):
        expect_square = 5.0
        expect = sqrt(expect_square)
        self.assertAlmostEqual(expect, vector_distance((11.0, 22.0), (10.0, 20.0)))
        self.assertAlmostEqual(expect_square, vector_distance_squared((11.0, 22.0), (10.0, 20.0)))

    def test_column_of_matrix_as_tuple(self):
        self.when_target_matrix_is_set()
        self.assertTupleEqual((1, 4), column_of_matrix_as_tuple(self.target_matrix))
        self.assertTupleEqual((1, 4), column_of_matrix_as_tuple(self.target_matrix, 0))
        self.assertTupleEqual((2, 5), column_of_matrix_as_tuple(self.target_matrix, 1))
        self.assertTupleEqual((3, 6), column_of_matrix_as_tuple(self.target_matrix, 2))

    def when_target_matrix_is_set(self):
        self.target_matrix = matrix([
            [1, 2, 3],
            [4, 5, 6]
        ])

    def test_row_of_matrix_as_tuple(self):
        self.when_target_matrix_is_set()
        self.assertTupleEqual((1, 2, 3), row_of_matrix_as_tuple(self.target_matrix))
        self.assertTupleEqual((1, 2, 3), row_of_matrix_as_tuple(self.target_matrix, 0))
        self.assertTupleEqual((4, 5, 6), row_of_matrix_as_tuple(self.target_matrix, 1))


class TestProjector(unittest.TestCase):
    """
    Test Projector Class
    """

    def clean_state(self):
        self.projector = None
        self.w = None
        self.x = None
        self.y = None
        self.z = None
        self.wxyz = None
        self.xyz = None

    def setUp(self):
        self.clean_state()
        self.w = 1.1
        self.x = 2.2
        self.y = 3.3
        self.z = 4.4
        self.wxyz = (self.w, self.x, self.y, self.z)
        self.xyz = (self.x, self.y, self.z)

    def tearDown(self):
        self.clean_state()

    def test_projector_constructor_requires_4_tuple(self):
        with self.assertRaises(ValueError):
            Projector(self.xyz)

    def test_projector_create_normal(self):
        self.when_projector_is_created()
        self.then_projector_has_expected_values()

    def when_projector_is_created(self):
        self.projector = Projector(self.wxyz)

    def then_projector_has_expected_values(self):
        coords = self.projector.coordinates
        self.assertTupleEqual(self.wxyz, coords)
        self.assertTupleEqual(self.xyz, self.projector.xyz)
        self.assertEqual(self.w, self.projector.w)
        self.assertEqual(self.x, self.projector.x)
        self.assertEqual(self.y, self.projector.y)
        self.assertEqual(self.z, self.projector.z)

    def test_projector_accepts_list(self):
        self.when_projector_is_created_from_list()
        self.then_projector_has_expected_values()

    def when_projector_is_created_from_list(self):
        self.projector = Projector([self.w, self.x, self.y, self.z])

    def test_projector_iterates(self):
        self.when_projector_is_created()
        self.then_projector_iterates_coordinates()

    def then_projector_iterates_coordinates(self):
        tuple_from_iteration = tuple(self.projector)
        self.assertTupleEqual(self.wxyz, tuple_from_iteration)

    def test_projector_recognizes_position(self):
        self.assertTrue(Projector((1.0, 0.0, 0.0, 0.0)).is_position)
        self.assertTrue(Projector((1.0, 2.2, 3.3, 4.4)).is_position)
        self.assertFalse(Projector((1.1, 0.0, 0.0, 0.0)).is_position)
        self.assertFalse(Projector((2.0, 2.2, 3.3, 4.4)).is_position)
        self.assertFalse(Projector((0.0, 0.0, 0.0, 0.0)).is_position)
        self.assertFalse(Projector((0.0, 1.0, 1.0, 1.0)).is_position)

    def test_projector_recognizes_direction(self):
        self.assertTrue(Projector((0.0, 0.0, 0.0, 0.0)).is_direction)
        self.assertTrue(Projector((0.0, 1.0, 1.0, 1.0)).is_direction)
        self.assertFalse(Projector((1.0, 0.0, 0.0, 0.0)).is_direction)
        self.assertFalse(Projector((1.0, 2.2, 3.3, 4.4)).is_direction)
        self.assertFalse(Projector((1.1, 0.0, 0.0, 0.0)).is_direction)
        self.assertFalse(Projector((2.0, 2.2, 3.3, 4.4)).is_direction)

    def test_projector_dot_product(self):
        alpha = Projector((9.0, 7.0, 5.0, 3.0))
        beta = Projector((1000, 100, 10, 1))
        self.assertAlmostEqual(9753.0, alpha.dot(beta))
        self.assertAlmostEqual(9753.0, beta.dot(alpha))
        self.assertAlmostEqual(9753.0, beta.dot((9.0, 7.0, 5.0, 3.0)))


class TestPosition(unittest.TestCase):
    """
    Test Position Class
    
    Indirectly tests ContravariantProjector base class
    """

    def clean_state(self):
        self.position = None
        self.w = None
        self.x = None
        self.y = None
        self.z = None
        self.wxyz = None
        self.xyz = None

    def setUp(self):
        self.clean_state()
        self.w = 1.0
        self.x = 3.0
        self.y = 4.0
        self.z = 5.0
        self.wxyz = (self.w, self.x, self.y, self.z)
        self.xyz = (self.x, self.y, self.z)

    def tearDown(self):
        self.clean_state()

    def test_position_constructor_requires_tuple(self):
        with self.assertRaises(TypeError):
            Position("wrong stuff")

    def test_position_constructor_requires_4_tuple(self):
        with self.assertRaises(ValueError):
            Position(self.xyz)

    def test_position_constructor_requires_unit_w(self):
        with self.assertRaises(ValueError):
            Position((1.0001, 2.2, 3.3, 4.4))

    def test_position_create_normal(self):
        self.when_position_created()
        self.then_position_has_expected_values()

    def when_position_created(self):
        self.position = Position(self.wxyz)

    def then_position_has_expected_values(self):
        self.assertTrue(self.position.is_position)
        self.assertFalse(self.position.is_direction)
        self.assertTupleEqual(self.xyz, self.position.xyz)

    def test_position_from_xyz_normal(self):
        self.when_position_created_from_xyz()
        self.then_position_has_expected_values()

    def when_position_created_from_xyz(self):
        self.position = Position.from_xyz(self.xyz)

    def test_position_from_xyz_requires_tuple(self):
        with self.assertRaises(TypeError):
            Position.from_xyz("wrong stuff")

    def test_position_from_xyz_requires_3_tuple(self):
        with self.assertRaises(ValueError):
            Position.from_xyz(self.wxyz)

    def test_position_origin(self):
        self.assertTupleEqual((0.0, 0.0, 0.0), Position.origin.xyz)

    def test_position_add(self):
        self.when_position_created()
        result = self.position.add((0.0, 1.0, 2.0, 3.0))
        self.assertTupleEqual((4.0, 6.0, 8.0), result.xyz)

    def test_position_direction_from_origin(self):
        self.when_position_created()
        self.direction = self.position.direction_from_origin()
        self.assertTupleEqual(self.position.xyz, self.direction.xyz)


class TestDirection(unittest.TestCase):
    """
    Test Direction Class
    """

    def clean_state(self):
        self.direction = None
        self.w = None
        self.x = None
        self.y = None
        self.z = None
        self.wxyz = None
        self.xyz = None

    def setUp(self):
        self.clean_state()
        self.w = 0.0
        self.x = 3.0
        self.y = 4.0
        self.z = 5.0
        self.wxyz = (self.w, self.x, self.y, self.z)
        self.xyz = (self.x, self.y, self.z)

    def tearDown(self):
        self.clean_state()

    def test_direction_constructor_requires_tuple(self):
        with self.assertRaises(TypeError):
            Direction("wrong stuff")

    def test_direction_constructor_requires_4_tuple(self):
        with self.assertRaises(ValueError):
            Direction((2.2, 3.3, 4.4))

    def test_direction_constructor_requires_zeroed_w(self):
        with self.assertRaises(ValueError):
            Direction((0.0001, 2.2, 3.3, 4.4))

    def test_direction_create_normal(self):
        self.when_direction_created()
        self.then_direction_has_expected_values()

    def when_direction_created(self):
        self.direction = Direction(self.wxyz)

    def then_direction_has_expected_values(self):
        self.assertFalse(self.direction.is_position)
        self.assertTrue(self.direction.is_direction)
        self.assertEqual(self.xyz, self.direction.xyz)
        expected_length = sqrt(50)
        self.assertAlmostEqual(expected_length, self.direction.length())

    def test_direction_from_xyz_normal(self):
        self.when_direction_created_from_xyz()
        self.then_direction_has_expected_values()

    def when_direction_created_from_xyz(self):
        self.direction = Direction.from_xyz(self.xyz)

    def test_direction_from_xyz_requires_tuple(self):
        with self.assertRaises(TypeError):
            Direction.from_xyz("wrong stuff")

    def test_direction_from_xyz_requires_3_tuple(self):
        with self.assertRaises(ValueError):
            Direction.from_xyz((0.0, 2.2, 3.3, 4.4))

    def test_direction_cardinal_directions(self):
        self.assertTupleEqual((0.0, 0.0, 1.0), Direction.up.xyz)
        self.assertTupleEqual((0.0, 0.0, -1.0), Direction.down.xyz)
        self.assertTupleEqual((-1.0, 0.0, 0.0), Direction.left.xyz)
        self.assertTupleEqual((1.0, 0.0, 0.0), Direction.right.xyz)
        self.assertTupleEqual((0.0, 1.0, 0.0), Direction.forward.xyz)
        self.assertTupleEqual((0.0, -1.0, 0.0), Direction.backward.xyz)
        self.assertTupleEqual((0.0, 0.0, 0.0), Direction.unknown.xyz)

    def test_direction_scales_by_one(self):
        self.when_direction_created_from_xyz()
        self.direction = self.direction.scale(1.0)
        self.then_direction_has_expected_values()

    def test_direction_scales(self):
        self.when_direction_created_from_xyz()
        self.direction = self.direction.scale(2.0)
        self.assertTrue(self.direction.is_direction)
        self.assertTupleEqual((6.0, 8.0, 10.0), self.direction.xyz)

    def test_direction_normalized(self):
        self.when_direction_created_from_xyz()
        self.direction = self.direction.normalized()
        self.assertAlmostEqual(1.0, vector_norm(self.direction))

    def test_cross_product(self):
        a = Direction.from_xyz((2.0, 3.0, 4.0))
        b = Direction.from_xyz((10.0, 100.0, 1000.0))
        expected_direction = Direction.from_xyz((
            3.0 * 1000.0 - 4.0 * 100.0,
            4.0 * 10.0 - 2.0 * 1000.0,
            2.0 * 100.0 - 3.0 * 10.0
        ))
        actual_direction = a.cross(b)
        self.assertTupleEqual(expected_direction.coordinates, actual_direction.coordinates)
        # cross product is orthoganal to arguments
        self.assertAlmostEqual(0.0, a.dot(actual_direction))
        self.assertAlmostEqual(0.0, b.dot(actual_direction))


class TestRay(unittest.TestCase):
    """
    Test Ray Class
    """

    def clean_state(self):
        self.expected_position = None
        self.expected_direction = None
        self.endpoint_xyz = None
        self.ten_scale_endpoint_xyz = None
        self.ray = None

    def setUp(self):
        self.clean_state()
        self.expected_position = Position.from_xyz((10.0, 20.0, 30.0))
        self.expected_direction = Direction.from_xyz((0.5, 0.4, 0.3))
        self.endpoint_xyz = (10.5, 20.4, 30.3)
        self.ten_scale_endpoint_xyz = (15.0, 24.0, 33.0)

    def tearDown(self):
        self.clean_state()

    def test_ray_normal_construction(self):
        self.when_ray_created()
        self.then_ray_has_normal_values()

    def when_ray_created(self):
        self.ray = Ray(self.expected_position, self.expected_direction)

    def then_ray_has_normal_values(self):
        self.then_ray_has_expected_position()
        self.then_ray_has_expected_direction()

    def then_ray_has_expected_position(self):
        self.assertAlmostEqual(self.expected_position.w, self.ray.position.w)
        self.assertAlmostEqual(self.expected_position.x, self.ray.position.x)
        self.assertAlmostEqual(self.expected_position.y, self.ray.position.y)
        self.assertAlmostEqual(self.expected_position.z, self.ray.position.z)

    def then_ray_has_expected_direction(self):
        self.assertAlmostEqual(self.expected_direction.w, self.ray.direction.w)
        self.assertAlmostEqual(self.expected_direction.x, self.ray.direction.x)
        self.assertAlmostEqual(self.expected_direction.y, self.ray.direction.y)
        self.assertAlmostEqual(self.expected_direction.z, self.ray.direction.z)

    def test_ray_construction_requires_valid_position(self):
        with self.assertRaises(TypeError):
            Ray("wrong stuff", self.expected_direction)

    def test_ray_construction_requires_valid_direction(self):
        with self.assertRaises(TypeError):
            Ray(self.expected_position, "wrong stuff")

    def test_ray_construction_refuses_swapped_arguments(self):
        with self.assertRaises(TypeError):
            Ray(self.expected_direction, self.expected_position)

    def test_ray_from_origin_endpoint(self):
        self.when_ray_created_from_origin_endpoint()
        self.then_ray_has_normal_values()

    def when_ray_created_from_origin_endpoint(self):
        self.ray = Ray.from_origin_endpoint(self.expected_position, Position.from_xyz(self.endpoint_xyz))

    def test_ray_wont_mutate(self):
        self.when_ray_created()
        with self.assertRaises(AttributeError):
            self.ray.position = Position((1.0, 2.0, 3.0, 4.0))
        with self.assertRaises(AttributeError):
            self.ray.direction = Direction((0.0, 2.0, 3.0, 4.0))

    def test_ray_has_cardinal_rays(self):
        for direction_name in ['up', 'down', 'left', 'right', 'forward', 'backward', 'unknown']:
            target = Ray.get_named_cardinal_ray(direction_name)
            expected_direction = Direction.get_named_cardinal_direction(direction_name)
            self.assertEqual(expected_direction.xyz, target.direction.xyz)
            self.assertEqual(Position.origin.xyz, target.position.xyz)

    def test_ray_endpoint(self):
        self.when_ray_created()
        endpoint = self.ray.endpoint;
        self.assertTrue(endpoint.is_position)
        self.assertEqual(self.endpoint_xyz, endpoint.xyz)

    def test_ray_scaled_endpoint(self):
        self.when_ray_created()
        endpoint = self.ray.scaled_endpoint(10.0);
        self.assertTrue(endpoint.is_position)
        self.assertEqual(self.ten_scale_endpoint_xyz, endpoint.xyz)
        self.assertLess(self.ray.distance_to_point(endpoint), 0.000001)

    def test_distance_to_ray(self):
        for ray in [Ray.up, Ray.down]:
            scaled_ray = ray.scaled_direction(10.0)
            for orthogonal_ray in [Ray.right, Ray.left, Ray.forward, Ray.backward]:
                scale_factor = 4.0
                scaled_orthogonal_ray = orthogonal_ray.scaled_direction(scale_factor)
                endpoint = scaled_orthogonal_ray.endpoint
                expected_distance = scale_factor
                actual_distance = ray.distance_to_point(endpoint)
                self.assertAlmostEqual(expected_distance, actual_distance)

    def test_ray_additon(self):
        self.when_ray_created()
        self.when_direction_added_to_ray()
        self.then_ray_has_translated_values()

    def when_direction_added_to_ray(self):
        self.ray = self.ray.add(Direction.from_xyz((100, 200, 300)))

    def then_ray_has_translated_values(self):
        self.then_ray_has_translated_position()
        self.then_ray_has_expected_direction()

    def then_ray_has_translated_position(self):
        self.assertTupleEqual((110, 220, 330), self.ray.position.xyz)


class TestPlane(unittest.TestCase):
    def clean_state(self):
        self.vector = None
        self.unit_vector = None
        self.position = None
        self.direction = None
        self.unit_direction = None
        self.ray = None
        self.plane = None
        self.orthogonal_directions = None
        self.orthogonal_rays = None
        self.other_position = None
        self.other_direction = None
        self.other_ray = None
        self.intersect_point = None

    def setUp(self):
        self.clean_state()
        self.vector = (2.0, 3.0, 4.0)
        vector_length = vector_norm(self.vector)
        self.unit_vector = tuple(component / vector_length for component in self.vector)
        self.position = Position.from_xyz((10.0, 20.0, 30.0))
        self.direction = Direction.from_xyz(self.vector)
        self.unit_direction = Direction.from_xyz(self.unit_vector)
        self.ray = Ray(self.position, self.direction)
        self.unit_ray = Ray(self.position, self.unit_direction)
        self.other_position = Position.from_xyz((-9.0, -88.0, 12.34))
        self.other_direction = Direction.from_xyz((19.0, 8.0, -0.03))
        self.other_ray = Ray(self.other_position, self.other_direction)
        offset = self.position.dot(self.unit_direction)
        plane_projector = (-offset,) + self.unit_vector
        self.plane = Plane(plane_projector)
        self.orthogonal_directions = [
            Direction.from_xyz((0.2, 0.2, -0.25)),
            Direction.from_xyz((-7.0, 2.0, 2.0))
        ]
        self.orthogonal_rays = [Ray(self.position, direction)
                                for direction in self.orthogonal_directions]

    def tearDown(self):
        self.clean_state()

    def test_plane_contains_position(self):
        point = self.position
        distance_from_plane = self.plane.distance_to_point(point)
        self.assertAlmostEqual(0.0, distance_from_plane)

    def test_plane_contains_orthogonal_rays(self):
        self.then_plane_contains_orthogonal_rays()

    def then_plane_contains_orthogonal_rays(self):
        for orthogonal_ray in self.orthogonal_rays:
            for scale in [-2.0, -1.0, 0.0, 0.4, 1.0]:
                point = orthogonal_ray.scaled_endpoint(scale)
                distance_from_plane = self.plane.distance_to_point(point)
                self.assertAlmostEqual(0.0, distance_from_plane)
                # this really is testing Ray.distance_to_point
                length_of_orthogonal_ray = orthogonal_ray.direction.length()
                expected_distance_from_ray = fabs(scale) * length_of_orthogonal_ray
                actual_distance_from_ray = self.ray.distance_to_point(point)
                self.assertAlmostEqual(expected_distance_from_ray, actual_distance_from_ray)

    def test_plane_distance(self):
        self.then_plane_has_correct_distances()

    def then_plane_has_correct_distances(self):
        ray_length = vector_norm(self.unit_direction)
        self.assertEqual(1.0, ray_length)
        ray = self.unit_ray
        for scale in [-2.0, -1.0, 0.0, 0.4, 1.0]:
            point = ray.scaled_endpoint(scale)
            distance_from_plane = self.plane.distance_to_point(point)
            self.assertAlmostEqual(scale, distance_from_plane)

    def test_plane_construction_from_ray(self):
        self.when_plane_constructed_from_ray()
        self.then_plane_contains_orthogonal_rays()
        self.then_plane_has_correct_distances()

    def when_plane_constructed_from_ray(self):
        self.plane = Plane.orthogonal_to_ray(self.ray)

    def test_plane_intersects_ray(self):
        self.when_plane_constructed_from_ray()
        self.when_plane_intersects_other_ray()
        self.then_intersection_point_lies_on_plane()
        self.then_point_lies_on_othe_ray()

    def when_plane_intersects_other_ray(self):
        self.intersect_point = self.plane.intersect(self.other_ray)

    def then_intersection_point_lies_on_plane(self):
        self.assertAlmostEqual(0.0, self.plane.distance_to_point(self.intersect_point), 5)

    def then_point_lies_on_othe_ray(self):
        self.assertLess(0.0, self.other_ray.distance_to_point(self.intersect_point), 5)


class TestSphere(unittest.TestCase):
    """
    Test Sphere class
    """

    def clean_state(self):
        self.sphere = None
        self.expected_center = None
        self.expected_radius = None
        self.starting_point = None
        self.test_ray = None
        self.intersection_points = None

    def setUp(self):
        self.clean_state()
        self.expected_center = Position.from_xyz((10.0, 20.0, 30.0))
        self.expected_radius = 5.0

    def tearDown(self):
        self.clean_state()

    def test_ordinary_sphere_construction(self):
        self.when_ordinary_sphere_is_constructed()
        self.then_sphere_has_ordinary_values()

    def when_ordinary_sphere_is_constructed(self):
        self.sphere = Sphere(self.expected_center, self.expected_radius)

    def then_sphere_has_ordinary_values(self):
        self.assertAlmostEqual(self.expected_radius, self.sphere.radius)
        self.assertAlmostEqual(self.expected_center.x, self.sphere.x)
        self.assertAlmostEqual(self.expected_center.y, self.sphere.y)
        self.assertAlmostEqual(self.expected_center.z, self.sphere.z)

    def test_elevation_above_sphere_surface(self):
        self.when_ordinary_sphere_is_constructed()
        for expected_elevation, test_point in [
            (-self.expected_radius, self.expected_center),
            (0.0, Position.from_xyz((15.0, 20.0, 30.0))),
            (0.0, Position.from_xyz((5.0, 20.0, 30.0))),
            (0.0, Position.from_xyz((10.0, 23.0, 34.0))),
            (sqrt(125.0) - 5.0, Position.from_xyz((0.0, 23.0, 34.0))),
        ]:
            self.then_point_has_correct_elevation(expected_elevation, test_point)

    def then_point_has_correct_elevation(self, expected_elevation, test_point):
        actual_elevation = self.sphere.elevation(test_point)
        self.assertAlmostEqual(expected_elevation, actual_elevation)

    def test_sphere_intersects_rays_through_center(self):
        self.when_ordinary_sphere_is_constructed()
        self.then_sphere_intersects_rays_correctly()

    def then_sphere_intersects_rays_correctly(self):
        starting_points = [
            Position.from_xyz((0.0, 0.0, 0.0)),
            Position.from_xyz((100.0, 200.0, 300.0)),
            Position.from_xyz((-50.0, 10.0, 10.0)),
        ]
        interior_points = [
            self.expected_center,
            self.expected_center.add(Direction.up.scale(4.0)),
            self.expected_center.add(Direction.left.scale(4.0)),
            self.expected_center.add(Direction.backward.scale(4.0)),
        ]
        for self.starting_point in starting_points:
            for self.interior_point in interior_points:
                self.when_sphere_intersects_ray_through_interior_point()
                self.then_intersection_points_are_correct()

    def when_sphere_intersects_ray_through_interior_point(self):
        self.test_ray = Ray.from_origin_endpoint(self.starting_point, self.interior_point)

    def then_intersection_points_are_correct(self):
        self.when_intersection_points_are_calculated()
        self.then_there_are_two_intersection_points()
        self.then_all_intersection_points_lie_on_both_sphere_and_ray()

    def when_intersection_points_are_calculated(self):
        self.intersection_points = self.sphere.intersection_points(self.test_ray)

    def then_there_are_two_intersection_points(self):
        self.assertEqual(2, len(self.intersection_points))

    def then_all_intersection_points_lie_on_both_sphere_and_ray(self):
        self.then_intersection_points_lie_on_ray()
        self.then_intersection_points_lie_on_sphere()

    def then_intersection_points_lie_on_ray(self):
        for intersection_point in self.intersection_points:
            distance_to_ray = self.test_ray.distance_to_point(intersection_point)
            self.assertAlmostEqual(0.0, distance_to_ray, 4)

    def then_intersection_points_lie_on_sphere(self):
        for intersection_point in self.intersection_points:
            elevation_above_sphere = self.sphere.elevation(intersection_point)
            self.assertAlmostEqual(0.0, elevation_above_sphere, 4)

    def test_sphere_has_no_intersection_points_with_distant_ray(self):
        self.when_ordinary_sphere_is_constructed()
        self.when_distant_ray_is_constructed()
        self.when_intersection_points_are_calculated()
        self.then_there_are_no_interection_points()

    def when_distant_ray_is_constructed(self):
        self.test_ray = Ray(Position.from_xyz((0.0, 20.0, 30.0)), Direction.up)

    def then_there_are_no_interection_points(self):
        self.assertEqual(0, len(self.intersection_points))


class TestWaveStatistics(unittest.TestCase):
    """
    Test WaveStatistics class
    """

    def clean_state(self):
        self.direction = None
        self.volume = None
        self.wave_statistic = None

    def setUp(self):
        self.clean_state()
        self.direction = Direction.from_xyz((2.0, 3.0, 4.0))
        self.volume = 10.0

    def tearDown(self):
        self.clean_state()

    def test_wave_statistics_construction(self):
        self.when_wave_statistic_constructed()
        self.then_wave_statistic_has_expected_values()

    def when_wave_statistic_constructed(self):
        self.wave_statistic = WaveStatistics(self.volume, self.direction)

    def then_wave_statistic_has_expected_values(self):
        self.assertEqual(self.volume, self.wave_statistic.volume)
        self.assertTupleEqual(self.direction.xyz, self.wave_statistic.central_direction.xyz)
        self.assertTupleEqual((20.0, 30.0, 40.0),
                              self.wave_statistic.volume_weighted_direction.xyz)
        self.assertTupleEqual(self.direction.normalized().xyz,
                              self.wave_statistic.normalized_direction.xyz)

    def test_wave_statistics_accumulate(self):
        self.when_wave_statistic_constructed()
        self.when_wave_statistic_added()
        self.when_wave_statistic_added()
        self.assertEqual(3.0 * self.volume, self.wave_statistic.volume)
        self.assertTupleEqual(self.direction.normalized().xyz,
                              self.wave_statistic.normalized_direction.xyz)

    def when_wave_statistic_added(self):
        self.wave_statistic.add(WaveStatistics(self.volume, self.direction))


class TestSphericalTriangle(unittest.TestCase):
    """
    Test SphericalTriangle Class
    """

    def clean_state(self):
        self.triangle = None
        self.vertices = None
        self.center_ray = None
        self.test_direction = None

    def setUp(self):
        self.clean_state()
        self.vertices = (Ray.right, Ray.forward, Ray.up)

    def tearDown(self):
        self.clean_state()

    def test_spherical_triangle_normal_constrcution(self):
        self.when_triangle_created()
        self.then_triangle_has_normal_values()

    def when_triangle_created(self):
        self.triangle = SphericalTriangle(self.vertices)

    def then_triangle_has_normal_values(self):
        self.assertEqual(self.triangle.vertices, self.vertices)
        self.assertAlmostEqual(1.0 / 6.0, self.triangle.volume)
        self.assertAlmostEqual(sqrt(2.0), self.triangle.diameter)

    def test_spherical_triangle_constructor_requires_tuple(self):
        with self.assertRaises(TypeError):
            SphericalTriangle("wrong stuff")

    def test_spherical_triangle_constructor_requires_3_tuple(self):
        with self.assertRaises(ValueError):
            SphericalTriangle((Ray.right, Ray.forward))

    def test_spherical_triangle_constructor_requires_ray_vertices(self):
        with self.assertRaises(TypeError):
            SphericalTriangle((Ray.right, "wrong stuff", Ray.forward))

    def test_spherical_triangle_orientation(self):
        self.when_triangle_created()
        self.then_triangle_has_positive_orientation()

    def then_triangle_has_positive_orientation(self):
        self.assertTrue(self.triangle.has_positive_orientation)

    def test_spherical_triangle_opposite_orientation(self):
        self.when_triangle_created()
        self.when_triangle_inverted()
        self.then_triangle_has_negative_orientation()

    def when_triangle_inverted(self):
        self.triangle = self.triangle.opposite_orientation()

    def then_triangle_has_negative_orientation(self):
        self.assertFalse(self.triangle.has_positive_orientation)

    def test_spherical_triangle_center_ray(self):
        self.when_triangle_created()
        self.when_center_ray_extracted()
        self.then_center_ray_has_correct_values()

    def when_center_ray_extracted(self):
        self.center_ray = self.triangle.center_ray

    def then_center_ray_has_correct_values(self):
        root_third = sqrt(1.0 / 3.0)
        expected_direction_xyz = (root_third, root_third, root_third)
        self.assertTupleEqual(expected_direction_xyz, self.center_ray.direction.xyz)

    def test_spherical_triangle_normal_constrcution(self):
        self.when_triangle_created()
        self.when_triangle_repositioned()
        self.then_triangle_has_repositioned_values()

    def when_triangle_repositioned(self):
        self.test_direction = Direction.from_xyz((1000.0, 2000.0, 3000.0))
        self.triangle = self.triangle.add(self.test_direction)

    def then_triangle_has_repositioned_values(self):
        expected_results = [vertex.add(self.test_direction) for vertex in self.vertices]
        actual_results = [vertex for vertex in self.triangle]
        for index in range(3):
            expected_vertex = expected_results[index]
            actual_vertex = actual_results[index]
            self.assertTupleEqual(expected_vertex.position.xyz, actual_vertex.position.xyz)
            self.assertTupleEqual(expected_vertex.direction.xyz, actual_vertex.direction.xyz)


class TestWaveFront(unittest.TestCase):
    """
    Test WaveFront Class
    """

    def setUp(self):
        self.wavefront = None
        self.triangles = [
            SphericalTriangle((Ray.right, Ray.forward, Ray.up)),
            SphericalTriangle((Ray.right, Ray.down, Ray.forward)),
            SphericalTriangle((Ray.left, Ray.forward, Ray.down)),
        ]
        self.unit_sphere_volume_sequence = [
            4.0 / 3.0,
            2.942809041582063,
            3.817729619828364,
            4.091600620085994,
            4.164203596248617,
        ]
        self.volume_of_unit_sphere = 4.0 / 3.0 * pi  # 4.1887902047863905
        self.accepted_diameter = 0.2
        self.orifice_diameter = 0.2

    def tearDown(self):
        self.wavefront = None
        self.triangles = None;
        self.unit_sphere_volume_sequence = None
        self.volume_of_unit_sphere = None
        self.accepted_diameter = None
        self.orifice_diameter = None

    def test_wave_front_normal_construction(self):
        self.when_wavefront_created()
        self.then_wavefront_has_normal_values()

    def when_wavefront_created(self):
        self.wavefront = WaveFront(self.triangles)

    def then_wavefront_has_normal_values(self):
        self.assertEqual(self.wavefront.triangles, tuple(self.triangles))

    def test_wavefront_constructor_requires_triangle_collection(self):
        with self.assertRaises(TypeError):
            WaveFront(self.triangles + ["wrong stuff"])

    def test_wavefront_unit_sphere_has_expected_values(self):
        self.when_wavefront_set_to_unit_sphere()
        self.then_wavefront_has_unit_sphere_content()

    def when_wavefront_set_to_unit_sphere(self):
        self.wavefront = WaveFront.unit_sphere

    def then_wavefront_has_unit_sphere_content(self):
        self.assertEqual(8, self.wavefront.count)
        expected_volume = self.unit_sphere_volume_sequence[0]
        self.assertAlmostEqual(expected_volume, self.wavefront.volume)

    def test_wavefront_subdivision(self):
        self.when_wavefront_set_to_unit_sphere()
        self.then_sequence_of_subdivided_wavefronts_meet_expectations()

    def then_sequence_of_subdivided_wavefronts_meet_expectations(self):
        expected_volume_sequence = self.unit_sphere_volume_sequence[1:]
        for depth, expected_volume in enumerate(expected_volume_sequence):
            self.when_wavefront_is_subdivided()
            self.then_wavefront_has_correct_count_for_depth(depth)
            self.then_wavefront_converges_on_perfect_volume_from_below()
            self.then_wavefront_has_expected_volume(expected_volume)

    def when_wavefront_is_subdivided(self):
        self.wavefront = self.wavefront.subdivide()

    def then_wavefront_has_correct_count_for_depth(self, depth):
        expected_count = 32 * 4 ** depth
        self.assertEqual(expected_count, self.wavefront.count)

    def then_wavefront_converges_on_perfect_volume_from_below(self):
        perfect_volume = self.volume_of_unit_sphere
        self.assertLess(self.wavefront.volume, perfect_volume)

    def then_wavefront_has_expected_volume(self, expected_volume):
        self.assertAlmostEqual(expected_volume, self.wavefront.volume)

    def test_wavefront_refine_with_diameter_filter(self):
        self.when_wavefront_set_to_unit_sphere()
        self.when_wavefront_is_refined_by_diameter()
        self.then_wavefront_has_expected_count_and_volume()
        self.then_each_triangle_in_wavefront_has_refined_diameter()

    def when_wavefront_is_refined_by_diameter(self):
        def testfilter(triangle):
            if triangle.diameter < self.accepted_diameter:
                return TriangleClassification.Accept
            else:
                return TriangleClassification.Refine

        self.wavefront = self.wavefront.refine_with_filter(testfilter)

    def then_wavefront_has_expected_count_and_volume(self):
        self.assertEqual(2048, self.wavefront.count)
        self.assertAlmostEqual(self.unit_sphere_volume_sequence[4], self.wavefront.volume)

    def then_each_triangle_in_wavefront_has_refined_diameter(self):
        for triangle in self.wavefront:
            self.assertLess(triangle.diameter, self.accepted_diameter)

    def test_wavefront_refine_with_orifice_filter(self):
        self.when_wavefront_set_to_unit_sphere()
        self.when_wavefront_is_refined_by_orifice()
        expected_volume = self.orifice_diameter ** 2
        self.assertAlmostEqual(expected_volume, self.wavefront.volume, 3)
        self.then_wavefront_points_up()

    def when_wavefront_is_refined_by_orifice(self):
        def testfilter(triangle):
            orifice_center = Ray.up.endpoint.coordinates
            orifice_diameter = self.orifice_diameter

            vertex_a = triangle.a.endpoint.coordinates
            vertex_b = triangle.b.endpoint.coordinates
            vertex_c = triangle.c.endpoint.coordinates

            a_length = vector_distance(vertex_a, orifice_center)
            b_length = vector_distance(vertex_b, orifice_center)
            c_length = vector_distance(vertex_c, orifice_center)

            farthest_distance = max(a_length, b_length, c_length)
            closest_distance = min(a_length, b_length, c_length)

            if farthest_distance <= orifice_diameter:
                return TriangleClassification.Accept
            elif triangle.diameter < 0.01 or closest_distance >= orifice_diameter:
                return TriangleClassification.Reject
            else:
                return TriangleClassification.Refine

        self.wavefront = self.wavefront.refine_with_filter(testfilter)

    def then_wavefront_points_up(self):
        actual_direction = self.wavefront.central_direction.normalized()
        expected_direction = Direction.up
        disparity = vector_distance(actual_direction.xyz, expected_direction.xyz)
        self.assertLess(disparity, 0.00000001)


class TestTriangleClassification(unittest.TestCase):
    """
    Test TriangleClassification class
    """

    def test_actions(self):
        actions = TriangleClassification.Actions
        self.assertEqual(3, len(actions))
        self.assertTrue(TriangleClassification.Accept in actions)
        self.assertTrue(TriangleClassification.Reject in actions)
        self.assertTrue(TriangleClassification.Refine in actions)

    def test_action_names(self):
        names = TriangleClassification.Names
        self.assertEqual(3, len(names))
        self.assertTrue("accepted" in names)
        self.assertTrue("rejected" in names)
        self.assertTrue("refined" in names)

    def test_action_by_name(self):
        self.assertEqual(TriangleClassification.Accept, TriangleClassification.by_name("accepted"))
        self.assertEqual(TriangleClassification.Reject, TriangleClassification.by_name("rejected"))
        self.assertEqual(TriangleClassification.Refine, TriangleClassification.by_name("refined"))

    def test_name_of_action(self):
        self.assertEqual("accepted", TriangleClassification.name_of(TriangleClassification.Accept))
        self.assertEqual("rejected", TriangleClassification.name_of(TriangleClassification.Reject))
        self.assertEqual("refined", TriangleClassification.name_of(TriangleClassification.Refine))


class TestFilterMonitor(unittest.TestCase):
    """
    Test Filter Monitor class
    """

    def clean_state(self):
        self.monitor = None

    def setUp(self):
        self.clean_state()
        self.monitor = FilterMonitor()

    def tearDown(self):
        self.clean_state()

    def test_empty_monitor(self):
        # when_nothing_added
        self.then_all_statistics_are_empty()

    def then_all_statistics_are_empty(self):
        self.then_total_count_is(0)

    def then_total_count_is(self, expected_count):
        self.assertEqual(expected_count, self.monitor.count)

    def test_monitor_accepts(self):
        self.when_triangles_are_accepted()
        self.then_total_count_is(8)
        self.then_accepted_triangle_count_is_correct(8)
        self.then_rejected_triangle_count_is_correct(0)
        self.then_refined_triangle_count_is_correct(0)

    def when_triangles_are_accepted(self, reason=None):
        for triangle in WaveFront.unit_sphere:
            self.monitor.accept(triangle, reason)

    def then_accepted_triangle_count_is_correct(self, expected_count):
        self.assertEqual(expected_count, self.monitor.accept_count)
        self.then_count_by_action_and_reason_is_correct(expected_count, TriangleClassification.Accept)

    def then_rejected_triangle_count_is_correct(self, expected_count):
        self.assertEqual(expected_count, self.monitor.reject_count)
        self.then_count_by_action_and_reason_is_correct(expected_count, TriangleClassification.Reject)

    def then_refined_triangle_count_is_correct(self, expected_count):
        self.assertEqual(expected_count, self.monitor.refine_count)
        self.then_count_by_action_and_reason_is_correct(expected_count, TriangleClassification.Refine)

    def test_monitor_rejects(self):
        self.when_triangles_are_rejected()
        self.then_total_count_is(8)
        self.then_accepted_triangle_count_is_correct(0)
        self.then_rejected_triangle_count_is_correct(8)
        self.then_refined_triangle_count_is_correct(0)

    def when_triangles_are_rejected(self, reason=None):
        for triangle in WaveFront.unit_sphere:
            self.monitor.reject(triangle, reason)

    def test_monitor_refines(self):
        self.when_triangles_are_refined()
        self.then_total_count_is(8)
        self.then_accepted_triangle_count_is_correct(0)
        self.then_rejected_triangle_count_is_correct(0)
        self.then_refined_triangle_count_is_correct(8)

    def when_triangles_are_refined(self, reason=None):
        for triangle in WaveFront.unit_sphere:
            self.monitor.refine(triangle, reason)

    def test_monitor_works_with_multiple_additions(self):
        self.when_triangles_are_accepted("alpha")
        self.when_triangles_are_accepted("beta")
        self.when_triangles_are_accepted("gamma")
        self.when_triangles_are_refined()
        self.when_triangles_are_rejected("delta")
        self.when_triangles_are_accepted()
        self.when_triangles_are_accepted("beta")
        self.when_triangles_are_refined()
        self.when_triangles_are_rejected()
        self.when_triangles_are_refined()
        self.then_accepted_triangle_count_is_correct(5 * 8)
        self.then_rejected_triangle_count_is_correct(2 * 8)
        self.then_refined_triangle_count_is_correct(3 * 8)
        self.then_total_count_is((5 + 2 + 3) * 8)
        self.then_count_by_action_and_reason_is_correct(
            1 * 8, TriangleClassification.Accept, "alpha")
        self.then_count_by_action_and_reason_is_correct(
            2 * 8, TriangleClassification.Accept, "beta")
        self.then_count_by_action_and_reason_is_correct(
            1 * 8, TriangleClassification.Accept, "gamma")
        self.then_count_by_action_and_reason_is_correct(
            1 * 8, TriangleClassification.Reject, "delta")
        self.assertTrue("alpha" in self.monitor.reasons)

    def then_count_by_action_and_reason_is_correct(self, expected_count, action=None, reason=None):
        self.assertEqual(expected_count, self.monitor.action_count(action, reason))


if __name__ == '__main__':
    unittest.main()
