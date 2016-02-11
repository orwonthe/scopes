# Copyright(c) 2014 WillyMillsLLC
import unittest
from numpy import matrix
import math

from optical.light import Position
from optical.light import Direction
from optical.light import Ray
from optical.light import Plane
from optical.transformations import Transformation

class  TestTransformation(unittest.TestCase):

    def clean_state(self):
        self.start_object = None
        self.stop_object = None
        self.transformation_under_test = None
        self.current_transformation = None
        self.start_position_coordinates = None
        self.stop_position_coordinates = None
        self.start_direction_coordinates = None
        self.stop_direction_coordinates = None
        self.start_position = None
        self.stop_position = None
        self.start_direction = None
        self.stop_direction = None
        self.start_ray = None
        self.stop_ray = None
        self.initial_object = None
        self.actual_object = None
        self.expected_object = None
        self.initial_coordinates = None
        self.actual_coordinates = None
        self.expected_coordinates = None
        self.direction_context = ""
        self.transformation_context = ""
        self.target_context = ""
        
    def setUp(self):
        self.clean_state()
        # Any explicit failure message passed in to the assert methods 
        # will be appended to the end of the normal failure message.
        self.longMessage = True
        
    def tearDown(self):
        self.clean_state()
        
    @property
    def is_covariant_test(self):
        return self.initial_object.is_covariant

    @property
    # Composes message used in asserts.
    # Each part of a specific test (transform, target, direction)
    # sets a piece of the context providing a sensible message.    
    def context(self):
        return (
            self.direction_context +
            " " + self.transformation_context +
            " of " + self.target_context
            )
            
    def test_transform_creation_requires_4_by_4_matrix(self):
        with self.assertRaises(Exception):
            Transformation("stuff")
        with self.assertRaises(TypeError):
            Transformation((1,2,3))

    def test_transform_creation_requires_non_projection(self):
        with self.assertRaises(ValueError):
            Transformation((
                (1, 2, 3, 4),
                (5, 6, 7, 8),
                (9, 10, 11, 12),
                (13, 14, 15, 16)
                ))
        with self.assertRaises(ValueError):
            Transformation((
                (1.1, 2, 3, 4),
                (0.0, 6, 7, 8),
                (0.0, 10, 11, 12),
                (0.0, 14, 15, 16)
                ))
        with self.assertRaises(ValueError):
            Transformation((
                (1.0, 2, 3, 4),
                (0.1, 6, 7, 8),
                (0.0, 10, 11, 12),
                (0.0, 14, 15, 16)
                ))
        with self.assertRaises(ValueError):
            Transformation((
                (1.0, 2, 3, 4),
                (0.0, 6, 7, 8),
                (0.1, 10, 11, 12),
                (0.0, 14, 15, 16)
                ))
        with self.assertRaises(ValueError):
            Transformation((
                (1.0, 2, 3, 4),
                (0.0, 6, 7, 8),
                (0.0, 10, 11, 12),
                (0.1, 14, 15, 16)
                ))

    def test_transform_creation_accepts_non_projection(self):
        four_by_four = (
            (1.0, 2, 3, 4),
            (0.0, 6, 7, 8),
            (0.0, 10, 11, 12),
            (0.0, 14, 15, 16)
            ) 
        Transformation(four_by_four)
        as_matrix = matrix(four_by_four)
        Transformation(as_matrix)
        as_another_matrix = matrix(as_matrix)
        Transformation(as_another_matrix)
    
    # Tests single specific combination or transformation, target, and direction
    def test_forward_translation_of_position(self):
        self.initialize_translation()
        self.generate_start_and_stop_objects_as_position()
        self.set_forward_transformation()
        self.then_object_transforms_correctly()
    
    # Tests single specific combination or transformation, target, and direction
    def test_reverse_translation_of_position(self):
        self.initialize_translation()
        self.generate_start_and_stop_objects_as_position()
        self.set_reverse_transformation()
        self.then_object_transforms_correctly()
    
    # Tests every combination of transformation, target, and direction
    def test_various_combinations(self):
        self.then_test_multiple_transformations_and_targets_forward_and_reverse()       
    def then_test_multiple_transformations_and_targets_forward_and_reverse(self):
        transformation_initializer_list = [
            self.initialize_translation,
            self.initialize_rotation_in_x,
            self.initialize_rotation_in_y,
            self.initialize_rotation_in_z,
            self.initialize_composite_transformation,
            ]
        for transformation_initializer in transformation_initializer_list:
            self.clean_state() # prevent cross talk between combinations
            transformation_initializer()
            self.then_test_multiple_target_objects_forward_and_reverse()    
    def then_test_multiple_target_objects_forward_and_reverse(self):
        # With current transformation test all combinations of target and direction
        target_generator_list =  [
            self.generate_start_and_stop_objects_as_position,
            self.generate_start_and_stop_objects_as_direction,
            self.generate_start_and_stop_objects_as_plane
            ]
        for target_generator in target_generator_list: 
            target_generator()
            self.then_test_forward_and_reverse_transformations_of_object()            
    def then_test_forward_and_reverse_transformations_of_object(self):
        # With current transformation and target test both directions
        direction_list = [
            self.set_forward_transformation,  
            self.set_reverse_transformation
            ]
        for direction_of_transformation in direction_list:
            direction_of_transformation()
            self.then_test_object_in_current_direction_with_current_transformation()    
    def then_test_object_in_current_direction_with_current_transformation(self):
        # Transformation, target, and direction are now set.
        # Test that specific combination.
        self.then_object_transforms_correctly()            
        
    def initialize_translation(self):
        self.transformation_context = "translation"
        self.start_position_coordinates =  (1.0, 20.0, 30.0, 40.0)
        self.stop_position_coordinates = (1.0, 22.0, 33.0, 44.0)
        delta_position = Position.from_xyz((2.0, 3.0, 4.0))
        self.start_direction_coordinates = (0.0, math.sqrt(0.4), math.sqrt(0.5), math.sqrt(0.1))
        self.stop_direction_coordinates = self.start_direction_coordinates
        self.transformation_under_test = Transformation.translation_by_position(delta_position)
        
    def initialize_rotation_in_x(self):
        self.transformation_context = "rotation in x"
        angle = math.pi / 6.0
        cosine_of_angle = math.cos(angle)
        sine_of_angle = math.sin(angle)
        self.start_position_coordinates = (1.0, 100.0, 200.0, 300.0)
        self.stop_position_coordinates = (
            1.0,
            100.0,
            200.0 * cosine_of_angle - 300.0 * sine_of_angle, 
            300.0 * cosine_of_angle + 200.0 * sine_of_angle
            )
        x = math.sqrt(0.4)
        y = math.sqrt(0.5)
        z = math.sqrt(0.1)
        self.start_direction_coordinates = (0.0, x, y, z)
        self.stop_direction_coordinates = (
            0.0, 
            x, 
            y * cosine_of_angle - z * sine_of_angle, 
            z * cosine_of_angle + y * sine_of_angle
            )
        self.transformation_under_test = Transformation.rotation_about_x_axis(angle)

    def initialize_rotation_in_y(self):
        self.transformation_context = "rotation in y"
        angle = math.pi / 6.0
        cosine_of_angle = math.cos(angle)
        sine_of_angle = math.sin(angle)
        self.start_position_coordinates = (1.0, 100.0, 200.0, 300.0)
        self.stop_position_coordinates = (
            1.0,
            100.0 * cosine_of_angle + 300.0 * sine_of_angle,
            200,
            300.0 * cosine_of_angle - 100.0 * sine_of_angle, 
            )
        x = math.sqrt(0.4)
        y = math.sqrt(0.5)
        z = math.sqrt(0.1)
        self.start_direction_coordinates = (0.0, x, y, z)
        self.stop_direction_coordinates = (
            0.0, 
            x * cosine_of_angle + z * sine_of_angle, 
            y,
            z * cosine_of_angle - x * sine_of_angle
            )
        self.transformation_under_test = Transformation.rotation_about_y_axis(angle)

    def initialize_rotation_in_z(self):
        self.transformation_context = "rotation in z"
        angle = math.pi / 6.0
        cosine_of_angle = math.cos(angle)
        sine_of_angle = math.sin(angle)
        self.start_position_coordinates = (1.0, 100.0, 200.0, 300.0)
        self.stop_position_coordinates = (
            1.0,
            100.0 * cosine_of_angle - 200.0 * sine_of_angle, 
            200.0 * cosine_of_angle + 100.0 * sine_of_angle, 
            300.0)
        x = math.sqrt(0.4)
        y = math.sqrt(0.5)
        z = math.sqrt(0.1)
        self.start_direction_coordinates = (0.0, x, y, z)
        self.stop_direction_coordinates = (
            0.0, 
            x * cosine_of_angle - y * sine_of_angle, 
            y * cosine_of_angle + x * sine_of_angle, 
            z)
        self.transformation_under_test = Transformation.rotation_about_z_axis(angle)

    def initialize_composite_transformation(self):
        self.transformation_context = "composite transformation"
        right_angle = math.pi / 2.0
        self.start_position_coordinates = (1.0, 100.0, 200.0, 300.0)
        delta_position = Position.from_xyz((2.0, 3.0, 4.0))
        self.stop_position_coordinates = (1.0, 297.0, 102.0, 204.0)
        self.start_direction_coordinates = (0.0, math.sqrt(0.4), math.sqrt(0.5), math.sqrt(0.1))
        self.stop_direction_coordinates = (0.0, math.sqrt(0.1), math.sqrt(0.4), math.sqrt(0.5))
        first_transform = Transformation.rotation_about_x_axis(right_angle)
        second_transform = Transformation.translation_by_position(delta_position)
        third_transform = Transformation.rotation_about_z_axis(right_angle)
        self.transformation_under_test = first_transform.compose(
            second_transform).compose(
                third_transform)
                
    def generate_start_and_stop_objects_as_position(self):
        self.target_context = "Position"
        self.generate_start_and_stop_position_direction_and_ray()
        self.start_object = self.start_position
        self.stop_object =  self.stop_position
    def generate_start_and_stop_position_direction_and_ray(self):
        self.start_position = Position(self.start_position_coordinates)
        self.stop_position = Position(self.stop_position_coordinates)

        self.start_direction = Direction(self.start_direction_coordinates)
        self.stop_direction = Direction(self.stop_direction_coordinates)

        self.start_ray = Ray(self.start_position, self.start_direction)
        self.stop_ray = Ray(self.stop_position, self.stop_direction)
        
    def generate_start_and_stop_objects_as_direction(self):
        self.target_context = "Direction"
        self.generate_start_and_stop_position_direction_and_ray()
        self.start_object = self.start_direction
        self.stop_object = self.stop_direction

    def generate_start_and_stop_objects_as_plane(self):
        self.target_context = "Plane"        
        self.generate_start_and_stop_position_direction_and_ray()
        self.start_object = Plane.orthogonal_to_ray(self.start_ray)
        self.stop_object =  Plane.orthogonal_to_ray(self.stop_ray)

    def set_forward_transformation(self):
        self.direction_context = "forward"
        self.initial_object = self.start_object
        self.expected_object = self.stop_object
        self.current_transformation = self.transformation_under_test

    def set_reverse_transformation(self):
        self.direction_context = "reverse "
        self.initial_object = self.stop_object
        self.expected_object = self.start_object
        self.current_transformation = self.transformation_under_test.inverse()
        
    def then_object_transforms_correctly(self):
        self.when_expected_coordinates_are_set()
        self.then_transformation_meets_expectations()
        self.then_object_transforms()
    def when_expected_coordinates_are_set(self):
        self.expected_coordinates = self.expected_object.coordinates
        
    def then_transformation_meets_expectations(self):
        self.when_object_is_transformed()
        self.then_expected_coordinates_match_actual_coordinates()
        
    def when_object_is_transformed(self):
        self.when_initial_coordinates_are_set_from_initial_object()
        self.when_actual_coordinates_are_transformed_from_initial_coordinates()
    def when_initial_coordinates_are_set_from_initial_object(self):
        self.initial_coordinates = self.initial_object.coordinates
    def when_actual_coordinates_are_transformed_from_initial_coordinates(self):
        if self.is_covariant_test:
            self.when_actual_coordinates_are_covariantly_transformed_from_initial_coordinates()
        else:
            self.when_actual_coordinates_are_contravariantly_transformed_from_initial_coordinates()
    def when_actual_coordinates_are_covariantly_transformed_from_initial_coordinates(self):
        self.actual_coordinates = self.current_transformation.covariant_transform(self.initial_coordinates)
    def when_actual_coordinates_are_contravariantly_transformed_from_initial_coordinates(self):
        self.actual_coordinates = self.current_transformation.contravariant_transform(self.initial_coordinates)
    
    def then_object_transforms(self):
        self.when_initial_object_is_transformed()
        self.then_expected_object_matches_actual_object()
    
    def when_initial_object_is_transformed(self):        
        self.actual_object = self.initial_object.transform(self.current_transformation)
    
    def then_expected_object_matches_actual_object(self):
        self.when_expected_coordinates_are_set()
        self.when_actual_coordinates_are_set()
        self.then_expected_coordinates_match_actual_coordinates()
    def when_actual_coordinates_are_set(self):
        self.actual_coordinates = self.actual_object.coordinates
    def then_expected_coordinates_match_actual_coordinates(self):
        self.then_expected_coordinates_have_correct_length()
        self.then_expected_coordinates_have_correct_values()      
    def then_expected_coordinates_have_correct_length(self):
        # Hard check for length of 4 could be loosened to check mutual equality
        self.assertEqual(4, len(self.expected_coordinates), msg = self.context)
        self.assertEqual(4, len(self.actual_coordinates), msg = self.context)
    def then_expected_coordinates_have_correct_values(self):
        # this gives a friendlier error message than does assertTupleEqual
        for i, expected in enumerate(self.expected_coordinates):
            self.assertAlmostEqual(expected, self.actual_coordinates[i], msg = self.context)
          
                        
    
if __name__ == '__main__':
    unittest.main()

