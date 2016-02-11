# Copyright(c) 2014 WillyMillsLLC

import operator
from collections import deque
from math import sqrt
from math import fabs
from numpy import array

# Design rationale
#
# Purpose is to reliably construct and analyze optical instruments.
# Evaluation, for a given design, occurs once.
# So performance is not a big issue.
#
# Immutable objects are used in preference to state changing objects.
# Immutability results in heavy use of @property's and rather
# more construction validation than the usual low friction pythonesque style.
#

def column_of_matrix_as_tuple(target_matrix, column_index = 0):
    return tuple(array(target_matrix)[:,column_index])

def row_of_matrix_as_tuple(target_matrix, row_index = 0):
    return tuple(array(target_matrix)[row_index])

def vector_add(this_vector, that_vector):
    return tuple(map(operator.add, this_vector, that_vector))

def vector_difference(this_vector, that_vector):
    return tuple(map(operator.sub, this_vector, that_vector))

def vector_distance(this_vector, that_vector):
    difference_vector = vector_difference(this_vector, that_vector)
    return vector_norm(difference_vector)

def vector_distance_squared(this_vector, that_vector):
    difference_vector = vector_difference(this_vector, that_vector)
    return vector_norm_squared(difference_vector)

def vector_dot_product(this_vector, that_vector):
    return sum(map(operator.mul, this_vector, that_vector))

def vector_interpolation(this_vector, that_vector, this_weight = 0.5):
    that_weight = 1.0 - this_weight
    return vector_add(
        vector_scale(this_vector, this_weight), 
        vector_scale(that_vector, that_weight))

def vector_norm(vector):
    return sqrt(vector_dot_product(vector, vector))

def vector_norm_squared(vector):
    return vector_dot_product(vector, vector)

def vector_scale(vector, scale):
    return tuple(scale * value for value in vector)

class Projector(object):
    """
    Immutable point in 4d projective space represents 3d position or direction
    
    Uses tuple rather than lists or numpy matrix or array.
    Use of tuple makes these low level objects slightly more complex, maybe slower.
    But using tuples enforces immutability.
    Immutability is a big win at the application level:
        Uexpected mutation bugs are subtle and hard to find
        Testing is simplified by lack of state change
        Lack of state change allows easier lazy initialization (for performance)
    """
    
    def __init__(self, wxyz_tuple):
        if not isinstance(wxyz_tuple, tuple):
            wxyz_tuple = tuple(wxyz_tuple)
        if len(wxyz_tuple) != 4:
            raise ValueError("tuple dimension must be 4")
        self._coordinates = wxyz_tuple
    
    def __iter__(self):
        return iter(self._coordinates)
    
    def __repr__(self):
        return "Projector(%r)" % [self._coordinates]

    def __str__(self):
        return  "%r" % [self._coordinates]

    @property
    def coordinates(self):
        return self._coordinates
    
    @property
    def is_position(self):
        return 1.0 == self.w
        
    @property
    def is_direction(self):
        return 0.0 == self.w
    
    @property
    # 4th dimension: zero for directions and one for positions
    def w(self):
        return self._coordinates[0] 
    
    @property
    def x(self):
        return self._coordinates[1] 
    
    @property
    def y(self):
        return self._coordinates[2] 
    
    @property
    def z(self):
        return self._coordinates[3] 
    
    @property
    def xyz(self):
        return self._coordinates[1:]
    
    def clone_from_coordinates(self, clone_coordinates):
        return self.__class__(clone_coordinates)        
    
    def dot(self, other):
        return vector_dot_product(self, other)
    
    def transform(self, transformation):
        # note coordinate_transform must be defined in derived class
        transformed_coordinates = self.coordinate_transform(transformation)
        return self.clone_from_coordinates(transformed_coordinates)
    
    def interpolate_with(self, other, self_weight = 0.5):
        interpolated_coordinates = vector_interpolation(
            self.coordinates, other.coordinates, self_weight)
        return self.clone_from_coordinates(interpolated_coordinates)
        

class CovariantProjector(Projector):
    """
    Immutable point in 4d projective space with covariant semantics
    
    Used in representation of planes or other objects whose
    measurment values grow (covariation) when measurement
    units grow  (e.g 4 per centimeter --> 400 per meter)
    """
    def __init__(self, wxyz_tuple):
        super(CovariantProjector, self).__init__(wxyz_tuple)
        
    @property
    def is_covariant(self):
        return True
        
    def coordinate_transform(self, transformation):
        return transformation.covariant_transform(self.coordinates)
        
class ContravariantProjector(Projector):
    """
    Immutable point in 4d projective space with contravariant semantics
    
    Used in reprentation of position or direction or other objects
    whose measurement values shrink (contravariation) when measurement
    units grow (e.g 400 centimeters --> 4 meters)
    """
    def __init__(self, wxyz_tuple):
        super(ContravariantProjector, self).__init__(wxyz_tuple)
        
    @property
    def is_covariant(self):
        return False
    
    def add(self, addend):
        new_location = tuple(map(operator.add, self, addend))
        return self.clone_from_coordinates(new_location)
            
    def coordinate_transform(self, transformation):
        return transformation.contravariant_transform(self.coordinates)

        
class Position(ContravariantProjector):
    """
    Immutable point in 4d projective space represents 3d position
    """
        
    def __init__(self, wxyz_tuple):
        if not isinstance(wxyz_tuple, tuple):
            raise TypeError("tuple required")
        if len(wxyz_tuple) != 4:
            raise ValueError("tuple dimension must be 4")
        if wxyz_tuple[0] not in (0.0, 1.0) :
            raise ValueError("projective coordinate must be zero or one")
        super(Position, self).__init__(wxyz_tuple)

    @classmethod
    def from_xyz(cls, xyz_three_tuple):
        wxyz_tuple = (1.0,) + xyz_three_tuple
        return cls(wxyz_tuple)
    
    def difference(self, other):
        delta_xyz = vector_difference(self.xyz, other.xyz)
        if other.is_direction:
            return Position.from_xyz(delta_xyz)
        return Direction.from_xyz(delta_xyz)
    
    def direction_from_origin(self):
        return Direction.from_xyz(self.xyz)
            
Position.origin = Position.from_xyz((0.0, 0.0, 0.0))


class Direction(ContravariantProjector):
    """
    Immutable point in 4d projective space represents 3d direction
    """
    
    cardinal_xyz_values = {
            'up': (0.0, 0.0, 1.0),
            'down': (0.0, 0.0, -1.0),
            'left': (-1.0, 0.0, 0.0),
            'right': (1.0, 0.0, 0.0),
            'forward': (0.0, 1.0, 0.0),
            'backward': (0.0, -1.0, 0.0),
            'unknown': (0.0, 0.0, 0.0)
            }
    cardinal_names = cardinal_xyz_values.keys()
    
    def __init__(self, wxyz_tuple):
        if not isinstance(wxyz_tuple, tuple):
            raise TypeError("tuple required")
        if len(wxyz_tuple) != 4:
            raise ValueError("tuple dimension must be 4")
        if 0.0 != wxyz_tuple[0]:
            raise ValueError("projective coordinate must be zero")
        super(Direction, self).__init__(wxyz_tuple)

    @classmethod
    def create_cardinals(cls):
        for direction_name in cls.cardinal_names:
            cls.create_cardinal_direction(direction_name)
    @classmethod
    def create_cardinal_direction(cls, direction_name):
        xyz = cls.cardinal_xyz_values[direction_name]
        direction = cls.from_xyz(xyz)
        setattr(cls, direction_name, direction)
        
    @classmethod
    def get_named_cardinal_direction(cls, direction_name):
        return getattr(cls, direction_name)
        
    @classmethod
    def from_xyz(cls, xyz_tuple):
        wxyz_tuple = (0.0,) + xyz_tuple
        return cls(wxyz_tuple)

    def clone_from_xyz(self, xyz_tuple):
        return self.__class__.from_xyz(xyz_tuple)
    
    def cross(self, direction):
        x = self.y * direction.z - self.z * direction.y
        y = self.z * direction.x - self.x * direction.z
        z = self.x * direction.y - self.y * direction.x
        return self.clone_from_xyz((x, y, z))
    
    def difference(self, other):
        delta_xyz = vector_difference(self.xyz, other.xyz)
        return Direction.from_xyz(delta_xyz)
    
    def length(self):
        return vector_norm(self)
    
    def length_squared(self):
        return vector_norm_squared(self)
    
    def normalized(self):
        the_length = self.length()
        if the_length == 0.0 or the_length == 1.0:
            return self
        return self.scale(1.0 / the_length)
    
    def scale(self, scale_factor):
        if scale_factor == 1.0:
            return self
        scaled_coordinates = tuple(scale_factor * value for value in self)
        return self.clone_from_coordinates(scaled_coordinates)
    
    def spherical_interpolate_with(self, other, self_weight = 0.5):        
        other_weight = 1.0 -self_weight
        raw_result = self.interpolate_with(other, self_weight)
        raw_length = raw_result.length()
        self_length = self.length()
        other_length = other.length()
        desired_length = self_weight * self_length + other_weight * other_length
        scale_factor = desired_length / raw_length
        scaled_result = raw_result.scale(scale_factor)
        return scaled_result
    
Direction.create_cardinals()

class Ray(object):
    """
    Immutable object with 3d position and direction
    """
    
    def __init__(self, position, direction):
        if not isinstance(position, Position):
            raise TypeError("Position required")
        if not isinstance(direction, Direction):
            raise TypeError("Direction required")
        self._position = position
        self._direction = direction
        
    def __str__(self):
        return  "position%s direction%s" % (self.position.xyz, self.direction.xyz)
    def __repr__(self):
        return  "position%s direction%s" % (self.position, self.direction)

    @classmethod
    def create_all_of_the_cardinals(cls):
        for direction_name in Direction.cardinal_names:
            cardinal_ray = cls.create_named_cardinal_ray(direction_name)
            cls.set_named_cardinal_ray(direction_name, cardinal_ray)
    @classmethod
    def create_named_cardinal_ray(cls, direction_name):
        position = Position.origin
        direction = Direction.get_named_cardinal_direction(direction_name)
        cardinal_ray = cls(position, direction)
        return cardinal_ray
    @classmethod
    def set_named_cardinal_ray(cls, direction_name, cardinal_ray):
        setattr(cls, direction_name, cardinal_ray) 

    @classmethod
    def from_origin_endpoint(cls, origin, endpoint):
        return Ray(origin, endpoint.difference(origin))
    
    @classmethod
    def get_named_cardinal_ray(cls, direction_name):
        return getattr(cls, direction_name)

    @property
    def position(self):
        return self._position
    
    @property
    def direction(self):
        return self._direction
    
    @property
    def endpoint(self):
        return self.scaled_endpoint()
            
    def add(self, direction):
        return Ray(self.position.add(direction), self.direction)
    
    def distance_to_point(self, point):
        orthogonal_plane = Plane.orthogonal_to_ray(self)
        distance_to_plane = orthogonal_plane.distance_to_point(point)
        distance_to_plane_squared = distance_to_plane * distance_to_plane
        
        direction_to_point = point.difference(self.position)
        hypotonuse_squared = direction_to_point.length_squared()
        
        # use of absolute value protects against negative numbers near zero
        distance_squared = fabs(hypotonuse_squared - distance_to_plane_squared)
        distance = sqrt(distance_squared)
        return distance        
                
    def interpolate_with(self, other, weight = 0.5):
        interpolated_position = self.position.interpolate_with(other.position, weight)
        interpolated_direction = self.direction.interpolate_with(other.direction, weight)
        return Ray(interpolated_position, interpolated_direction)

    def normalized(self):
        return Ray(self.position, self.direction.normalized())

    def scaled_direction(self, scale_factor):
        return Ray(self.position, self.direction.scale(scale_factor))

    def scaled_endpoint(self, scale_factor = 1.0):
        return self.position.add(self.direction.scale(scale_factor))

    def spherical_interpolate_with(self, other, weight = 0.5):
        interpolated_position = self.position.interpolate_with(other.position, weight)
        interpolated_direction = self.direction.spherical_interpolate_with(other.direction, weight)
        return Ray(interpolated_position, interpolated_direction)
    
    def transform(self, transformation):
        transformed_position = self.position.transform(transformation)
        transformed_direction = self.direction.transform(transformation)
        return Ray(transformed_position, transformed_direction)

    
Ray.create_all_of_the_cardinals()


class Plane(CovariantProjector):
    """
    Immutable 3d plane
    """
    
    def __init__(self, four_tuple):
        super(Plane, self).__init__(four_tuple)
    
    @classmethod
    def orthogonal_to_ray(cls, ray):
        """
        Create a plane orthoganal to a given ray
        """
        position = ray.position
        direction = ray.direction.normalized()
        offset = direction.dot(position)
        projection = (-offset,) + direction.xyz
        return Plane(projection)
    
    def distance_to_point(self, point):
        return self.dot(point)
    
    def intersect(self, ray):
        origin_distance_from_plane = self.distance_to_point(ray.position)
        if origin_distance_from_plane == 0.0:
            return ray.position
        endpoint_distance_from_plane = self.distance_to_point(ray.endpoint)
        delta_distance = origin_distance_from_plane - endpoint_distance_from_plane
        if delta_distance == 0.0:
            return ray.direction # intersects at infinity
        scale_factor = origin_distance_from_plane / delta_distance
        intersect_point = ray.scaled_endpoint(scale_factor)
        return intersect_point

class Sphere(object):
    """
    Sphere in 3 space
    """
    def __init__(self, center_position, radius):
        self.ray = Ray(center_position, Direction.up.scale(radius))
        
    @property
    def center_position(self):
        return self.ray.position
    
    @property
    def radius(self):
        return self.ray.direction.length()
    
    @property
    def x(self):
        return self.ray.position.x
    
    @property
    def y(self):
        return self.ray.position.y
    
    @property
    def z(self):
        return self.ray.position.z
    
    def elevation(self, point):
        return self.ray.position.difference(point).length() - self.radius
    
    def intersection_points(self, ray):
        ray_distance_from_sphere_center = ray.distance_to_point(self.center_position)
        discrepancy_squared = (self.radius * self.radius 
            - ray_distance_from_sphere_center * ray_distance_from_sphere_center)
        if discrepancy_squared < 0.0:
            return []
        ray_normal = Ray(self.center_position, ray.direction.normalized())
        plane_normal = Plane.orthogonal_to_ray(ray_normal)
        ray_intersects_plane_normal = plane_normal.intersect(ray)
        discrepancy = sqrt(discrepancy_squared)
        if discrepancy <= 0.0:
            return ray_intersects_plane_normal
        direction_normal = ray_normal.direction
        forward_direction = direction_normal.scale(discrepancy)
        backward_direction = direction_normal.scale(-discrepancy)
        forward_point = ray_intersects_plane_normal.add(forward_direction)
        backward_point = ray_intersects_plane_normal.add(backward_direction)
        return [forward_point, backward_point]
    
class WaveStatistics(object):
    """
    Acquires and calculates statistics about a wavefront
    """
    def __init__(self, volume = 0.0, direction = Direction.unknown):
        self.volume = volume
        self._weighted_direction = direction.scale(volume)
    
    @classmethod
    def accumulator(cls, iteration):
        wave_statistic = cls()
        for item in iteration:
            wave_statistic.add(item)
        return wave_statistic

    @property
    def central_direction(self):
        if self.volume == 0.0:
            return Direction.unknown
        return self._weighted_direction.scale(1.0 / self.volume)
    
    @property
    def normalized_direction(self):
        return self.central_direction.normalized()
    
    @property
    def volume_weighted_direction(self):
        return self._weighted_direction
    
    def add(self, other):
        self.volume += other.volume
        self._weighted_direction = self._weighted_direction.add(
            other.central_direction.scale(other.volume))


class SphericalTriangle(object):
    """
    Immutable spherical triangle with three Ray vertices
    
    Spherical nature of triangle manifests when interpolating along edges.
    Interpolation preserves distance from origin, hence it curves
    Volume calculation is approximate when vertices do not share origin
    """
    
    def __init__(self, vertices):
        if not isinstance(vertices, tuple):
            raise TypeError("tuple required")
        if len(vertices) != 3:
            raise ValueError("three vertices required")
        for vertex in vertices:
            if not isinstance(vertex, Ray):
                raise TypeError("Ray required")                
        self._vertices = vertices
    
    def __iter__(self):
        return iter(self._vertices)

    @property
    def a(self):
        return self._vertices[0]
    
    @property
    def b(self):
        return self._vertices[1]
    
    @property
    def c(self):
        return self._vertices[2]
    
    @property
    def center_position(self):
        position_b_mid_c = self.b.position.interpolate_with(self.c.position)
        position =  self.a.position.interpolate_with(position_b_mid_c, 1.0 / 3.0)
        return position

    @property
    def center_ray(self):
        return Ray(self.center_position, self.central_direction)
    
    @property
    def central_direction(self):
        direction_a = self.a.direction
        direction_b = self.b.direction
        direction_c = self.c.direction
        sum_abc = direction_a.add(direction_b).add(direction_c)
        length_a = direction_a.length()
        length_b = direction_b.length()
        length_c = direction_c.length()
        scale_factor = (length_a + length_b + length_c) / (3.0 * sum_abc.length())
        scaled_center = sum_abc.scale(scale_factor)
        return scaled_center
    
    @property
    def diameter(self):
        vertex_a = self.a.endpoint.coordinates
        vertex_b = self.b.endpoint.coordinates
        vertex_c = self.c.endpoint.coordinates
        ab_length = vector_distance(vertex_a, vertex_b)
        bc_length = vector_distance(vertex_b, vertex_c)
        ca_length = vector_distance(vertex_c, vertex_a)
        return max(ab_length, bc_length, ca_length)
    
    @property
    def directions(self):
        return tuple(vertex.direction for vertex in self.vertices)
    
    @property
    def has_positive_orientation(self):
        return self.volume > 0.0
    
    @property
    def vertices(self):
        return self._vertices

    @property
    def volume(self):
        # correct for rays with coincident positions, approximate otherwise
        directions = self.directions
        return directions[0].cross(directions[1]).dot(directions[2]) / 6.0  
    
    def add(self, direction):
        return self.clone_from_vertices(tuple(ray.add(direction) for ray in self))
    
    def clone_from_vertices(self, vertices):
        return self.__class__(vertices)
    
    def opposite_orientation(self):
        return self.clone_from_vertices((self.c, self.b, self.a))
    
    def positive_orientation(self):
        if self.has_positive_orientation:
            return self
        else:
            return self.opposite_orientation()
            
    def subdivide(self):
        """ subdivide the triangle into 4 triangles, each with half sized edges
        """        
        # mid points of each edge
        a_mid_b = self.a.spherical_interpolate_with(self.b)
        b_mid_c = self.b.spherical_interpolate_with(self.c)
        c_mid_a = self.c.spherical_interpolate_with(self.a)
        # triangle connecting the midpoints
        interior_triangle = self.clone_from_vertices((a_mid_b, b_mid_c, c_mid_a))
        # triangles with one vertex common to original
        a_triangle = self.clone_from_vertices((self.a, a_mid_b, c_mid_a))
        b_triangle = self.clone_from_vertices((self.b, b_mid_c, a_mid_b))
        c_triangle = self.clone_from_vertices((self.c, c_mid_a, b_mid_c))
        return WaveFront((interior_triangle, a_triangle, b_triangle, c_triangle))

    def subdivision_iteration(self):
        triangle_list = self.subdivide()
        for triangle in triangle_list:
            yield triangle

    
class TriangleEvaluator(object):
    """
    Helper class for evaluating lists of triangles
    """
    def __init__(self, name, accumulator, metric):
        self.accumulator = accumulator
        self.metric = metric
        self.name = name
    
    def evaluate_triangle(self, triangle):
        return self.metric(triangle)
    
    def evaluate_triangle_iteration(self, triangle_iteration):
        return self.accumulator(self.metric(triangle) for triangle in triangle_iteration)

TriangleEvaluator.Counter = TriangleEvaluator(
        "count", sum, lambda t: 1)
TriangleEvaluator.TotalVolume = TriangleEvaluator(
        "volume", sum, lambda t: t.volume)
TriangleEvaluator.WaveStastics = TriangleEvaluator(
        "statistics", WaveStatistics.accumulator, lambda t:t)

class WaveFront(object):
    """
    Immutable mesh of spherical triangles representing wave front of light rays
    """
    def __init__(self, triangles):
        positive_triangle_list = []
        for triangle in triangles:
            if not isinstance(triangle, SphericalTriangle):
                raise TypeError("SphericalTriangle required") 
            positive_triangle_list.append(triangle.positive_orientation())
        self._triangles = tuple(positive_triangle_list)
        # setting up lazy initialization
        self._wave_statistics = None 
    
    def __iter__(self):
        return iter(self._triangles)

    @classmethod
    def create_unit_sphere(cls):
        # unit sphere from unit ocohedron of spherical triangles
        octohedron = [SphericalTriangle((x,y,z))
                for x in [Ray.right, Ray.left]
                for y in [Ray.forward, Ray.backward]
                for z in [Ray.up, Ray.down]
            ]
        cls.unit_sphere = WaveFront(octohedron)
        
    @property
    def count(self):
        return len(self.triangles)
    
    @property
    def triangles(self):
        return self._triangles
    
    @property
    def volume(self):
        # lazy initialization for performance reasons
        self.__check_statistics()
        return self._wave_statistics.volume
    def __check_statistics(self):
        if self._wave_statistics == None:
            self._wave_statistics = (
                TriangleEvaluator.WaveStastics.evaluate_triangle_iteration(self))
    
    @property
    def central_direction(self):
        # lazy initialization for performance reasons
        self.__check_statistics()
        return self._wave_statistics.central_direction            
    
    def add(self, direction):
        return self.clone_from_triangles([triangle.add(direction) for triangle in self])
    
    def clone_from_triangles(self, triangles):
        return self.__class__(triangles)
    
    def refine_with_filter_iteration(self, classification_filter, triangle_iterator = None):
        """
        Return iterartion of contents subdivided by means of classification filter
        
        Iteration progresses (recursively) until all triangles have been either
        accepted or rejected.
        Recursion occurs when a triangle is neither accepted nor rejected.
        These triangles are subdivided into subtriangles for further refinement.
        """
        
        # refinement starts with self content but recursing calls
        # provide the source content in the call
        if triangle_iterator == None:
            triangle_iterator = self
        
        # use queue to get breadth first expansion
        needs_filtering_queue = deque(triangle_iterator)
        while len(needs_filtering_queue) > 0:
            triangle = needs_filtering_queue.popleft()
            triangle_evaluation = classification_filter(triangle)
            if triangle_evaluation == TriangleClassification.Accept:
                yield triangle
            elif triangle_evaluation == TriangleClassification.Refine:
                needs_filtering_queue.extend(triangle.subdivide())
            # else reject by ignoring
                        
    def refine_with_filter(self, classification_filter, triangle_iterator = None):
        """
        Return WaveFront that has been refined using classification filter
        """
        list_of_subdivided_triangles = [triangle for triangle in self.refine_with_filter_iteration(classification_filter, triangle_iterator)]
        return self.clone_from_triangles(list_of_subdivided_triangles)

    def subdivide(self):
        """Returns subdivided mesh built from subdivided its contents
        """
        list_of_subdivided_triangles = [triangle for triangle in self.subdivision_iteration()]
        return WaveFront(list_of_subdivided_triangles)

    def subdivision_iteration(self):
        """Returns iteration of subdivided contents
        """
        for triangle in self:            
            for subtriangle in triangle.subdivide():
                yield subtriangle
                
WaveFront.create_unit_sphere()


class TriangleClassification:
    """
    Represents results of a triangle classification filter
          
    classification_filter(triangle) should return TriangleClassification:
        Reject : do not include this triangle
        Accepet : triangle is accepted as is
        Refine : triangle should be subdivided and each sub triangle reclassified
    
    """
    Reject, Refine, Accept = range(3)
    
    ActionNameList = {"accepted":Accept, "rejected":Reject, "refined":Refine}
    NameActionList = {value:key for key,value in ActionNameList.iteritems()}
    Actions = ActionNameList.values()
    Names = ActionNameList.keys()
    
    @classmethod
    def by_name(cls, name):
        return cls.ActionNameList[name]
    
    @classmethod
    def name_of(cls, action):
        return cls.NameActionList[action]
                

class BaseFilterMonitor(object):
    """
    Base class used to monitors WaveFront filtering
    """
    def accept(self, triangle, reason = None):
        return self.update(TriangleClassification.Accept, triangle, reason)
    
    def reject(self, triangle, reason = None):
        return self.update(TriangleClassification.Reject, triangle, reason)
        
    def refine(self, triangle, reason = None):
        return self.update(TriangleClassification.Refine, triangle, reason)


class NullFilterMonitor(BaseFilterMonitor):
    """
    Minimally intrusive monitor for WaveFront filtering retains no information
    """
    def update(self, action, triangle, reason = None):
        return action


class FilterMonitor(BaseFilterMonitor):
    """
    Sorts all of the results of a WaveFront filtering operation.
    
    Provides information on counts and volumes  by action and reason.
    Useful for providing information useful in crafting complex optical 
    interactions;
    """
    def __init__(self):        
        self.action_dictionary = {}
    
    def update(self, action, triangle, reason = None):
        if reason == None:
            reason = TriangleClassification.name_of(action)
        if not action in self.action_dictionary:
            self.action_dictionary[action] = {}
        reason_dictionary = self.action_dictionary[action]
        if not reason in reason_dictionary:
            reason_dictionary[reason] = []
        reason_list = reason_dictionary[reason]
        reason_list.append(triangle)
        return action
    
    @property
    def count(self):
        return self.action_count()
    
    @property
    def accept_count(self):
        return self.action_count(action = TriangleClassification.Accept)
    
    @property
    def reject_count(self):
        return self.action_count(action = TriangleClassification.Reject)
    
    @property
    def refine_count(self):
        return self.action_count(action = TriangleClassification.Refine)
    
    @property
    def reasons(self):
        reason_set = set()
        for reason_dictionary in self.action_dictionary.values():
            for reason in reason_dictionary.keys():
                reason_set.add(reason)
        return reason_set
    
    def action_count(self, action = None, reason = None):
        return self.evaluate_triangles(TriangleEvaluator.Counter, action, reason)

    def generate_triangles(self, gather_action = None, gather_reason = None):
        # if no action provided then expand to all of the actions
        if gather_action == None:
            for action in TriangleClassification.Actions:
                for triangle in self.generate_triangles(action, gather_reason):
                    yield triangle
        else:
            # action is specific
            reason_dictionary = self.action_dictionary.get(gather_action, {})
            # if no reason is provided then expand to all of the reasons
            if gather_reason == None:
                for reason in reason_dictionary:
                    for triangle in self.generate_triangles(gather_action, reason):
                        yield triangle
            else:
                # both action and reason are now specific
                # generate the triangles
                reason_list = reason_dictionary.get(gather_reason, [])
                for triangle in reason_list:
                    yield triangle
            
    def evaluate_triangles(self, evaluator, action = None, reason = None):
        return evaluator.evaluate_triangle_iteration(
            self.generate_triangles(action, reason))
    
    def print_report(self):
        report = self.report()
        key_list = [key for key in report.keys()]
        key_list.sort()
        for key in key_list:
            print key, report[key]

    def report(self, show_accepted_triangles = False):
        report_list = {
            "accept all":self.accept_count,
            "reject all":self.reject_count,
            "refine all":self.refine_count,
            "complete all":self.accept_count + self.reject_count,
            "all":self.count,
        }
        for reason in self.reasons:
            report_list[reason] = self.action_count(
                    action = None, reason = reason)
            volume = self.evaluate_triangles(
                    TriangleEvaluator.TotalVolume, action=None, reason = reason)
            report_list[reason + ".volume"] = volume
        if show_accepted_triangles:            
            for index, triangle in enumerate(
                    self.monitor.generate_triangles(TriangleClassification.Accept)):
                for vertex_number, vertex in enumerate(triangle.vertices):
                    key = "accepted[{:d}.{:d}]".format(index, vertex_number)
                    report_list[key] = vertex        
        return report_list

if __name__ == '__main__':
    pass
