# Copyright(c) 2014 WillyMillsLLC

import math

from optical.light import Direction
from optical.light import FilterMonitor
from optical.light import NullFilterMonitor
from optical.light import Plane
from optical.light import Position
from optical.light import Ray
from optical.light import Sphere
from optical.light import SphericalTriangle
from optical.light import WaveFront
from optical.light import vector_dot_product

class SpecificationsHolder(object):
    """
    Holds specifications for various optical instruments and their components
    """
    def __init__(self, **kwargs):
        self._specs = {}
        self._specs.update(kwargs)

    def prefixed_specificatons(self, prefix):
         return {
                (prefix + key): value for key, value in self.specifications.iteritems()
            }
            
    @property
    def specifications(self):
        return self._specs

class InstrumentWithAperture(SpecificationsHolder):
    """
    Base class for instruments featuring imager and aperture
    """
    def __init__(self, **kwargs):
        super(InstrumentWithAperture, self).__init__(**kwargs)  
        self._build()
    
    @property
    def construction_specifications(self, prefix = "construction: "):
        specs = {}
        specs.update(self.imager.prefixed_specificatons(prefix + "imager: "))
        specs.update(self.aperture.prefixed_specificatons(prefix + "aperture: "))
        return specs
    
    @property
    def aperture_diameter(self):
        return self._specs['diameter']

    @property
    def horizontal_axis(self):
        return self._specs['horizontal_axis']
    
    @property
    def vertical_axis(self):
        return self._specs['vertical_axis']
    
    @property
    def height(self):
        return self._specs['height']
    
    @property
    def placement(self):
        return self._specs['placement']
    
    @property
    def width(self):
        return self._specs['width']    

    def _build(self):
        self._add_manufacturing_specs()
        self._build_imager()
        self._build_aperture()
    def _build_imager(self):
        self.imager_position = Position.from_xyz((0.0, 0.0, 0.0))
        self.imager_direction = self.placement.direction.normalized()
        self.image_placement = Ray(self.imager_position, self.imager_direction)
        self.imager = ImageSensor(
                width = self.width,
                height = self.height, 
                placement = self.image_placement,
                horizontal_direction = self.horizontal_axis,
                vertical_direction = self.vertical_axis,
                )
    def _build_aperture(self):
        self.calculate_aperture_location()
        self.create_aperture_at_location()
    def calculate_aperture_location(self):        
        self.aperture_position = self.placement.endpoint
        self.aperture_directon = self.placement.direction.normalized()
        self.aperture_placement = Ray(self.aperture_position, self.aperture_directon)

    def restrict_wavefront_to_view(self, wavefront):
        return self.aperture.filter_wavefront(wavefront)
    
class PinHoleCamera(InstrumentWithAperture):
    """
    A camera consisting of an imaging array with a pinhole aperture.
    """
    def __init__(self, **kwargs):
        super(PinHoleCamera, self).__init__(**kwargs)  

    def _add_manufacturing_specs(self):
        specs = {'type' : "Pin Hole Camera", 
            'manufacturor':'Acme Inc', 'model':'pineapple'}
        self._specs.update(specs)        
                
    def create_aperture_at_location(self):        
        self.aperture = Aperture(
                diameter = self.aperture_diameter, 
                placement = self.aperture_placement)

    def effective_view_from_apparent_view(self, wavefront):
        return wavefront # pin hole camera has no optically active elements

class SphericalMirrorScope(InstrumentWithAperture):
    """
    Scope built with spherical reflecting mirror
    """
    def __init__(self, **kwargs):
        super(SphericalMirrorScope, self).__init__(**kwargs)  
        self._build()
    
    @property
    def focal_length(self):
        return self._specs['focal_length']
    
    def _add_manufacturing_specs(self):
        specs = {'type' : "Spherical Mirror Scope", 
            'manufacturor':'Acme Inc', 'model':'eagle'}
        self._specs.update(specs)        

    def calculate_aperture_location(self):
        super(SphericalMirrorScope, self).calculate_aperture_location()
        
    def create_aperture_at_location(self):
        self.aperture = SphericalMirror(
                focal_length = self.focal_length,
                diameter = self.aperture_diameter, 
                placement = self.aperture_placement)
        
    def effective_view_from_apparent_view(self, wavefront):
        return WaveFront(self.aperture.ReflectWaveFront(wavefront))


class Aperture(SpecificationsHolder):
    """
    Represents circular opening in light blocking plane.
    """
    def __init__(self, **kwargs):
        super(Aperture, self).__init__(**kwargs)
        self.plane = Plane.orthogonal_to_ray(self.placement)
    
    @property
    def diameter(self):
        return self._specs['diameter']
    
    @property
    def origin(self):
        return self.placement.position
    
    @property
    def placement(self):
        return self._specs['placement']
    
    @property
    def radius(self):
        return 0.5 * self.diameter 

    def distance_to_ray(self, ray):
        plane_intercept_point = self.intercept_point(ray)
        distance_from_center = self.origin.difference(plane_intercept_point).length()
        return distance_from_center
    
    class FilterResolver:
        def __init__(self, aperture):
            self.aperture = aperture
            self.intercept_point = aperture.intercept_point
            self.monitor = aperture.monitor
            self.origin = aperture.origin
            self.placement = aperture.placement
            self.radius = aperture.radius
            self.count = 0
            self.calculate_size_criteria()
        def calculate_size_criteria(self):
            absolute_minimum_size = 0.000000001
            relative_minimum_size = self.radius * 0.005
            self.minimum_size = max(absolute_minimum_size, relative_minimum_size)
             
        def __call__(self, triangle):
            self.triangle = triangle
            
            # these checks have side effects, so do not change the order
            tests_to_try = [
                self.check_too_many,
                self.check_too_big,
                self.check_backwards,
                self.check_too_small,
                self.check_inside,
                self.check_straddle,
                self.check_outside,
                ]
            for test in tests_to_try:
                result = test()
                if result != None:
                    return result
                
            # still ambiguous
            return self.monitor.refine(triangle, "ambiguous")

        def check_too_many(self):
            # sanity limit
            self.count = self.count + 1
            if self.count > 300000:
                return self.monitor.reject(self.triangle, "reject since too many")
            return None
        
        def check_too_big(self):
            # refine large triangles because later tests assume they are
            # smaller than a full octant
            if self.triangle.diameter > 0.5:
                return self.monitor.refine(self.triangle, "refine since too big")
            return None

        def check_backwards(self):
            # reject backwards facing triangles
            for vertex in self.triangle:
                criteria = self.placement.direction.normalized().dot(vertex.direction)
                if criteria <= 0.0:
                    return self.monitor.reject(self.triangle, "reject since backwards")
            return None
            
        def check_too_small(self):
            self.calculate_relative_vertices() # side effect needed later
            self.calculate_perimeter()
            # reject if too small
            if self.perimeter < self.minimum_size:
                return self.monitor.reject(self.triangle, "reject since too small")
            return None
        def calculate_relative_vertices(self):
            absolute_vertices = [self.intercept_point(vertex) for vertex in self.triangle] 
            self.relative_vertices = [vertex.difference(self.origin) for vertex in absolute_vertices]
            self.a, self.b, self.c = self.relative_vertices
        def calculate_perimeter(self):
            edges = [self.a.difference(self.b), self.b.difference(self.c), self.c.difference(self.a)]
            self.perimeter = sum(edge.length() for edge in edges)            
           
        def check_inside(self):
            self.calculate_near_and_far_vertex_distances() # side effect
            # accept if all vertices are inside aperture
            if self.farthest_vertex_distance <= self.radius:
                return self.monitor.accept(self.triangle, "accept since inside")
            return None
        def calculate_near_and_far_vertex_distances(self):
            self.vertex_distances = [point.length() for point in self.relative_vertices]
            self.farthest_vertex_distance = max(self.vertex_distances)
            self.nearest_vertex_distance = min(self.vertex_distances)            
        
        def check_straddle(self):
            # refine if some, but not all, vertices are inside aperture
            if self.nearest_vertex_distance < self.radius:
                return self.monitor.refine(self.triangle, "refine since straddle")
            return None
            
        def check_outside(self):
            # reject if some projection of the vertices exceed aperture radius
            test_directions = [
                    # use triangle vertices themselves...
                    self.a.normalized(),
                    self.b.normalized(),
                    self.c.normalized(),
                    # ... plus these will reject outside of a cube containing the aperture
                    Direction.up, 
                    Direction.down, 
                    Direction.right,
                    Direction.forward, 
                    Direction.backward,
                ]
            for direction in test_directions:
                projected_vertices = [direction.dot(vertex) for vertex in self.relative_vertices]
                if min(projected_vertices) > self.radius:
                    return self.monitor.reject(self.triangle, "reject since outside")                
            return None
            
    def filter_wavefront(self, wavefront, make_report = False):
        if make_report:
            self.monitor = FilterMonitor()
        else:
            self.monitor = NullFilterMonitor()
        self.count = 0
        
        # filter the wavefront until it fits through the hole.
        resolver = self.FilterResolver(self)
        result = wavefront.refine_with_filter(resolver)
        
        if make_report:
            self.monitor.print_report()
            
        return result
            
    def intercept_point(self, ray):
        return self.plane.intersect(ray)
    
class SphericalMirror(Aperture):
    """
    Spherical mirror. Immutable.
    """
    def __init__(self, **kwargs):        
        super(SphericalMirror, self).__init__(**kwargs)
        self.build_sphere()
    def build_sphere(self):        
        # positive concave, negative convex
        signed_spherical_radius = 2.0 * self.focal_length
        radius_squared = self.radius * self.radius
        spherical_radius_squared = (signed_spherical_radius * signed_spherical_radius)
        spherical_radius = math.sqrt(spherical_radius_squared)
        distance_to_center_squared = spherical_radius_squared - radius_squared
        if distance_to_center_squared < 0.0:
            raise ValueError("diameter larger than sphere")
        distance_to_center = math.sqrt(distance_to_center_squared)
        if self.is_convex:
            offset_to_center = distance_to_center
        else:
            offset_to_center = -distance_to_center
        center_position = self.placement.scaled_endpoint(offset_to_center)
        self._sphere = Sphere(center_position, spherical_radius)
            
        
    
    @property
    def focal_length(self):
        return self._specs['focal_length']
    
    @property
    def is_concave(self):
        return self.focal_length > 0.0
    
    @property
    def is_convex(self):
        return self.focal_length < 0.0
    
    @property
    def sphere(self):
        return self._sphere
    
    def intersection_point(self, ray):
        points = self.sphere.intersection_points(ray)
        if len(points) == 0:
            return None
        if len(points) == 1:
            return points[0]
        if self.is_convex:
            return points[1] # nearest
        return points[0] # farthest
    
    def reflect(self, ray):
        intercept = self.intersection_point(ray)
        if intercept == None:
            return None
        normal_to_surface = intercept.difference(self.sphere.center_position).normalized()
        normal_component_of_ray = vector_dot_product( ray.direction.xyz, normal_to_surface.xyz)
        reflection_action = normal_to_surface.scale(-2.0 * normal_component_of_ray)
        reflected_direction = ray.direction.add(reflection_action)
        return Ray(ray.position, reflected_direction)
    
    def ReflectWaveFront(self, triangles):
        for triangle in triangles:
            yield self.ReflectTriangle(triangle)
    def ReflectTriangle(self, triangle):
        vertex_list = [self.reflect(vertice) for vertice in triangle]
        return SphericalTriangle(tuple(vertex_list))
        
    
class ImageSensor(SpecificationsHolder):
    """
    Image sensor represents film surface or electronic sensor.
    """
    def __init__(self, **kwargs):
        super(ImageSensor, self).__init__(**kwargs)
        self.image_center = self.placement.position
        self.image_look_axis = self.placement.direction.normalized()
        self.image_right_axis = self.horizontal_direction.normalized().scale(
                self.width * 0.5)
        self.image_up_axis = self.vertical_direction.normalized().scale(
                self.height * 0.5)
                
    @property
    def diagonal(self):
        return math.sqrt(self.width*self.width + self.height* self.height)
    
    @property
    def height(self):
        return self._specs['height']  
    
    @property
    def horizontal_direction(self):
        return self._specs['horizontal_direction']

    @property
    def placement(self):
        return self._specs['placement']
    
    # Unit of distance is cm or inches or ...
    @property
    def unit_of_distance(self):
        return self._specs['unit']    
    
    @property
    def vertical_direction(self):
        return self._specs['vertical_direction']

    @property
    def width(self):
        return self._specs['width']   
    
    @property
    def area(self):
        return self.width * self.height
    
    def view_at_location(self, image_location):
        image_x, image_y = image_location
        motion_x = self.image_right_axis.scale(image_x)
        motion_y = self.image_up_axis.scale(image_y)
        total_motion = motion_x.add(motion_y)
        position = self.image_center.add(total_motion)
        return Ray(position, self.image_look_axis)


class ScopeMeasurements(dict):
    """
    Holds measurments made on scope.
    """
    
    @property
    def status(self):
        return self['status']    
    @status.setter
    def status(self, value):
        self['status'] = value
            
    def print_evaluation(self):
        keys = self.keys()
        keys.sort()
        for key in keys:
            print key + ':', self[key]
        print

class ImagePointEvaluation(object):
    """
    Evaluation of optical instrument at a single point of the imaging array
    """
    def __init__(self, weight, location_in_image):
        self.weight = weight
        self.location_in_image = location_in_image
        
    @property
    def brightness(self):
        return self.weight * self.apparent_view.volume
    
    @property
    def blurring(self):
        return self.weight * math.sqrt(self.effective_view.volume)
    
    @property
    def effective_center(self):
        return self.effective_view.central_direction
    
    def peer_through_scope(self, scope, verbose = False):
        self.scope = scope
        self.imager = scope.imager
        self.verbose = verbose
        self.measure_light_cone()
        
    def measure_light_cone(self):
        self.find_location_in_space()
        self.find_apparent_view_cone()
        self.find_effective_light_cone()
    def find_location_in_space(self):            
        self.location_in_space = self.imager.view_at_location(self.location_in_image)
    def find_apparent_view_cone(self):
        if self.verbose:
            print "examining apparent view", self.location_in_image
        self.position_sphere_of_light_at_location_in_space()
        self.restrict_sphere_of_light_to_apparent_view()
    def position_sphere_of_light_at_location_in_space(self):
        self.offset =  self.location_in_space.position.direction_from_origin()
        self.sphere_of_light = WaveFront.unit_sphere.add(self.offset)
    def restrict_sphere_of_light_to_apparent_view(self):
        self.apparent_view = self.scope.restrict_wavefront_to_view(self.sphere_of_light)
    def find_effective_light_cone(self):
        if self.verbose:
            print "examining effective view", self.location_in_image
        self.effective_view = self.scope.effective_view_from_apparent_view(self.apparent_view)
    
class ScopeEvaluator(object):
    """
    Evaluates quality of a scope
    """
    
    def __init__(self, scope):
        self.scope = scope
        self.measurements = ScopeMeasurements()
        self.measurements.status = 'not done'
        
    def evaluate(self, verbose = False): 
        self.verbose = verbose
        self.get_specifications()
        self.evaluate_image_sensor()
        self.peer_through_scope()
        self.record_observations()
        self.mark_complete()

    def get_specifications(self):
        self.measurements.update(self.scope.specifications)
        self.measurements.update(self.scope.construction_specifications)    

    def evaluate_image_sensor(self):
        self.imager = self.scope.imager 
        self.measurements["imager area"] = self.imager.area
        
    def peer_through_scope(self):
        self.choose_image_sample_grid()
        self.peer_through_scope_at_each_image_sample()
    def choose_image_sample_grid(self):
        self.choose_5_by_5_grid()
    def choose_center_and_corners(self):
        # Weights are based on assumning:
        #   image edge point values are linear interpolation between image corners
        #   image is divided into four equal quadrants
        #   quadrant edge points are linear interpolation between quadrant corners
        #   interior point values are linear interpolation between quadrant edges
        self.image_sample_grid = [        
                ImagePointEvaluation(0.25, (0.0, 0.0)),
                ImagePointEvaluation(0.1875, (1.0, 1.0)),
                ImagePointEvaluation(0.1875, (1.0, -1.0)),
                ImagePointEvaluation(0.1875, (-1.0, 1.0)),
                ImagePointEvaluation(0.1875, (-1.0, -1.0)),                
        ]
    def choose_5_by_5_grid(self):
        self.image_sample_grid = []
        for y_index in range(5):
            y = float(y_index - 2) * 0.5
            if y_index == 0 or y_index == 4:
                y_weight = 0.125
            else:
                y_weight = 0.25
            for x_index in range(5):
                x = float(x_index - 2) * 0.5
                if x_index == 0 or y_index == 4:
                    x_weight = 0.125
                else:
                    x_weight = 0.25
                weight = x_weight * y_weight
                self.image_sample_grid.append(
                        ImagePointEvaluation(weight, (x, y)))
                
                
    def peer_through_scope_at_each_image_sample(self):
        for image_sample in self.image_sample_grid:
            image_sample.peer_through_scope(self.scope, self.verbose)

    def record_observations(self):
        self.calculate_and_record_brightness()
        self.calculate_and_record_field_of_view()
        self.calculate_and_record_resolution()
        
    def calculate_and_record_brightness(self):        
        self.average_brightness = sum(sample.brightness for sample in self.image_sample_grid)
        self.measurements['brightness'] = self.average_brightness
        
    def calculate_and_record_field_of_view(self):
        self.calculate_maximum_divergence_in_view_directions()
        self.adjust_for_imager_width_versus_diagonal()
        self.record_field_of_view_measurements()
    def calculate_maximum_divergence_in_view_directions(self):
        self.diagonal_divergence =  max(
                self.generate_all_distances_between_effective_view_pairs())
        self.measurements['diagonal divergence'] =  self.diagonal_divergence
    def generate_all_distances_between_effective_view_pairs(self):
        for image_sample in self.image_sample_grid:
            sample_view_center = image_sample.effective_center
            for other_image_sample in self.image_sample_grid:
                other_view_center = other_image_sample.effective_center
                delta = sample_view_center.difference(other_view_center)
                delta_length = delta.length()
                yield delta_length
            
    def adjust_for_imager_width_versus_diagonal(self):
        half_divergence = self.diagonal_divergence / 2.0
        half_angle_divergence = math.asin(half_divergence)
        diagonal_half_tangent = math.tan(half_angle_divergence)
        horizontal_half_tangent = diagonal_half_tangent * self.imager.width / self.imager.diagonal
        self.horizontal_divergence = 2.0 * horizontal_half_tangent
        self.horizontal_viewing_angle = 2.0 * math.atan(horizontal_half_tangent)
        
    def record_field_of_view_measurements(self):
        # traditional 1000 yard view
        self.measurements['field of view at 1000'] = self.horizontal_divergence * 1000.0 
        # record as degrees
        self.measurements['angular view in degress'] = 180.0 * self.horizontal_viewing_angle / math.pi 
    
    def calculate_and_record_resolution(self):
        self.measure_average_blurring()
        self.calculate_resolution()
        self.record_resolution()
    def measure_average_blurring(self):
        self.average_blurring = sum(sample.blurring for sample in self.image_sample_grid)
    def calculate_resolution(self):
        self.resolution_size = self.average_blurring
        self.horizontal_resolution = self.horizontal_divergence / self.resolution_size
        self.vertical_resolution = self.horizontal_resolution * self.imager.height / self.imager.width
    def record_resolution(self):
        self.measurements['blurring'] = self.average_blurring
        self.measurements['horizontal resolution'] = self.horizontal_resolution
        self.measurements['vertical resolution'] = self.vertical_resolution
    
    def mark_complete(self):
        self.measurements.status = 'complete'

    @property
    def evaluation(self):
        return self.measurements

    def print_evaluation(self, title = None):
        if title != None:
            print title
        self.measurements.print_evaluation()

def PinholeCameraCreator():
    return  PinHoleCamera(
                width = 40.0, 
                height = 30.0, 
                unit = "cm",
                placement = Ray.up.scaled_direction(50.0),
                horizontal_axis = Direction.right,
                vertical_axis = Direction.forward,
                diameter = 1.0
            )
            
def SphericalMirrorScopeCreator():
    diameter = 5.0
    radius = diameter / 2.0
    focal_length = 100.0
    offset = math.sqrt(focal_length * focal_length - radius * radius)
    return  SphericalMirrorScope(
                width = 1.0, 
                height = 0.75, 
                unit = "cm",
                focal_length = focal_length,
                placement = Ray.up.scaled_direction(offset),
                horizontal_axis = Direction.right,
                vertical_axis = Direction.forward,
                diameter = diameter
            )
    
class ExampleRunner(object):
    def __init__(self, scope_creator):
        self.scope_creator = scope_creator
        self.create_example_scope()
    def __call__(self):
        self.perform_example_evaluation()
    def create_example_scope(self):
        self.example_scope = self.scope_creator()
    def perform_example_evaluation(self):
        scope_evaluator = ScopeEvaluator(self.example_scope)
        scope_evaluator.print_evaluation("Before:")
        scope_evaluator.evaluate(verbose = True)
        scope_evaluator.print_evaluation("After:")


if __name__ == "__main__":
    #ExampleRunner(PinholeCameraCreator)()
    ExampleRunner(SphericalMirrorScopeCreator)()
