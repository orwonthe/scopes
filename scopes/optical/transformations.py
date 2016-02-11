# Copyright(c) 2014 WillyMillsLLC
import math
from numpy import matrix
from numpy import array

class Transformation(object):
    """
    Projective and linear transformations of 3d objects
    
    Reperesents translation, rotation, and their composites
    """
    
    def __init__(self, params, inverse_matrix = None):
        self._matrix = matrix(params)
        self._inverse_matrix = inverse_matrix 
        if self._matrix.shape != (4, 4):
            raise TypeError("must be 4 by 4")
        if not (1.0, 0.0, 0.0, 0.0) == tuple(self._matrix[i,0] for i in range(4)):
            raise ValueError("must preserve scaling")
        if self._inverse_matrix == None :
            self._inverse_matrix = self._matrix.I 
            
    @classmethod
    def rotation_about_x_axis(cls, angle):
        cos = math.cos(angle)
        sin = math.sin(angle)
        return Transformation(matrix([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, cos, sin],
            [0.0, 0.0, -sin, cos],
        ])) 
        
    @classmethod
    def rotation_about_y_axis(cls, angle):
        cos = math.cos(angle)
        sin = math.sin(angle)
        return Transformation(matrix([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, cos, 0.0, -sin],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, sin, 0.0, cos],
        ])) 
        
    @classmethod
    def rotation_about_z_axis(cls, angle):
        cos = math.cos(angle)
        sin = math.sin(angle)
        return Transformation(matrix([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, cos, sin, 0.0],
            [0.0, -sin, cos, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])) 
    
    @classmethod
    def translation_by_position(cls, position):
        position_list = [each for each in position]
        return Transformation(matrix([
            position_list, 
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])) 
        
    def clone_from_matrices(self, transformation_matrix, inverse_matrix):
        return self.__class__(transformation_matrix, inverse_matrix)

    def clone_from_matrix(self, transformation_matrix):
        inverse_matrix = transformation_matrix.I
        return self.clone_from_matrices(transformation_matrix, inverse_matrix)
    
    def compose(self, other_transformation):
        # combine transformations: self happens first, then other_transformation
        transformation_matrix = self._matrix * other_transformation._matrix
        inverse_matrix = other_transformation._inverse_matrix * self._inverse_matrix
        return self.clone_from_matrices(transformation_matrix,inverse_matrix)
    
    def contravariant_transform(self, contravariant_target):
        target_row = tuple(contravariant_target)
        target_as_row_matrix = matrix(target_row)
        result_row_matrix = target_as_row_matrix * self._matrix
        result_tuple = tuple(array(result_row_matrix)[0])
        return result_tuple
    
    def covariant_transform(self, covariant_target):
        target_column = tuple(covariant_target)
        target_column_matrix = matrix(target_column).T 
        result_column_matrix = self._inverse_matrix * target_column_matrix
        result_tuple = tuple(array(result_column_matrix)[:,0])
        return result_tuple
    
    def inverse(self):
        return self.clone_from_matrices(self._inverse_matrix, self._matrix)
    
if __name__ == "__main__":
    pass
