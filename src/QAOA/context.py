import pandas as pd
from itertools import combinations

from .concept import Concept
from .lattice import ConceptLattice


class Context:
    
    def __init__(self, data = None):

        """
        Initializes the context with a DataFrame. that will allow me 
        to access sets of data in a convinent way.
        """

        if isinstance(data, pd.DataFrame):
            self.data = data
        else:
            raise ValueError("Data must be a pandas DataFrame.")
        
    def get_extents(self):

        """Returns the rows of the DataFrame."""
        return set(self.data.index)
    
    def get_intents(self):

        """Returns the columns of the DataFrame."""
        return set(self.data.columns)
    
    def get_powerSet(self, objects):

        """Returns the power set of the DataFrame's rows, Taking objects and returns 
        the power set of the objects which will be used to extract concepts """

        # objects = self.get_extents()
        # print("Objects:", objects)
        power_list = []
        for r in range(1,len(objects)+1):
            for item in combinations(objects, r):
                element = set(item)
                power_list.append(element)  
        return power_list
    
    def _feature_contained(self, object):

        """ It accept an object and it returns a set of features that the object contains"""
        features = set()
        row_Series = self.data.loc[object]  # This is a Series
        #print("Row Series:", row_Series)
        for feature, value in row_Series.items():
            #print("Feature:", feature, "Value:", value)
            if value == 1:
                features.add(feature)
        
        #print("features:", features)

        return features
    
    def _objects_shared(self, feature):

        """ It accepts a feature and returns a set of objects that share the feature"""
        objects = set()
        column_Series = self.data[feature] # This is a Series
        #print("Column Series:", column_Series)
        for object, value in column_Series.items():
            if value == 1:
                objects.add(object)

        #print("Objects:", objects)
        return objects
    
    def Differentiate(self, set1):

        """Return the derivation of set1 in terms of formal concept analysis. """
        #print("Differentiating set:", set1)
        derivative = set()
        if set1.issubset(self.get_extents()):
            for index, element in enumerate(set1):
                features = self._feature_contained(element)
                if index == 0:
                    derivative = derivative.union(features)
                else:
                    derivative = derivative.intersection(features)

            return derivative
        
        elif set1.issubset(self.get_intents()):
            for index, element in enumerate(set1):
                features = self._objects_shared(element)
                if index == 0:
                    derivative = derivative.union(features)
                else:
                    derivative = derivative.intersection(features)

            return derivative

    def extract_concepts(self):

        """From the context, extract all the concepts
        remember that a concept is a pair (A, B) where A' = B and B' = A"""

        power_set = self.get_powerSet(self.get_extents())
        #print("Power Set:", power_set)
        concepts = []
        for extent in power_set:
            # print("Extent:", extent)
            intent = self.Differentiate(extent)
            # print("Intent:", intent)
            double_derivative = self.Differentiate(intent)
            # print("Double Derivative:", double_derivative)
            if double_derivative == extent:
                # print("Concept found: Extent:", extent, "Intent:", intent)
                concepts.append(Concept(extent, intent))
        
        concept_lattice = ConceptLattice(concepts, self)

        return concept_lattice
    
    def __str__(self):
        """String representation of the context."""
        return f"Context with {len(self.data)} objects and {len(self.data.columns)} features."
    
            
    


    

        



