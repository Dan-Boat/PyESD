# -*- coding: utf-8 -*-
"""
Created on Thu May 12 13:43:08 2022

@author: dboateng
"""

import unittest

from pyESD.ESD_utils import haversine


class TestHaversine(unittest.TestCase):
    """
    Test if the haversine function use in dertiming the radius for
    regional means is correct.
    """
    
    @classmethod
    def setUpClass(self):
        
        self.distance = haversine(lon1=0, lat1=47, lon2=10, lat2=47)
    
    def test_distance(self):
        self.assertGreater(self.distance, 700, "The estimated distance between the two coordinates is incorrect")
        
if __name__ == '__main__': 
    unittest.main()