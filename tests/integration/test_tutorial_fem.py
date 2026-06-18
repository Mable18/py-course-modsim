
from utils import  set_test_dir, test_file

import os
import unittest
import xmlrunner    

import sys
my_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..","content", "tutorial-fem"))
sys.path.insert(0, my_dir)
print(sys.path)


class TestFEM(unittest.TestCase):
    def setUp(self):
        os.environ['PYVISTA_OFF_SCREEN'] = "True"
        set_test_dir("content/tutorial-fem")
        os.makedirs("tmp", exist_ok=True) # create tmp dir for VTK output 

    def tearDown(self):
        set_test_dir("../..")

    def test_fem_tutorial1(self):
        self.assertEqual(test_file("tutorial-fem-01.py"), 0, "Must return true")
    
    def test_fem_tutorial2(self):
        self.assertEqual(test_file("tutorial-fem-02.py"), 0, "Must return true")
    
    #def test_fem_tutorial3(self):
    #    self.assertEqual(test_file("tutorial-fem-03.py"), 0, "Must return true")
        


if __name__ == "__main__":
    # unittest.main()
    unittest.main(
        testRunner=xmlrunner.XMLTestRunner(output='test-reports'),
        # these make sure that some options that are not applicable
        # remain hidden from the help menu.
        failfast=False, buffer=False, catchbreak=False)
