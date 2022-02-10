#!/usr/bin/env python

"""Testing psf.py module

"""

__author__      = "Michael Stadler"
__copyright__   = "Copyright 2021, California, USA"

import unittest
from flymovie.psf import *
from flymovie.simnuc import *

# Make fake image with 2 beads inbounds, 1 bead out of bounds
mask = np.zeros((50,50,50))
sim = fm.Sim(mask, 100,100)
sim.add_object((15,15,15), 100, 1, 1)
sim.add_object((35,35,35), 100, 1, 1)
sim.add_object((1,1,1), 100, 1, 1)
kernel = np.ones((3,3,3))
kernel[1,1,1] = 2
sim.add_kernel(kernel, 100,100)
sim.convolve()
im = sim.im.copy()

class TestPSF(unittest.TestCase):
    
    def test_extract_beads(self):
        boxes = extract_beads(im, 1, (5,5,5))
        self.assertTrue(np.array_equal(boxes.shape, (2,5,5,5)), 'Wrong shape.')
        # Make sure beads are centered.
        for i in range(boxes.shape[0]):
            max_coords = [x[0] for x in np.where(boxes[i] == np.max(boxes[i]))]
            self.assertTrue(np.array_equal(max_coords, (2,2,2)), 'Wrong max coord.')

    def test_extract_beads_batch(self):
        boxes = extract_beads_batch([im, im], 1, (5,5,5))
        self.assertTrue(np.array_equal(boxes.shape, (4,5,5,5)), 'Wrong shape.')
        # Make sure beads are centered.
        for i in range(boxes.shape[0]):
            max_coords = [x[0] for x in np.where(boxes[i] == np.max(boxes[i]))]
            self.assertTrue(np.array_equal(max_coords, (2,2,2)), 'Wrong max coord.')

    def test_remove_bad_beads(self):
        boxes = extract_beads_batch([im, im], 1, (5,5,5))
        boxes[1,0,0,0] = 1000
        self.assertEqual(np.max(boxes), 1000, 'Should be 1000.')
        good = remove_bad_beads(boxes, [1])
        self.assertTrue(np.array_equal(good.shape, (3,5,5,5)), "Should have lost a slice.")
        self.assertLess(np.max(good), 1000, "Should be less than 1000.")

if __name__ == '__main__':
	unittest.main()