#!/usr/bin/env python3
"""Test that plotMcpFiducials output matches old fiducial tables."""

import unittest
import numpy as np
import plotMcpFiducials
import sys,os

def read_fiducials(infile):
    types= ('i4', 'i8', 'S3', 'f8', 'i4', 'i8', '|S2', 'f8', 'i4')
    names = ('fiducial', 'Encoder1', '', 'error1', 'npoint1', 'Encoder2', '', 'error2', 'npoint2')
    return np.loadtxt(infile,np.dtype(list(zip(names,types))))
    

class Test58799(unittest.TestCase):
    def setUp(self):
        ver='v1_141'; mjd='58799';  axis="alt"
        #args='-alt -dir data/58799 -fiducialFile=data/v1_141/alt.dat -mjd 58799 '
        path=os.path.dirname(sys.argv[0])
        args='-alt -dir %s/data/58799 -mjd 58799 ' % path
        args=args+' -time0 1573545352 -time1 1573546087'
        args=args+' -canon -reset -scale '
        args=args+' -noplot'  
        self.fiducials, self.fpos, self.fposErr, self.nfpos = plotMcpFiducials.main(args.split())
        self.expect = read_fiducials('%s/data/v1_141/alt.dat' % path)           
        
    def test_fiducials(self,):
        np.testing.assert_array_equal(self.fiducials[1:], self.expect['fiducial'], err_msg="fiducials failed")
    def test_points1(self,):    
        np.testing.assert_array_equal(self.nfpos['pos1'][1:],self.expect['npoint1'], err_msg="nfpos['pos1'] failed")
    def test_points2(self, ):    
        np.testing.assert_array_equal(self.nfpos['pos2'][1:],self.expect['npoint2'], err_msg="nfpos['pos2'] failed")
                
    def test_encoder1(self,):
        np.testing.assert_array_equal(self.fpos['pos1'][1:], self.expect['Encoder1'], err_msg="fpos['pos1'] failed")
        #np.testing.assert_allclose(self.fpos['pos1'][1:], self.expect['Encoder1'], atol=0.5, err_msg="fpos['pos1'] failed")
    def test_encoder2(self,):
        np.testing.assert_array_equal(self.fpos['pos2'][1:], self.expect['Encoder2'], err_msg="fpos['pos2'] failed")
        #np.testing.assert_allclose(self.fpos['pos2'][1:], self.expect['Encoder2'], atol=0.5, err_msg="fpos['pos2'] failed")

    #@unittest.expectedFailure
    def test_errors1(self, ):
        #np.testing.assert_allclose(self.fposErr['pos1'][1:],self.expect['error1'],atol=0.1, err_msg="fposErr['pos1'] failed")
        np.testing.assert_equal(self.fposErr['pos1'][1:],self.expect['error1'],err_msg="fposErr['pos1'] failed")

    def test_errors2(self, ):
        #np.testing.assert_allclose(self.fposErr['pos2'][1:],self.expect['error2'],atol=0.1, err_msg="fposErr['pos2'] failed")
        np.testing.assert_equal(self.fposErr['pos2'][1:],self.expect['error2'],err_msg="fposErr['pos2'] failed")
    
if __name__ == '__main__':
    unittest.main()