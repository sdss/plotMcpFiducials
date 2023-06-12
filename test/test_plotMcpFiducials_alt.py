#!/usr/bin/env python3
"""Test that plotMcpFiducials output matches old fiducial tables."""

import os
import unittest

import numpy as np

from plotmcpfiducials import plotMcpFiducials


def read_fiducials(infile):
    types = ("i4", "i8", "S3", "f8", "i4", "i8", "|S2", "f8", "i4")
    names = (
        "fiducial",
        "Encoder1",
        "",
        "error1",
        "npoint1",
        "Encoder2",
        "",
        "error2",
        "npoint2",
    )
    return np.loadtxt(infile, np.dtype(list(zip(names, types))))


class Test58799(unittest.TestCase):
    def setUp(self):
        path = os.path.dirname(__file__)
        args = "-alt -dir %s/data/58799 -mjd 58799 " % path
        args = args + " -time0 1573545352 -time1 1573546087"
        args = args + " -canon -reset -scale "
        args = args + " -noplot"
        argv = args.split()
        self.fiducials, self.fpos, self.fposErr, self.nfpos = plotMcpFiducials(argv)
        self.expect = read_fiducials("%s/data/v1_141/alt.dat" % path)

    def test_fiducials(self):
        np.testing.assert_array_equal(
            self.fiducials[1:],
            self.expect["fiducial"],
            err_msg="fiducials failed",
        )

    def test_points1(self):
        np.testing.assert_array_equal(
            self.nfpos["pos1"][1:],
            self.expect["npoint1"],
            err_msg="nfpos['pos1'] failed",
        )

    def test_points2(self):
        np.testing.assert_array_equal(
            self.nfpos["pos2"][1:],
            self.expect["npoint2"],
            err_msg="nfpos['pos2'] failed",
        )

    def test_encoder1(self):
        np.testing.assert_array_equal(
            self.fpos["pos1"][1:],
            self.expect["Encoder1"],
            err_msg="fpos['pos1'] failed",
        )

    def test_encoder2(self):
        np.testing.assert_array_equal(
            self.fpos["pos2"][1:],
            self.expect["Encoder2"],
            err_msg="fpos['pos2'] failed",
        )

    def test_errors1(self):
        np.testing.assert_equal(
            self.fposErr["pos1"][1:],
            self.expect["error1"],
            err_msg="fposErr['pos1'] failed",
        )

    def test_errors2(self):
        np.testing.assert_equal(
            self.fposErr["pos2"][1:],
            self.expect["error2"],
            err_msg="fposErr['pos2'] failed",
        )


if __name__ == "__main__":
    unittest.main()
