#!/usr/bin/env python
# =====================================================================

# Globally useful modules:

import numpy
import sys, getopt
from PIL import Image
import humvi
import os

# =====================================================================


def rgb_composer(rfile, gfile, bfile, source_r, source_g, source_b):
    """
    NAME
        compose.py

    PURPOSE
      Make a color composite PNG image from 3 FITS images, using the Lupton
      algorithm. Part of the HumVI package.

    COMMENTS
      Reads filter name and zero point from header and tries to set scales
      automatically - unless scales are set on command line. Note
      telescope/survey has to be recognised for this to work...

    USAGE
      compose.py [flags] red.fits green.fits blue.fits

    FLAGS
      -h        Print this message
      -v        Verbose operation
      -b --subtract-background

    INPUTS
      red.fits etc  Names of FITS files containing image data

    OPTIONAL INPUTS
      -s --scales   3,2,4     Comma-separated scales for R,G,B channels [None]
      -p --parameters  5,0.05 Non-linearity parameters Q,alpha
      -o --output   outfile   Name of output filename [guessed]
      -x --saturate-to style  Saturation method, color or white [white]
      -z --offset 0.1       Offset image level (+ve = gray background)
      -m --mask -1.0        Mask images below level [don't]

    OUTPUTS
      stdout      Useful information
      outfile     Output plot in png format


    EXAMPLES

        Note the different alpha required, to cope with survey depth!

        CFHTLS:
          HumVI.py  -v -s 0.4,0.6,1.7 -z 0.0 -p 1.7,0.09 -m -1.0 \
             -o examples/CFHTLS-test_gri.png \
                examples/CFHTLS-test_i.fits \
                examples/CFHTLS-test_r.fits \
                examples/CFHTLS-test_g.fits

        PS1 (Needs optimizing on a larger image.):
          HumVI.py  -v -s 0.6,0.6,1.7 -z 0.0 -p 1.7,0.00006 -m -1.0 \
             -o examples/PS1-test_riz.png \
                examples/PS1-test_z.fits \
                examples/PS1-test_i.fits \
                examples/PS1-test_r.fits

        DES (Experimental. No images checked in.):
          HumVI.py -v -s 1.0,1.2,2.8 -z 0.0 -p 1.0,0.03 -m -1.0 \
             -o examples/DES-test_gri.png \
                examples/DES-test_i.fits \
                examples/DES-test_r.fits \
                examples/DES-test_g.fits

        VICS82:
          HumVI.py -v -s 1.0,1.4,2.0 -z 0.0 -p 1.5,0.4 -m -1.0 \
             -o examples/VICS82-test_iJKs.png \
                examples/VICS82-test_Ks.fits \
                examples/VICS82-test_J.fits \
                examples/VICS82-test_i.fits
        KiDS:
          HumVI.py -v -s 0.4,0.6,1.7 -z 0.0 -p 1.5,0.02 -m -1.0 \
             -o examples/KiDS-test_gri.png \
                examples/KiDS-test_i.fits \
                examples/KiDS-test_r.fits \
                examples/KiDS-test_g.fits

    BUGS

    - Renormalized scales will not be appropriate if one image is in very
      different units to another, or if images are in counts, not counts per
      second or AB maggies.

    - Code currently assumes one file per channel, whereas we might want to
      use N>3 images in combination.  The image to channel transformation
      would be performed after scaling to flux units but before
      stretching. Masking would need to be done at this point too.

    - Should cope with just two, or even one, input image file(s).

    HISTORY
      2012-05-11    started Marshall (Oxford)
      2012-07-??    integrated with wherry.py Sandford (NYU)
      2012-12-19    defaults set for CFHTLS images Marshall (Oxford)
      2013-02-21    defaults set for PS1 images Marshall (Oxford)
      2013-02-26    experimenting with DES images Hirsch (UCL)
    """

    # -------------------------------------------------------------------

    vb = False

    outfile = "color.png"

    # Defaults optimized for CFHTLS...
    pars = "1.7,0.09"
    scales = "0.4,0.6,1.7"

    # More general sensible choices:
    backsub = False
    saturation = "white"
    offset = 0.0
    masklevel = None  # Better default would be -1.0

    # Parse nonlinearity parameters:
    Qs, alphas = pars.split(",")
    Q = float(Qs)
    alpha = float(alphas)

    # Parse channel colour scales:
    x, y, z = scales.split(",")
    rscale, gscale, bscale = float(x), float(y), float(z)

    # Compose the image!

    image = humvi.compose_mod(
        rfile,
        gfile,
        bfile,
        source_r,
        source_g,
        source_b,
        scales=(rscale, gscale, bscale),
        Q=Q,
        alpha=alpha,
        masklevel=masklevel,
        saturation=saturation,
        offset=offset,
        backsub=backsub,
        vb=vb,
        outfile=outfile,
    )

    return image
