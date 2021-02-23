import numpy as np
from cxotime import CxoTime
from numpy.random import RandomState
from astropy.coordinates import solar_system_ephemeris, \
    get_body, SkyCoord
import Ska.Sun
from pathlib import Path
import xija
from astropy.io import fits
from Quaternion import Quat


solar_system_ephemeris.set('jpl')

earth_vis_grid_path = Path(xija.heat.__file__).parent / 'earth_vis_grid_nside32.fits.gz'
earth_vis_grid = None
alts = None
log_earth_vis_dists = None


def calc_earth_vis_from_grid(ephems, q_atts):
    global earth_vis_grid, alts, log_earth_vis_dists
    import astropy_healpix
    from xija.component.heat import get_dists_lons_lats
    import astropy.units as u

    healpix = astropy_healpix.HEALPix(nside=32, order='nested')

    if earth_vis_grid is None:
        with fits.open(earth_vis_grid_path) as hdus:
            hdu = hdus[0]
            hdr = hdu.header
            earth_vis_grid = hdu.data / hdr['scale']  # 12288 x 100

            alts = np.logspace(np.log10(hdr['alt_min']),
                               np.log10(hdr['alt_max']),
                               hdr['n_alt'])
            log_earth_vis_dists = np.log(hdr['earthrad'] + alts)

    ephems = ephems.astype(np.float64)
    dists, lons, lats = get_dists_lons_lats(ephems, q_atts)

    hp_idxs = healpix.lonlat_to_healpix(lons * u.rad, lats * u.rad)

    # Linearly interpolate distances for appropriate healpix pixels.
    # Code borrowed a bit from Ska.Numpy.Numpy._interpolate_vectorized.
    xin = log_earth_vis_dists
    xout = np.log(dists)
    idxs = np.searchsorted(xin, xout)

    # Extrapolation is linear.  This should never happen in this application
    # because of how the grid is defined.
    idxs = idxs.clip(1, len(xin) - 1)

    x0 = xin[idxs - 1]
    x1 = xin[idxs]

    # Note here the "fancy-indexing" which is indexing into a 2-d array
    # with two 1-d arrays.
    y0 = earth_vis_grid[hp_idxs, idxs - 1]
    y1 = earth_vis_grid[hp_idxs, idxs]

    esa = (xout - x0) / (x1 - x0) * (y1 - y0) + y0

    return esa


def generate_worst_targets(times, ephem, prng=None, num_targs=5000):
    from tqdm.auto import trange
    if prng is None:
        prng = np.random
    elif isinstance(prng, int):
        prng = RandomState(prng)

    worst_q = []
    t = CxoTime(times)
    sun = get_body('sun', t)
    for i in trange(t.size):
        ra = 360.0*prng.uniform(low=0.0, high=1.0, size=num_targs)
        dec = np.arccos(prng.uniform(low=-1.0, high=1.0, size=num_targs))
        dec = np.rad2deg(dec)-90.0
        tcoord = SkyCoord(ra=ra, dec=dec, unit='deg')
        pitch = sun[i].separation(tcoord).to_value("deg")
        good = (pitch > 46.2) & (pitch < 177.9)
        ra = tcoord.ra.to_value("deg")[good]
        dec = tcoord.dec.to_value("deg")[good]
        sun_ra = sun[i].ra.to_value('deg')
        sun_dec = sun[i].dec.to_value('deg')
        roll = np.array([
            Ska.Sun.nominal_roll(r, d, sun_ra=sun_ra, sun_dec=sun_dec)
            for r, d in zip(ra, dec)
        ])
        roll += prng.uniform(low=-10.0, high=10.0, size=ra.size)
        roll[roll < 0.0] += 360.0
        roll[roll > 360.0] -= 360.0
        eq = np.array([ra, dec, roll]).T
        q = Quat(equatorial=eq).q
        e = np.array([ephem[:, i]]*ra.size)
        illum = calc_earth_vis_from_grid(e, q)
        imin = np.argmax(illum)
        worst_q.append(q[imin,:])

    return np.array(worst_q)


def generate_constant_targets(tstart, num_targs, prng=None):
    if prng is None:
        prng = np.random
    elif isinstance(prng, int):
        prng = RandomState(prng)

    ra_t = np.array([])
    dec_t = np.array([])

    t = CxoTime(tstart)
    sun = get_body('sun', t)

    while len(ra_t) < num_targs:
        ra = 360.0*prng.uniform(low=0.0, high=1.0, size=num_targs)
        dec = np.arccos(prng.uniform(low=-1.0, high=1.0, size=num_targs))
        dec = np.rad2deg(dec)-90.0
        tcoord = SkyCoord(ra=ra, dec=dec, unit='deg')
        pitch = sun.separation(tcoord).to_value("deg")
        good = (pitch > 46.2) & (pitch < 177.9)
        ra_t = np.append(ra_t, tcoord.ra.to_value("deg")[good])
        dec_t = np.append(dec_t, tcoord.dec.to_value("deg")[good])

    sun_ra = sun.ra.to_value('deg')
    sun_dec = sun.dec.to_value('deg')

    roll_t = [
        Ska.Sun.nominal_roll(ra, dec, sun_ra=sun_ra, sun_dec=sun_dec)
        for ra, dec in zip(ra_t, dec_t)
    ]

    eq = np.array([ra_t, dec_t, roll_t]).T
    q = Quat(equatorial=eq).q

    return q
