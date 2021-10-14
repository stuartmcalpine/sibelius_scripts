import numpy as np
from scipy.spatial import distance

class SibeliusInfo:
    """ Coordinates of MW, M31, virgo and coma in the Sibelius_200Mpc_1 run.
        Positions are ComovingMostBoundPosition, vels are PhysicalAverageVelocity """

    def __init__(self):

        self.loaded_objects = []

        self.add_object('mw', np.array([499.34264252, 504.50740204, 497.31107545]),
                np.array([-12.43349451, 350.16214811, -152.84008118]), 17791952)
        self.add_object('m31', np.array([499.97779324, 504.68340428, 497.67532689]),
                np.array([-171.075620, 311.68471382, -205.25150656]), 5098129)
        self.add_object('coma', np.array([553.6426445, 477.52715798, 407.73658318]),
                np.array([-90.64376762, 116.13374223, -188.75870998]), 337006)
        self.add_object('virgo', np.array([505.13311359, 498.62148024, 477.93478113]),
                np.array([-320.46377981, 255.85761576, -313.29669271]), 58233)
        self.add_object('lg', (self.mw.coords + self.m31.coords) /2.,
                (self.mw.vels + self.m31.vels) /2., 0)

    def add_object(self, name, coords, vels, trackid):
        """ Add an object to the list ."""

        setattr(self, name, _SibeliusInfo(name, coords, vels, trackid))
        self.loaded_objects.append(name)

class _SibeliusInfo:

    def __init__(self, name, coords, vels, trackid):
        self.name       = name
        self.coords     = coords
        self.vels       = vels
        self.trackid    = trackid

def sib_comp_dist(coords, ref_coords):
    """ Compute the distance of subhaloes relative to reference object. """

    return distance.cdist(coords, ref_coords.reshape(1,3),
            metric='euclidean').reshape(len(coords,))

def sib_comp_ra_dec(coords, mw_coords, mw_distance):
    """ Compute position on the sky of subhaloes. """
    
    coords = coords - mw_coords
    dec_rad = np.arcsin(coords[:,0] / mw_distance)   # Polar. [-pi/2,pi/2]
    ra_rad = np.arctan2(coords[:,1], coords[:,2])          # Azimuthal [-pi, pi]

    # Convert to degrees.
    dec_deg = np.degrees(dec_rad)     # -90 -> + 90 degrees
    ra_deg = np.degrees(ra_rad)
    ra_deg[ra_deg < 0] += 360.        # 0 -> 360 degrees

    return ra_rad, dec_rad, ra_deg, dec_deg

def sib_comp_velocity(coords, vels, dists, ref_coords, ref_vel, H):
    """ Compute radial and tangential vels relative to reference object. """

    dx = coords - ref_coords
    r = dists
    dx = dx / r.reshape(len(r),1)
    dv = vels - ref_vel

    vr = np.einsum('ij,ij->i', dx, dv)
    vt = np.sqrt(np.linalg.norm(dv,axis=1)**2. - vr**2.)
    vr = vr + r * H

    return vr, vt

def compute_angsep(ra, dec, ref_ra, ref_dec):
    """ Compute angular separation between subhaloes and passed ra/dec.
        Passed ra/dec must be in radians. 
        Returned angsep is in radians also. """
    
    angsep = np.arccos(np.sin(ref_dec)*np.sin(dec)+ np.cos(ref_dec)*np.cos(dec)*np.cos(ref_ra-ra))

    return angsep

def compute_observed_clustercentric_velocity(ra, dec, d_mw, vr_mw, ref_ra, ref_dec, ref_d, ref_vr,
        distance_errors=0.0):
    """ For a given cluster, compute the radial distance and velocity from the 
        centre of that cluster to all other subhaloes.

        This is how an observer would estimate the values.

        Uses the formulism from Karachentsev & Kashibadze (2006).

        The passed "ref" values are of the cluster you are interested in,
        and the values are all relative to the MW.

        Can add distance errors, the other 3 vars are assumed to have no error.
    """

    # First compute the angsep between the cluster centre and subhaloes.
    theta = compute_angsep(ra, dec, ref_ra, ref_dec)

    # Now compute observed clustercentric distance.
    D = ref_d # Distance from cluster to MW.
    Dn = d_mw
    Dn_err = np.random.rand(len(Dn)) * distance_errors * Dn
    Dn = d_mw + Dn_err
    Rc = np.sqrt(D**2 + Dn**2 - 2*D*Dn*np.cos(theta))

    # Now compute observed clustercentric radial velocity.
    Vc = np.zeros(len(Rc), dtype='f4')
    V = ref_vr # Radial velocity of cluster (w.r.t MW).
    Vn = vr_mw 
    mask = np.where(Vn != ref_vr)
    lam = D*np.sin(theta[mask]) / (Dn[mask] - D*np.cos(theta[mask]))
    mu = lam + theta[mask]

    Vc[mask] = Vn[mask]*np.cos(lam) - V*np.cos(mu)

    return Rc, Vc

def sib_comp_galactic(ra, dec):
    """ Convert ra dec (equitorial) to galactic latitude and longitude (b, l) """

    # J2000
    dec0 = np.radians(27.1284)
    ra0 = np.radians(192.8595)
    l0 = np.radians(122.9320)

    A = np.sin(dec)*np.sin(dec0)
    B = np.cos(dec)*np.cos(dec0)*np.cos(ra-ra0)

    b = np.arcsin(A + B) # -pi/2 - > pi/2

    A = np.cos(dec)*np.sin(ra - ra0)
    B = np.sin(dec)*np.cos(dec0) - np.cos(dec)*np.sin(dec0)*np.cos(ra-ra0)

    l = l0 - np.arctan2(A,B)
    
    l[l < 0] += 2*np.pi
    l[l > 2*np.pi] -= 2*np.pi # 0 -> 2pi

    return l, b

#def sib_comp_supergalactic(ra, dec):
#    """ Convert ra dec (equitorial) to supergalactic latitude and longitude (b, l) """
#
#    # J2000
#    dec0 = np.radians(27.1284)
#    ra0 = np.radians(192.8595)
#    l0 = np.radians(122.9320)
#
#    A = np.sin(dec)*np.sin(dec0)
#    B = np.cos(dec)*np.cos(dec0)*np.cos(ra-ra0)
#
#    b = np.arcsin(A + B) # -pi/2 - > pi/2
#
#    A = np.cos(dec)*np.sin(ra - ra0)
#    B = np.sin(dec)*np.cos(dec0) - np.cos(dec)*np.sin(dec0)*np.cos(ra-ra0)
#
#    l = l0 - np.arctan2(A,B)
#    
#    l[l < 0] += 2*np.pi
#    l[l > 2*np.pi] -= 2*np.pi # 0 -> 2pi
#
#    return l, b

def sib_compute_apparent_mag(M, d):
    """ Compute apparent magnitude from absolute magnitude and distance. """

    # M is pure magnitude units, and d is in Mpc.

    d_rat = d * 1e6 / 10.
    return M + 5*np.log10(d_rat)

def compute_sibelius_properties(data, from_what, compute_distance, compute_ra_dec,
        compute_velocity, compute_galactic, compute_apparent_mag,
        compute_extra_coordinates, compute_extra_objects, h=0.6777):
    """ Main function to append data array with Sibelius info. """

    # Contains positions of known objects from the Sibelius_200Mpc_1 run.
    sib = SibeliusInfo()

    if from_what.lower() == 'galform':
        c = np.c_[data['xgal'], data['ygal'], data['zgal']]
        if 'vxgal' in data.keys():
            v = np.c_[data['vxgal'], data['vygal'], data['vzgal']]
    elif from_what.lower() == 'hbt':
        c = data['ComovingMostBoundPosition']
        if 'PhysicalAverageVelocity' in data.keys():
            v = data['PhysicalAverageVelocity']
    else:
        raise ValueError("Bad from_what")   
 
    # Compute distances of subhaloes from known objects.
    if compute_distance:
        for obj in sib.loaded_objects:
            if obj != 'mw' and compute_extra_objects == False: continue
            data[f'd_{obj}'] = sib_comp_dist(c, getattr(sib, obj).coords)

    # Compute RA/DEC of subhaloes.
    if compute_ra_dec:
        assert compute_distance, "Need compute_distance for compute_ra_dec"

        ra_rad, dec_rad, ra_deg, dec_deg \
                = sib_comp_ra_dec(c, sib.mw.coords, data['d_mw'])

        data['ra_rad'] = ra_rad; data['dec_rad'] = dec_rad
        data['ra_deg'] = ra_deg; data['dec_deg'] = dec_deg

        if compute_extra_coordinates:
            # Simulation coordinates x and z are swapped, this returns true eq coordinates.
            # Coordinates are relative to MW.
            data['coords_eq'] = np.empty_like(c)
            data['coords_eq'][:,0] = c[:,2] - sib.mw.coords[2]
            data['coords_eq'][:,1] = c[:,1] - sib.mw.coords[1]
            data['coords_eq'][:,2] = c[:,0] - sib.mw.coords[0]

    # Compute radial and tangential velocities.
    if compute_velocity:
        for obj in sib.loaded_objects:
            if obj != 'mw' and compute_extra_objects == False: continue
            vr, vt = sib_comp_velocity(c, v, data[f'd_{obj}'],
                    getattr(sib, obj).coords, getattr(sib, obj).vels, h*100.)

            data[f'vr_{obj}'] = vr
            data[f'vt_{obj}'] = vt

    # Compute apparent magnitudes.
    if compute_apparent_mag:
        for att in list(data.keys()):
            if 'mag_' in att:
                data[att + '_app'] = sib_compute_apparent_mag(data[att], data['d_mw'])

    # Compute galactic coordinates from ra/dec.
    if compute_galactic:
        l, b = sib_comp_galactic(data['ra_rad'], data['dec_rad'])
        data['l_rad'] = l; data['b_rad'] = b
        data['l_deg'] = np.degrees(data['l_rad'])
        data['b_deg'] = np.degrees(data['b_rad'])
    
        if compute_extra_coordinates:
            data['coords_gal'] = np.empty_like(c)
            data['coords_gal'][:,0] = \
                data['d_mw'] * np.cos(data['l_rad']) * np.cos(data['b_rad'])
            data['coords_gal'][:,1] = \
                data['d_mw'] * np.sin(data['l_rad']) * np.cos(data['b_rad'])
            data['coords_gal'][:,2] = data['d_mw'] * np.sin(data['b_rad'])
