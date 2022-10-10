import numpy as np
from scipy.spatial import distance


def _sib_comp_dist(coords, ref_coords):
    """
    Compute the distance of subhaloes relative to reference object.

    Parameters
    ----------
    coords : array
        Coordinates of objects in catalog
    ref_coords : array
        The reference coordinates we compute the distance to

    Returns
    -------
    - : array
        3D Euclidean distances between coords and ref_coords
    """

    return distance.cdist(coords, ref_coords.reshape(1, 3), metric="euclidean").reshape(
        len(
            coords,
        )
    )


def _sib_comp_ra_dec(coords, obs_coords, distances):
    """
    Compute Right Ascention and Declination for catalog objects relative to the observer.

    Parameters
    ----------
    coords : array
        Coordinates of catalog objects
    obs_coords : array
        Coordinates of the observer
    distances : array
        3D distance to the observer

    Returns
    -------
    ra_rad / dec_rad : array
        Right Ascention / Declination in radians
    ra_deg / dec_deg : array
        Right Ascention / Declination in degrees
    """

    coords = coords - obs_coords
    dec_rad = np.arcsin(coords[:, 0] / distances)
    ra_rad = np.arctan2(coords[:, 1], coords[:, 2])

    # Convert to degrees.
    dec_deg = np.degrees(dec_rad)  # -90 -> + 90 degrees
    ra_deg = np.degrees(ra_rad)
    ra_deg[ra_deg < 0] += 360.0  # 0 -> 360 degrees

    return ra_rad, dec_rad, ra_deg, dec_deg


def _sib_comp_velocity(coords, vels, dists, obs_coords, obs_vel, H):
    """
    Compute radial and tangential vels relative to the observer.

    Parameters
    ----------
    coords : array
        Coordinates of catalog objects
    vels : array
        Velocities of catalog objects
    dists : array
        3D distances to catalog objects from observer
    obs_coords : array
        Coordinates of observer
    obs_vel : array
        Velocity of observer
    H : float
        Hubble param (h * 100)

    Returns
    -------
    vr : array
        Radial velocity wrt observer
    vt : array
        Tangential velocity wrt observer
    """

    dx = coords - obs_coords
    r = dists
    dx = dx / r.reshape(len(r), 1)
    dv = vels - obs_vel

    vr = np.einsum("ij,ij->i", dx, dv)
    vt = np.sqrt(np.linalg.norm(dv, axis=1) ** 2.0 - vr**2.0)
    vr = vr + r * H

    return vr, vt


def _sib_comp_galactic(ra, dec):
    """
    Convert ra dec (equitorial) to galactic latitude and longitude (b, l).

    Parameters
    ----------
    ra : array
        Right ascention of catalog objects
    dec : array
        Declination of catalog objects

    Returns
    -------
    l : array
        Galactic latitude
    b : array
        Galactic longitude
    """

    # J2000
    dec0 = np.radians(27.1284)
    ra0 = np.radians(192.8595)
    l0 = np.radians(122.9320)

    A = np.sin(dec) * np.sin(dec0)
    B = np.cos(dec) * np.cos(dec0) * np.cos(ra - ra0)

    b = np.arcsin(A + B)  # -pi/2 - > pi/2

    A = np.cos(dec) * np.sin(ra - ra0)
    B = np.sin(dec) * np.cos(dec0) - np.cos(dec) * np.sin(dec0) * np.cos(ra - ra0)

    l = l0 - np.arctan2(A, B)

    l[l < 0] += 2 * np.pi
    l[l > 2 * np.pi] -= 2 * np.pi  # 0 -> 2pi

    return l, b


def _sib_compute_apparent_mag(M, d):
    """
    Compute apparent magnitude from absolute magnitude and distance.

    Parameters
    ----------
    M : array
        Absolute magnitude
    d : array
        3D distance to the observer

    Returns
    -------
    - : array
        Apparent magnitude
    """

    # M is pure magnitude units, and d is in Mpc.

    d_rat = d * 1e6 / 10.0
    return M + 5 * np.log10(d_rat)


def _refactor_catalog_data(data, catalog_type):
    """
    Construct coordinate and velocity arrays depending on the source catalog.

    Parameters
    ----------
    data : dict
        Catalog data
    catalog_type : string
        Type of catalog (hbt, galform or velociraptor)

    Returns
    -------
    c : array
        Coordinates of catalog objects
    v : array
        Velocities of catalog objects
    """

    v = None
    if catalog_type == "galform":
        c = np.c_[data["xgal"], data["ygal"], data["zgal"]]
        if "vxgal" in data.keys():
            v = np.c_[data["vxgal"], data["vygal"], data["vzgal"]]
    elif catalog_type == "hbt":
        c = data["ComovingMostBoundPosition"]
        if "PhysicalAverageVelocity" in data.keys():
            v = data["PhysicalAverageVelocity"]
    elif catalog_type == "velociraptor":
        c = np.c_[data["Xc"], data["Yc"], data["Zc"]]
        if "VXc" in data.keys():
            v = np.c_[data["VXc"], data["VYc"], data["VZc"]]

    return c, v


def compute_sibelius_properties(
    data,
    catalog_type,
    compute_distance,
    compute_ra_dec,
    compute_velocity,
    compute_galactic,
    compute_apparent_mag,
    observer,
    h=0.6777,
):
    """
    Main function to append data array with Sibelius info.

    Adds the new computed properties to the data array.

    Parameters
    ----------
    data : dict
        Catalog data from HBT, Velociraptor or GALFORM
    catalog_type : string
        Source of catalog ('hbt', 'velociraptor' or 'galform')
    compute_distance : bool
        True to compute 3D distance to observer for each catalog object
    compute_ra_dec : bool
        True to compute RA/DEC for each catalog object
    compute_velocity : bool
        True to compute radial and tangential velocities of catalog objects
        relative to obsever
    compute_galactic : bool
        True to compute galactic latitude and longitude for each catalog object
    compute_apparent_mag : bool
        True to compute apparent magnitudes as viewed from the observer for
        given absolute magnitues
    observer : string or coordinate vector
        Default observer is "sibelius_dark_mw", can overwrite this by passing
        manual observer coordinates
    h : float
        Value of hubble param for the simulation

    Appends data array with properties
    ----------------------------------
    compute_distance:
        Adds 'distance'
    compute_ra_dec:
        Adds 'ra_deg', 'dec_deg', 'ra_rad' and 'dec_rad'
    compute_velocity:
        Added 'vt' and 'vr'
    compute_galactic:
        Adds 'l_deg', 'b_deg', 'l_rad' and 'b_rad'
    compute_apparent_mag:
        Adds 'mag_*_app' for each absolute magnitude array
    """

    # Allowed catalog formats.
    _allowed_cats = ["galform", "hbt", "velociraptor"]
    assert catalog_type in _allowed_cats, f"{catalog_type} is a bad catalog type"

    # Coordinates and velocities of catalog objects
    c, v = _refactor_catalog_data(data, catalog_type)

    # What is the position of the observer.
    if observer == "sibelius_dark_mw":
        obs_coords = np.array([499.34264252, 504.50740204, 497.31107545])
        obs_vel = np.array([-12.43349451, 350.16214811, -152.84008118])
    else:
        assert len(observer) == 3, "Bad observer loc"
        obs_coords = observer
        obs_vel = None

    # Compute distance between catalog objects and the observer.
    if compute_distance:
        data["distance"] = _sib_comp_dist(c, obs_coords)

    # Compute RA/DEC of catalog objects.
    if compute_ra_dec:
        assert compute_distance, "Need compute_distance for compute_ra_dec"

        ra_rad, dec_rad, ra_deg, dec_deg = _sib_comp_ra_dec(
            c, obs_coords, data["distance"]
        )

        data["ra_rad"] = ra_rad
        data["dec_rad"] = dec_rad
        data["ra_deg"] = ra_deg
        data["dec_deg"] = dec_deg

    # Compute radial and tangential velocities.
    if compute_velocity:
        assert v is not None, "No velocities loaded"
        assert obs_vel is not None, "No observer velocity"

        vr, vt = _sib_comp_velocity(
            c, v, data["distance"], obs_coords, obs_vel, h * 100.0
        )

        data[f"vr"] = vr
        data[f"vt"] = vt

    # Compute apparent magnitudes.
    if compute_apparent_mag:
        assert compute_distance, "Need compute_distance for compute_apparent_mag"

        for att in list(data.keys()):
            if "mag_" in att:
                print(att)
                data[att + "_app"] = _sib_compute_apparent_mag(
                    data[att], data["distance"]
                )

    # Compute galactic coordinates from ra/dec.
    if compute_galactic:
        assert compute_ra_dec, "Need compute_ra_dec for compute_galactic"

        l, b = _sib_comp_galactic(data["ra_rad"], data["dec_rad"])
        data["l_rad"] = l
        data["b_rad"] = b
        data["l_deg"] = np.degrees(data["l_rad"])
        data["b_deg"] = np.degrees(data["b_rad"])
