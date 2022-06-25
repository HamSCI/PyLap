#M
#M PHaRLAP toolbox documentation
#M
#M Name :
#M   PHaRLAP - Provision of High-frequency Raytracing LAboratory for Propagation
#M             studies
#M
#M Purpose :
#M   Provides 2D and 3D numerical raytracing engines and supporting routines 
#M   (geomagnetic, ionospheric, etc.) to enable modeling of radio wave
#M   propogation through the ionosphere. Geometric focussing, ionospheric
#M   absorption (SiMIAN, George and Bradley, and deviative),
#M   ground forward scatter and backscatter losses, backscattered loss due to
#M   field aligned irregularites, ray becoming evanescent, O-X splitting,
#M   polarization, Doppler shift and Doppler spread are considered. Raytracing 
#M   through Es patches is not considered (however see ray_test2.m as a
#M   potential method for modelling Es). The ray trajectories and the full
#M   state-vector of the ray at each  point along the ray trajectory is
#M   available. User modified state-vectors may be input to the raytracing
#M   routines for advanced ray studies. The coordinate system used is WGS84. 
#M
#M Supplied routines (type help <routine name> for further help) :
#M   
#M   Core routines
#M   -------------
#M   raytrace_2d         - 2D numerical raytracing engine for a multihop ray 
#M                         (assumes WGS84 coordinate system)
#M   raytrace_2d_sp      - 2D numerical raytracing engine for a multihop ray 
#M                         (assumes spherical Earth coordinate system).
#M   raytrace_2d_gm      - uses raytrace_2d and applies O-X correction factor
#M                         calculated from gm_freq_offset so that approximations
#M                         for O and X mode 2D rays may be obtained
#M   raytrace_3d         - full 3D magneto-iono numerical raytrace engine for
#M                         O-mode, X-mode and "no-field" rays (assumes WGS84
#M                         coordinate system)
#M   raytrace_3d_sp      - full 3D magneto-iono numerical raytrace engine for
#M                         O-mode, X-mode and "no-field" rays (assumes
#M                         spherical Earth coordinate system with user
#M                         specified radius)
#M   abso_bg             - ionospheric absorption via George and Bradley
#M   abso_simian_3dnrt   - calculates absorption via SiMIAN
#M   chapman             - calculates ionospheric plasma profiles based on
#M                         Chapman layers
#M   dop_spread_eq       - simple model of Doppler spread imposed on ray 
#M                         traversing the equatorial region
#M   eff_coll_freq_neutrals  - calculates the effective electron collision
#M                             frequency with various neutral species 
#M   eff_coll_freq_ion   - the effective electron-ion collision frequency
#M   eff_coll_freq       - calculates the effective electron collision frequency
#M   ground_bs_loss      - power loss of the radio-waves back-scattered from
#M                         the ground
#M   ground_fs_loss      - calculates forward ground scattering losses
#M   gen_iono_grid_2d    - generates ionospheric parameters array, ionospheric 
#M                         plasma density grid, and irregularity strength in
#M                         format required by the 2D raytracing engine - uses
#M                         iri2007
#M   gen_iono_grid_3d    - generates ionospheric parameters array, ionospheric 
#M                         plasma density grid, geomagnetic field grid, and 
#M                         irregularity strength in format required by the 3D 
#M                         raytracing engine - uses iri2007 and igrf2007  
#M   gm_freq_offset      - calculates the approximate geomagnetic O-X mode
#M                         frequency split (MHz) for a specified propagation 
#M                         path.
#M   igrf2007            - International Geomagnetic Reference Field (distributed
#M                         with IRI2007).
#M   igrf2011            - International Geomagnetic Reference Field (distributed
#M                         with IRI2012).
#M   igrf2016            - International Geomagnetic Reference Field (distributed
#M                         with IRI2016).
#M   iri2007             - International Reference Ionosphere (2007)
#M   iri2012             - International Reference Ionosphere (2012) 
#M   iri2016             - International Reference Ionosphere (2016) 
#M   iri2012_firi_interp - calls International Reference Ionosphere (2012)
#M                         with FIRI rocketsonde-based lower ionosphere and
#M                         performs interpolation/smooth to remove discontinuity
#M   iri2016_firi_interp - calls International Reference Ionosphere (2016)
#M                         with FIRI rocketsonde-based lower ionosphere and
#M                         performs interpolation/smooth to remove discontinuity
#M   irreg_strength      - simple model of irregularity strength in the
#M                         equatorial and auroral regions
#M   land_sea            - returns land or sea for given location on Earth
#M   nrlmsise00          - NRLMSISe-00 model atmos. (distributed with IRI2016)
#M   noise_ccir          - CCIR (now ITU) environmental noise model
#M   plot_ray_iono_slice - plots ionospheric slice in an arc (to preserve
#M                         Earth geometry) and overplots rays
#M
#M   Ancilliary routines
#M   -------------------
#M   coning              - calulates correction to azimuth of ray due to the  
#M                         cone effect of linear arrays 
#M   deriv               - calculates derivative via 3-point, Lagrangian 
#M                         interpolation
#M   earth_radius_wgs84  - returns the WGS84 radius of the Earth at input 
#M                         geodetic latitude
#M   ENU2xyz             - convert the location of a point specified in an East, 
#M                         North, Up frame at a local origin on the Earth to
#M                         cartesian coordinates (x, y, z) 
#M   julday              - calculates the Julian Day Number 
#M   latlon2raz          - converts spherical Earth and longitude to range and 
#M                         azimuth with respect to an origin for various geoids
#M   raz2latlon          - converts range and azimuth from origin to spherical 
#M                         Earth latitude and longitude for various geoids
#M   solar_za            - returns the solar zenith angle
#M   wgs842gc_lat        - convert WGS84 geodetic latitude to geocentric
#M                         latitude  
#M   wgs84_llh2xyz       - converts WGS64 geodetic lat., long. and height to 
#M                         Earth centred x, y, z coordinates
#M   wgs84_xyz2llh       - converts Earth centred x, y, z coordinates to WGS84
#M                         geodetic lat., long. and height
#M   xyz2ENU             - convert the location of a point specified in an
#M                         cartesian coordinate (x, y, z) frame at a local
#M                         origin on the Earth to East, North, Up coordinates.
#M
#M   Examples
#M   --------
#M   ray_test1           - simple 2D NRT example showing fan of rays using
#M                         WGS84 coordinates
#M   ray_test2           - 2D NRT example showing ray state vector modification
#M   ray_test3           - simple 2D NRT showing fan of rays using spherical
#M                         Earth corrdinates
#M   ray_test4           - simple 2D NRT showing rays with different carrier
#M                         frequencies
#M   ray_test_3d         - 3D NRT (WGS84 coordinates) example showing O, X and 
#M                         no-field rays
#M   ray_test_3d_sp      - 3D NRT (spherical Earth coordinates) example
#M                         showing O, X and no-field rays
#M   ray_test_3d_pol     - 3D NRT for single ray with polarization calculations 
#M   ray_test_3d_iono_tilt - 3D NRT thorough a "tilted" ionosphere
#M   ois_synth           - example of synthetic single hop OIS ionogram
#M                         generation using 2D numerical raytracing
#M   ois_synth_mh        - multi-hop OIS ionogram synthesis with GUI
#M   bss_synth           - example of multi-hop back-scatter ionogram synthesis
#M   NRT_comparision     - comparison of a fan of rays using 2D and 3D NRT in an
#M                         ionosphere with no cross-range gradients
#M   NRT_ART_Comparison  - comparison of a fan of rays using ART, 2D NRT, 3D NRT
#M                         in a spherically symmetric ionosphere
#M   abso_comp_2dnrt     - compares ionospheric absorption calculated by
#M                         various methods for 2D NRT
#M   abso_comp_3dnrt     - compares ionospheric absorption calculated by
#M                         various methods for 3D NRT
#M
#M
#M Notes:
#M   1. The routines irreg_strength and dop_spread_eq are required for 
#M      propagation studies where Doppler spread is desired. Currently
#M      they are based on very simple models. Treat the these results with
#M      caution. See M.A. Cervera for further details. 
#M
#M Author:
#M   M.A. Cervera  16/06/2006
#M   Last update to this file:  24/10/2018 (M.A. Cervera)
#M
#M Modification History:
#M   See release notes.
#M
#     W. C. Liles 06/08/2020
#    Convert to Python comments



#M  This file constitutes documentation only.
