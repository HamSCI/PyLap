/* 
  Name : 
    iono_structure.h

  Purpose :
    Header file containing structures required to compile PHaRLAP mex files

  Warning :
    Any changes made to this include file must be followed by a rebuild of
    every module that references this file and programs using those modules.

  Modification History :
    Based on iono_structures.for      
                             
    20-04-2007 M.A. Cervera  1.00  
      Converted iono_structures.for code to C

    16-04-2008 M.A. Cervera  1.01  
       Increased size of ionospheric grid

    28-04-2008  M. A. Cervera 
      Defunct includes (for support of defunct code) removed
*/


#define max_num_ht 3001
#define max_num_rng 3001

/* structure containing ionospheric parameters */
struct ionosphere_struct{

  /* For each range: the vertical electron densities (electrons/cm^3) profile */
  double eN[max_num_rng][max_num_ht];

  /* For each range: the vertical electron density profiles 5 minutes later */
  double eN_5[max_num_rng][max_num_ht];

  /* For each range: the collision frequency height profile */  
  double col_freq[max_num_rng][max_num_ht];
  
  double HtMin;                     /* minimum height for array eN */
  double HtInc;                     /* height increment for array eN */
  double dRange;                    /* range increment for arrays eN and fcE */
  int    nRange;                    /* number of ranges for arrays eN and fcE */
  int    NumHt;                     /* number of height for array eN */

  /* strength of the ionospheric irregularities (ratio of irregular electon density 
     to the background value */  
  double irreg_strength[max_num_rng];

  /* azimuth and dip of the semi-major axis */ 
  double irreg_sma_azim[max_num_rng];

  /* of the ionospheric irregularities (rods  aligned with geomagnetic field) */
  double irreg_sma_dip[max_num_rng];

  /* square of frequency spread (Hz^2) per unit path length (Km) at a carrier 
     frequency of 1MHz scaled by the electron density cm^-3) */
  double dop_spread_sq[max_num_rng];   
};

static struct ionosphere_struct ionosphere;