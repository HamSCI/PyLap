#!/usr/bin/env python3
"""
Test IRI-2020 ionosphere model (works without PHaRLAP)

This script demonstrates the IRI-2020 integration and can be run
before PHaRLAP is installed to verify the ionosphere model works.
"""

import datetime
import numpy as np
import matplotlib.pyplot as plt

# Import iri2020 package
import iri2020

def main():
    # Test parameters
    lat = 40.0      # Latitude (degrees)
    lon = -75.0     # Longitude (degrees)
    dt = datetime.datetime(2024, 6, 21, 12, 0)  # Summer solstice, noon UTC
    
    # Height range (km) - IRI2020 expects [min, max, step]
    alt_min = 100
    alt_max = 600
    alt_step = 5
    heights = [alt_min, alt_max, alt_step]
    
    print(f"Running IRI-2020 for:")
    print(f"  Location: {lat}°N, {lon}°E")
    print(f"  Time: {dt}")
    print(f"  Heights: {alt_min} - {alt_max} km (step: {alt_step} km)")
    print()
    
    # Call IRI-2020
    print("Calling IRI-2020...")
    result = iri2020.IRI(dt, heights, lat, lon)
    
    # Reconstruct height array for plotting
    height_arr = np.arange(alt_min, alt_max + alt_step, alt_step)
    
    # Extract electron density
    ne = result['ne'].values  # electrons/m^3
    
    # Find F2 peak
    f2_idx = np.nanargmax(ne)
    f2_height = height_arr[f2_idx]
    f2_density = ne[f2_idx]
    
    # Calculate plasma frequency at F2 peak
    # fp = sqrt(ne * e^2 / (4 * pi^2 * epsilon_0 * m_e))
    # Simplified: fp (MHz) = 9e-6 * sqrt(ne)
    fp_f2 = 9e-6 * np.sqrt(f2_density)
    
    print(f"Results:")
    print(f"  F2 peak height: {f2_height:.1f} km")
    print(f"  F2 peak density: {f2_density:.2e} electrons/m³")
    print(f"  F2 critical frequency (foF2): {fp_f2:.2f} MHz")
    print()
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Electron density profile
    ax1.plot(ne / 1e11, height_arr, 'b-', linewidth=2)
    ax1.axhline(y=f2_height, color='r', linestyle='--', label=f'F2 peak: {f2_height:.0f} km')
    ax1.set_xlabel('Electron Density (×10¹¹ m⁻³)')
    ax1.set_ylabel('Height (km)')
    ax1.set_title(f'IRI-2020 Electron Density Profile\n{lat}°N, {lon}°E, {dt}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Temperature profiles
    Te = result['Te'].values
    Ti = result['Ti'].values
    ax2.plot(Te, height_arr, 'r-', linewidth=2, label='Electron Temp (Te)')
    ax2.plot(Ti, height_arr, 'b-', linewidth=2, label='Ion Temp (Ti)')
    ax2.set_xlabel('Temperature (K)')
    ax2.set_ylabel('Height (km)')
    ax2.set_title('IRI-2020 Temperature Profiles')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('iri2020_test_output.png', dpi=150)
    print("Plot saved to: iri2020_test_output.png")
    plt.show()

if __name__ == '__main__':
    main()
