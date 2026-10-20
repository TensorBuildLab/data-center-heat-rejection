"""
reference_case.py - the reference night and the two reference campuses (manuscript Section 2.8).

Shared by the figure scripts in this folder so every figure uses identical inputs.

  Night   : Phoenix summer night, rural, Pasquill class F, U10 = 2 m/s, z_i = 150 m,
            T = 32 C, RH = 20 %, elevation 340 m.
  Campus  : 100 MW (IT) in two cooling technologies. The heat rejected differs because the
            air-cooled plant uses more electricity:
              air-cooled   - dry coolers, 137 MW, 12 K air-side rise, fans at 12 m,
                             600 m x 600 m footprint
              water-cooled - cooling towers, 125 MW, 1,650 m3/s inlet air, 15 m,
                             300 m x 200 m footprint
  Fan array: plume rise of the fan array follows Briggs' (1974) multiple-source merging
            (dc_plume HeatSource.fan_heat): 80 kW per dry-cooler fan with 3 m/s exit
            velocity, 8 MW per cooling-tower cell. Set DC_MERGED=1 to reproduce the earlier
            fully merged plume (fan_heat=None).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # repo root, for dc_plume

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dc_plume import Ambient, cooling_tower_source, dry_cooler_source

NIGHT = Ambient(T=32.0, RH=0.20, U10=2.0, stability="F", mixing_height=150.0, elevation=340.0)

MERGED = os.environ.get("DC_MERGED", "0") == "1"
FAN_HEAT_AIR = None if MERGED else 80e3      # W per dry-cooler fan
FAN_HEAT_TOWER = None if MERGED else 8e6     # W per cooling-tower cell

# total heat rejected per MW of IT load (W/W): air-cooled 1.37, water-cooled 1.25
HEAT_PER_IT = dict(air=1.37, water=1.25)


def air_cooled(amb=NIGHT, Q=137e6, side=600.0):
    """Dry-cooler campus rejecting Q watts from a square footprint of the given side (m)."""
    return dry_cooler_source(Q, 12.0, amb, height=12.0, footprint_x=side, footprint_y=side,
                             building_height=10.0, exit_velocity=6.0 if MERGED else 3.0,
                             fan_heat=FAN_HEAT_AIR)


def water_cooled(amb=NIGHT, Q=125e6, inlet_flow=1650.0, fx=300.0, fy=200.0):
    """Cooling-tower campus rejecting Q watts with the given inlet airflow (m3/s)."""
    return cooling_tower_source(Q, inlet_flow, amb, height=15.0, footprint_x=fx, footprint_y=fy,
                                building_height=10.0, fan_heat=FAN_HEAT_TOWER)


# figure style and colours used throughout the paper
AIR, WATER, GREY = "#c0392b", "#2471a3", "#7f8c8d"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})

FIG_DIR = os.path.join(os.getcwd(), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def fig_path(name):
    return os.path.join(FIG_DIR, name)
