# rc_track_params.py
"""
Shared geometry configuration for the rounded-rectangle track.

All lengths in meters.
"""

TRACK_CFG = {
    # outer dimensions of the track centerline rectangle
    "length": 2.0,          # long side (x direction)
    "width": 1.0,           # short side (y direction)
    "corner_radius": 0.333,   # radius of rounded corners
    # line appearance
    "line_width": 0.02,     # physical width of dashed line
    "dash_length": 0.06,     # length of a dash
    "dash_gap": 0.05,        # gap between dashes
    "line_height": 0.001,    # thickness in Z
}
