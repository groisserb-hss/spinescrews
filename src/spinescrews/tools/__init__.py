"""Core data structures and constants shared across the pipeline.

Defines the vertebra-label encoding (`seg_val` / `val_seg`), the canonical caudal-to-cranial
level ordering (`possible_levels`), the RAS axis indices (`dimR`, `dimA`, `dimS`), and the
measurement record types (`ScrewMeasures`, `BreachMeasures`, `MeshLabels`).
"""

from collections import namedtuple

seg_val = {'C1': 1, 'C2': 2, 'C3': 3, 'C4': 4, 'C5': 5, 'C6': 6, 'C7': 7,
           'T1': 8, 'T2': 9, 'T3': 10, 'T4': 11, 'T5': 12, 'T6': 13, 'T7': 14,
           'T8': 15, 'T9': 16, 'T10': 17, 'T11': 18, 'T12': 19, 'T13': 28,
           'L1': 20, 'L2': 21, 'L3': 22, 'L4': 23, 'L5': 24, 'LS': 25,
           'SA': 26, 'CX': 27, 'background': 0}

val_seg = {v: k for k, v in seg_val.items()}

# NOTE: a verbatim copy lives in dicom_tools/HybridScrewPlanner/HybridScrewPlanner.py
# (POSSIBLE_LEVELS) for export-time plan validation; keep the two in sync.
# The copy is checked against this list by dicom_tools/tests/test_possible_levels_sync.py.
possible_levels = ['LS', 'L5', 'L4', 'L3', 'L2', 'L1',
                   'T13', 'T12', 'T11', 'T10', 'T9', 'T8', 'T7', 'T6', 'T5',
                   'T4', 'T3', 'T2', 'T1', 'C7', 'C6', 'C5', 'C4', 'C3', 'C2']

ScrewMeasures = namedtuple('ScrewMeasures',
                       ['entry_x', 'entry_y', 'entry_z',
                        'ped_x', 'ped_z',
                        'tip_x', 'tip_y', 'tip_z',
                        'theta', 'theta_s', 'theta_c', 'theta_a'])

BreachMeasures = namedtuple('BreachMeasures', ['breach_dist', 'breach_angle', 'planned_breach_dist',
                                                'screw_pt_x', 'screw_pt_y', 'screw_pt_z',
                                                'ped_pt_x', 'ped_pt_y', 'ped_pt_z',
                                                'tip_distance_signed'])

MeshLabels = namedtuple('MeshLabels', ['left_ped', 'right_ped', 'canal', 'body_walls',
                                       'endplate_top', 'endplate_bottom'])

dimR, dimA, dimS = 0, 1, 2

