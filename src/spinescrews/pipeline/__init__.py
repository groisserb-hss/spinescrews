"""Pipeline entry points for pedicle screw placement analysis.

- align_preop: Steps 01-04 (segmentation import, orientation, correspondence, refinement)
- register_postop: Steps 05-06 (screw detection, articulated + volumetric registration)
- align_vertebrae: Orchestrator calling align_preop then register_postop
- compute_accuracy: Step 07 (error measurement and breach analysis)
- run_segmentation: Step 01 (vertebral segmentation via external backend)
"""
