"""Figure generation for each pipeline step.

- seg_overlay: 3-panel segmentation overlay (axial/coronal/sagittal)
- preop_orientation: Grid of oriented vertebrae with CT slices and meshes
- correspondence_preprocess: 4-panel geodesic distance visualization
- correspondence_match: 2x2 label and gradient correspondence QC
- orient_refinement: Bar charts of rotation/translation corrections
- spine_template: Anterior + lateral views of seg and template meshes
- detection_screws: Multi-angle MIP with detected screw lines
- detection_plan_vs_detected: Multi-angle MIP comparing planned vs detected screws
- CT_visualization: 4-panel CT visualization per vertebral level
- visualize_breach: 3-panel screw-aligned breach visualization
- group_statistics: Cross-specimen error analysis
"""
