"""Thin orchestrator: runs preop alignment (steps 01-04) then postop registration (steps 05-06).

For preop-only or postop-only runs, use align_preop.py or register_postop.py directly.
"""
import os
import sys
from os.path import join, expanduser
import logging
from time import time

from spinescrews.tools.paths import (segmentation_dir, preop_dir, correspondence_dir,
                         orient_dir, detection_dir, registration_dir,
                         step_complete, read_summary)

from spinescrews.pipeline.align_preop import Aligner, run_preop
from spinescrews.pipeline.register_postop import Registrar, run_postop

log = logging.getLogger(__name__)


def _executive_summary(analysis_dir):
    """Read all step summaries and report any quality warnings."""
    warnings = []

    # Step 01 — segmentation
    seg_dir = segmentation_dir(analysis_dir)
    if not step_complete(seg_dir):
        warnings.append('Step 01 (segmentation) did not complete')

    # Step 02 — preop
    pre_dir = preop_dir(analysis_dir)
    if not step_complete(pre_dir):
        warnings.append('Step 02 (preop) did not complete')

    # Step 03 — correspondence
    corr_dir_ = correspondence_dir(analysis_dir)
    if not step_complete(corr_dir_):
        warnings.append('Step 03 (correspondence) did not complete')
    else:
        s = read_summary(corr_dir_)
        for level, m in s.get('per_level', {}).items():
            if m.get('status') == 'warning':
                warnings.append('%s: correspondence match sketchy (dg=%.3f, threshold 0.06)' % (level, m['dg']))
            cov = m.get('coverage')
            if cov is not None and cov < 0.5:
                warnings.append('%s: low correspondence coverage (%.0f%%)' % (level, cov * 100))

    # Step 04 — orient
    ori_dir = orient_dir(analysis_dir)
    if not step_complete(ori_dir):
        warnings.append('Step 04 (orient) did not complete')
    else:
        s = read_summary(ori_dir)
        for level, m in s.get('per_level', {}).items():
            if m.get('angle_deg', 0) > 25:
                warnings.append('%s: large orientation correction (%.1f deg)' % (level, m['angle_deg']))
            if m.get('trans_mm', 0) > 10:
                warnings.append('%s: large translation correction (%.1f mm)' % (level, m['trans_mm']))
            if m.get('anchor_weight', 1) < 0.5:
                warnings.append('%s: low anchor weight (%.2f) -- refinement uncertain' % (level, m['anchor_weight']))

    # Step 05 — detection
    det_dir_ = detection_dir(analysis_dir)
    if not step_complete(det_dir_):
        warnings.append('Step 05 (detection) did not complete')
    else:
        s = read_summary(det_dir_)
        n_plan = s.get('n_screws_planned', 0)
        n_det = s.get('n_screws_detected', n_plan)
        if n_det < n_plan:
            warnings.append('Only %d/%d screws detected' % (n_det, n_plan))
        for name, m in s.get('per_screw', {}).items():
            ir = m.get('inlier_ratio', 1.0)
            if ir < 0.5:
                warnings.append('%s: poor screw fit (%.0f%% inliers)' % (name, ir * 100))

    # Step 06 — registration
    reg_dir = registration_dir(analysis_dir)
    if not step_complete(reg_dir):
        warnings.append('Step 06 (registration) did not complete')
    else:
        s = read_summary(reg_dir)
        icp = s.get('icp', {})
        n_corr = icp.get('n_corrective', 0)
        if n_corr > 0:
            warnings.append('%d levels required corrective ICP realignment' % n_corr)
        for level, ratio in icp.get('per_level_ratios', {}).items():
            if ratio < 0.6:
                warnings.append('%s: low ICP alignment ratio (%.2f)' % (level, ratio))
        vol = s.get('volumetric', {})
        for w in vol.get('warnings', []):
            warnings.append('MI warning: %s' % w)
        for f in vol.get('failures', []):
            warnings.append('MI failure: %s' % f)

    # Report
    if warnings:
        log.info('=' * 40)
        log.info('=== PIPELINE WARNINGS ===')
        for w in warnings:
            log.info('  - %s' % w)
        log.info('%d warning(s) found -- review before trusting results' % len(warnings))
        log.info('=' * 40)
    else:
        log.info('=== Pipeline completed cleanly (steps 01-06) ===')


def main():
    """CLI entry point for full pipeline (steps 01-06). Called by spinescrews-align console script."""
    t0 = time()
    import argparse
    from spinescrews.tools.config import load_config, save_resolved_config

    parser = argparse.ArgumentParser(description='Run pipeline on a single CT scan.')
    parser.add_argument('specimen_dir', type=str)
    parser.add_argument('--debug', action='store_true', default=None)
    parser.add_argument('--n-jobs', type=int, default=None,
                        help='Number of CPU cores for parallel steps (-1=all, -3=all-but-2, etc.)')
    parser.add_argument('--no-patches', action='store_true', default=None,
                        help='Skip writing postop-reg.nii.gz volumes (saves ~59 MB/level)')
    args = parser.parse_args()

    overrides = {}
    if args.debug is not None:
        overrides['debug'] = args.debug
        overrides['n_jobs'] = 1
        overrides['mi_n_jobs'] = 1
    if args.n_jobs is not None:
        overrides['n_jobs'] = args.n_jobs
        overrides['mi_n_jobs'] = args.n_jobs
    if args.no_patches is not None:
        overrides['no_patches'] = args.no_patches

    config = load_config(args.specimen_dir, overrides=overrides)
    save_resolved_config(config)

    data_dir = expanduser(config.specimen_dir)
    analysis_dir = join(data_dir, config.output_dir)
    os.makedirs(analysis_dir, exist_ok=True)

    logfile = join(analysis_dir, 'pipeline.log')
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)
    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.DEBUG if config.debug else logging.INFO)
    logging.basicConfig(level=logging.DEBUG, force=True, handlers=[fh, sh])

    log.info('*' * (31 + len(data_dir)))
    log.info('**  Aligning vertebrae for %s  **' % data_dir)
    log.info('*' * (31 + len(data_dir)))

    run_preop(config)
    run_postop(config)

    _executive_summary(analysis_dir)

    log.info('*** Total time: %.2f minutes' % ((time() - t0) / 60))

if __name__ == '__main__':
    main()
