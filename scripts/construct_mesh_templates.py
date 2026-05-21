import os
from os.path import join, expanduser, isfile
from os import remove
from glob import glob
import logging

from spectral_match.pipeline import FunctionalMapper
from spinescrews.tools.correspondence_tools import SpectralDescriptor
from spectral_match.correspondence.deep_functional_maps.training import cache_and_train
from spectral_match import SigConfig

from bg3dtools.render.trimesh import scatt, scatts, trisurfsm, draw_geometries

log = logging.getLogger(__name__)


def build_template(template_level, sig_config, source_path, template_dir):
    """Load or train ResNet weights for one vertebral level's spectral template."""
    log.info('building average template for ' + template_level)

    weight_file = join(template_dir, 'resnet_weights', '%s_w%d_h%d_g%d' %
                       (template_level, sig_config.num_wks, sig_config.num_hks, sig_config.num_gaussian))

    # base-level functional mapper with anatomical signature
    mapper = FunctionalMapper(sig_config=sig_config, extra_sig_fun=SpectralDescriptor.anatomical_signature,
                               compute_spectral=True)
    mapper.describer = SpectralDescriptor(sig_config)

    chkpt_file = join(template_dir, 'resnet_weights', 'checkpoint')
    if isfile(chkpt_file): remove(chkpt_file)

    if isfile(weight_file + '.index'):
        log.info('Pretrained weights found, skipping training')
        mapper.load_resnet(weight_file)
    else:
        log.info('Training new weights')
        raw_files = glob(join(source_path, template_level, 'preop_seg.ply'))
        raw_files.sort()
        raw_files += [join(template_dir, 'meshes', 'template_%s.ply' % template_level)]
        mapper.res = cache_and_train(raw_files, weight_file, mapper, epochs=301)


if __name__ == "__main__":
    import argparse as _ap
    _parser = _ap.ArgumentParser(description='Build template meshes and train ResNet weights.')
    _parser.add_argument('--source_path', type=str, required=True,
                         help='Glob pattern for specimen analysis dirs (e.g. /path/to/specimen_*/analysis2)')
    _parser.add_argument('--template_dir', type=str,
                         default=join(os.path.dirname(os.path.abspath(__file__)), '..', 'vertebra_templates'))
    _args = _parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    log.info('starting')

    source_path = expanduser(_args.source_path)
    output_dir = expanduser(_args.template_dir)
    sig_config = SigConfig(emin=0.01, emax=1000., num_wks=75, num_hks=25, num_gaussian=18, num_signatures=121)

    for level in ['L5', 'L4', 'L3', 'L2', 'L1',
                  'T12', 'T11', 'T10', 'T9', 'T8', 'T7', 'T6', 'T5', 'T4', 'T3', 'T2', 'T1']:
        build_template(level, sig_config, source_path, output_dir)

    log.info('done')