"""Check the installed distribution, including when CI installs a built wheel."""
from importlib.metadata import distribution
from pathlib import Path

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
import pytest

import posecalib


@pytest.fixture(scope='module')
def installed_metadata():
    runtime_path = Path(posecalib.__file__).resolve()
    source_package = Path(__file__).resolve().parents[1] / 'src' / 'posecalib'
    if runtime_path.parent == source_package.resolve():
        pytest.skip('installed-wheel metadata checks do not apply to source imports')
    installed = distribution('posecalib')
    assert Path(installed.locate_file('posecalib/__init__.py')).resolve() == runtime_path, (
        'distribution metadata must belong to the imported posecalib wheel')
    return installed.metadata


def test_distribution_version_matches_runtime(installed_metadata):
    metadata = installed_metadata
    assert metadata['Name'] == 'posecalib'
    assert metadata['Version'] == posecalib.__version__


@pytest.mark.parametrize('python_version', ['3.9', '3.10', '3.11', '3.12', '3.13', '3.14'])
def test_python_support_and_optional_video_metadata(python_version, installed_metadata):
    metadata = installed_metadata
    assert python_version in SpecifierSet(metadata['Requires-Python'])
    assert 'video' in metadata.get_all('Provides-Extra', [])
    requirements = [Requirement(value) for value in metadata.get_all('Requires-Dist', [])]
    environment = default_environment()
    environment.update(python_version=python_version,
                       python_full_version=python_version + '.0')

    def active(extra):
        return {requirement.name for requirement in requirements
                if requirement.marker is None
                or requirement.marker.evaluate(dict(environment, extra=extra))}

    assert active('') == {'numpy', 'scipy'}
    assert active('video') - active('') == {'mediapipe', 'opencv-contrib-python'}
