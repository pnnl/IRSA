from setuptools import find_packages, setup

with open('irsa/__init__.py') as f:
    exec([x for x in f.readlines() if '__version__' in x][0])

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

with open('requirements.txt') as f:
    requirements = f.read()

pkgs = find_packages(exclude=('examples', 'docs', 'tests'))

setup(
    name='irsa',
    version=__version__,
    description='Infrared Spectra Annotation',
    long_description=readme,
    author='Sean M. Colby',
    author_email='sean.colby@pnnl.gov',
    url='https://github.com/pnnl/irsa',
    install_requires=requirements,
    python_requires='>=3.7',
    license=license,
    packages=pkgs,
)
