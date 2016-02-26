try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'Crop Yield Loss Data Processing Toolbox',
    'url': 'https://github.com/nicococo/CropYieldLoss',
    'author': 'Nico Goernitz',
    'author_email': 'nico.goernitz@tu-berlin.de',
    'version': '0.1',
    'install_requires': ['nose', 'cvxopt','scikit-learn','pandas'],
    'packages': ['CropYieldLoss'],
    'scripts': [],
    'name': 'CropYieldLoss',
    'classifiers':['Intended Audience :: Science/Research',
                   'Programming Language :: Python',
                   'Topic :: Scientific/Engineering',
                   'Operating System :: POSIX',
                   'Operating System :: Unix',
                   'Operating System :: MacOS',
                   'Programming Language :: Python :: 2',
                   'Programming Language :: Python :: 2.7']
}

setup(**config)