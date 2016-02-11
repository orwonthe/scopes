__author__="willy"
__date__ ="$Aug 31, 2014 1:20:15 PM$"

from setuptools import setup,find_packages

setup (
  name = 'ScopeAnalysis',
  version = '0.1',
  packages = find_packages(),

  # Declare your packages' dependencies here, for eg:
  install_requires=['foo>=3'],

  # Fill in these to make your Egg ready for upload to
  # PyPI
  author = 'willy',
  author_email = '',

  # TODO: fix commented out for now because
  # /usr/lib/python2.7/distutils/dist.py:267: UserWarning: Unknown distribution option: 'summary'
  # summary = 'Just another Python package for the cheese shop',
  url = '',
  license = '',
  long_description= 'Long description of the package',

  # could also include long_description, download_url, classifiers, etc.

  
)