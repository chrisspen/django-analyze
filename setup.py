 
from setuptools import setup, find_packages, Command

import os
import urllib

import django_analyze

def get_reqs(reqs=["Django>=1.4.0", "python-dateutil", "dtree"]):
    # optparse is included with Python <= 2.7, but has been deprecated in favor
    # of argparse.  We try to import argparse and if we can't, then we'll add
    # it to the requirements
    try:
        import argparse
    except ImportError:
        reqs.append("argparse>=1.1")
    return reqs

class TestCommand(Command):
    description = "Runs unittests."
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        args = dict(
            virtual_env_dir = './.env',
        )
        if not os.path.isdir(args['virtual_env_dir']):
            os.system('virtualenv %(virtual_env_dir)s' % args)
            for package in get_reqs():
                args['package'] = package
                cmd = '. %(virtual_env_dir)s/bin/activate; pip install -U %(package)s; deactivate' % args
                print cmd
                os.system(cmd)
        os.system('. ./.env/bin/activate; django-admin.py test --pythonpath=. --settings=sense.tests.settings tests; deactivate')

setup(
    name = "django-analyze",
    version = django_analyze.__version__,
    packages = find_packages(),
    author = "Chris Spencer",
    author_email = "chrisspen@gmail.com",
    description = "",
    license = "LGPL",
    url = "https://github.com/chrisspen/django-analyze",
    classifiers = [
        'Environment :: Web Environment',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: LGPL License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Framework :: Django',
    ],
    install_requires = get_reqs(),
#    dependency_links = [
#    ],
#    cmdclass={
#        'test': TestCommand,
#    },
)
