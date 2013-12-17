 
from setuptools import setup, find_packages, Command

import os
import urllib

import django_analyze

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
        os.system('. ./.env/bin/activate; django-admin.py test --pythonpath=. --settings=django_analyze.tests.settings tests; deactivate')

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
    install_requires = ["Django>=1.4.0",],
#    dependency_links = [
#    ],
#    cmdclass={
#        'test': TestCommand,
#    },
)
