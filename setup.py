import sys
import os

from setuptools import setup, find_packages, Command

import django_analyze

def get_reqs():
    return [
        'Django>=1.4.0',
        #'django_materialized_views',
        
        'django-materialized-views>=0.2.0',
        'django-admin-steroids>=0.1.13',
        #'django-chroniker',
        
        'psutil>=2.1.1',
        'six>=1.5.2',
        
        'numpy', # required by scikit-learn
        'scipy', # required by scikit-learn, takes a long time to build
        'scikit-learn',
        'django-picklefield',
        
        'joblib>=0.8.1',
    ]

package_lookup = {
    'django-materialized-views>=0.2.0': 'https://github.com/chrisspen/django-materialized-views/archive/django-materialized-views-0.2.0.tar.gz#egg=django-materialized-views-0.2.0',
    'django-admin-steroids>=0.1.13': 'https://github.com/chrisspen/django-admin-steroids/archive/django-admin-steroids-0.1.13.tar.gz#egg=django-admin-steroids-0.1.13',
}

class TestCommand(Command):
    description = "Runs unittests."
    user_options = [
        ('name=', None,
         'Name of the specific test to run.'),
        ('package=', None,
         'Name of the specific package to install.'),
        ('virtual-env-dir=', None,
         'The location of the virtual environment to use.'),
        ('upgrade=', '1',
         'Upgrade package in virtual environment.'),
        ('virtonly=', '0',
         'If specified, only modifies the test virtual environment and does not run any tests.'),
        ('forcevirt=', '0',
         'If specified, forcibly reinstalls packages in the virtual environment.'),
    ]
    def initialize_options(self):
        self.name = None
        self.package = None
        self.virtual_env_dir = './.env'
        self.upgrade = 1
        self.virtonly = 0
        self.forcevirt = 0
    def finalize_options(self):
        self.package = (self.package or '').strip()
        self.upgrade = int(self.upgrade)
        self.virtonly = int(self.virtonly)
        self.forcevirt = int(self.forcevirt)
    def run(self):
        args = dict(
            virtual_env_dir=self.virtual_env_dir,
            upgrade_str = '-U' if self.upgrade else '',
        )
        if self.forcevirt or not os.path.isdir(args['virtual_env_dir']):
            print 'Virtual environment not found. Initializing.'
            os.system('virtualenv --no-site-packages %(virtual_env_dir)s' % args)
            for package in get_reqs():
                if self.package and not package.startswith(self.package):
                    continue
                if package in package_lookup:
                    package = package_lookup[package]
                args['package'] = package
                cmd = '. %(virtual_env_dir)s/bin/activate; pip install %(upgrade_str)s %(package)s; deactivate' % args
                print cmd
                os.system(cmd)
        if self.virtonly:
            return
                
        if self.name:
            cmd = '. ./.env/bin/activate; django-admin.py test --pythonpath=. --traceback --settings=django_analyze.tests.settings django_analyze.tests.tests.Tests.%s; deactivate' % self.name
        else:
            #cmd = '. ./.env/bin/activate; django-admin.py test --pythonpath=. --settings=django_analyze.tests.settings tests; deactivate'
            cmd = '. ./.env/bin/activate; django-admin.py test --pythonpath=. --traceback --settings=django_analyze.tests.settings; deactivate'
        print cmd
        os.system(cmd)

setup(
    name = "django-analyze",
    version = django_analyze.__version__,
    packages = find_packages(),
    author = "Chris Spencer",
    author_email = "chrisspen@gmail.com",
    description = "",
    license = "LGPL",
    url = "https://github.com/chrisspen/django-analyze",
    #https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers = [
        'Development Status :: 3 - Alpha',
        'Environment :: Web Environment',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Framework :: Django',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    install_requires = get_reqs(),
    dependency_links = package_lookup.values(),
    cmdclass={
        'test': TestCommand,
    },
)
