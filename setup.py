import sys
import os

from setuptools import setup, find_packages, Command

import django_analyze

try:
    from pypandoc import convert
    read_md = lambda f: convert(f, 'rst')
except ImportError:
    print("Warning: pypandoc module not found, could not convert "
        "Markdown to RST")
    def read_md(f):
        try:
            return open(f, 'r').read()
        except IOError:
            return ''

def get_reqs():
    return [
        'Django>=1.4.0',
        #'django_materialized_views',
        
        'django-materialized-views>=0.3.1',
        'django-admin-steroids>=0.1.13',
        #'django-chroniker',
        
        'psutil>=2.1.1',
        'six>=1.5.2',
        
#        'numpy', # required by scikit-learn
#        'scipy', # required by scikit-learn, takes a long time to build
#        'scikit-learn',
        'django-picklefield',
        
        'joblib>=0.8.1',
    ]

#package_lookup = {
#    'django-materialized-views>=0.2.0': 'https://github.com/chrisspen/django-materialized-views/archive/django-materialized-views-0.2.0.tar.gz#egg=django-materialized-views-0.2.0',
#    'django-admin-steroids>=0.1.13': 'https://github.com/chrisspen/django-admin-steroids/archive/django-admin-steroids-0.1.13.tar.gz#egg=django-admin-steroids-0.1.13',
#}

class TestCommand(Command):
    description = "Runs unittests."
    user_options = [
        ('name=', None,
         'Name of the specific test to run.'),
        ('virtual-env-dir=', None,
         'The location of the virtual environment to use.'),
        ('pv=', None,
         'The version of Python to use. e.g. 2.7 or 3'),
    ]
    
    def initialize_options(self):
        self.name = None
        self.virtual_env_dir = './.env%s'
        self.pv = 0
        self.versions = [2.7]#, 3]
        
    def finalize_options(self):
        pass
    
    def build_virtualenv(self, pv):
        virtual_env_dir = self.virtual_env_dir % pv
        kwargs = dict(virtual_env_dir=virtual_env_dir, pv=pv)
        if not os.path.isdir(virtual_env_dir):
            cmd = 'virtualenv -p /usr/bin/python{pv} {virtual_env_dir}'.format(**kwargs)
            #print(cmd)
            os.system(cmd)
            
            cmd = '{virtual_env_dir}/bin/easy_install -U distribute'.format(**kwargs)
            os.system(cmd)
            
            for package in get_reqs():
                kwargs['package'] = package
                cmd = '{virtual_env_dir}/bin/pip install -U {package}'.format(**kwargs)
                #print(cmd)
                os.system(cmd)
    
    def run(self):
        versions = self.versions
        if self.pv:
            versions = [self.pv]
        
        for pv in versions:
            
            self.build_virtualenv(pv)
            kwargs = dict(
                pv=pv,
                name=self.name,
                virtual_env_dir=self.virtual_env_dir % pv)
                
            if self.name:
                cmd = '{virtual_env_dir}/bin/django-admin.py test --pythonpath=. --traceback --settings=django_analyze.tests.settings django_analyze.tests.tests.Tests.{name}'.format(**kwargs)
            else:
                cmd = '{virtual_env_dir}/bin/django-admin.py test --pythonpath=. --traceback --settings=django_analyze.tests.settings'.format(**kwargs)
                
            print(cmd)
            ret = os.system(cmd)
            if ret:
                return
                
#        if self.name:
#            cmd = '. ./.env/bin/activate; django-admin.py test --pythonpath=. --traceback --settings=django_analyze.tests.settings django_analyze.tests.tests.Tests.%s; deactivate' % self.name
#        else:
#            #cmd = '. ./.env/bin/activate; django-admin.py test --pythonpath=. --settings=django_analyze.tests.settings tests; deactivate'
#            cmd = '. ./.env/bin/activate; django-admin.py test --pythonpath=. --traceback --settings=django_analyze.tests.settings; deactivate'
#        print cmd
#        os.system(cmd)

try:
    long_description = read_md('README.md')
except IOError:
    long_description = ''

setup(
    name = "django-analyze",
    version = django_analyze.__version__,
    packages = find_packages(),
    author = "Chris Spencer",
    author_email = "chrisspen@gmail.com",
    description = "A general purpose framework for training and testing classification algorithms.",
    long_description = long_description,
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
    #dependency_links = package_lookup.values(),
    cmdclass={
        'test': TestCommand,
    },
)
