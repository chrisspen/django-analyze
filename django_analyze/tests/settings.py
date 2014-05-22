import os, sys
PROJECT_DIR = os.path.dirname(__file__)
DATABASES = {
    'default':{
        'ENGINE': 'django.db.backends.sqlite3',
        # Don't do this. It dramatically slows down the test.
#        'NAME': '/tmp/test.db',
#        'TEST_NAME': '/tmp/test.db',
        'NAME': ':memory:',
    }
}
#ROOT_URLCONF = 'django_analyze.tests.urls'
INSTALLED_APPS = [
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.sites',
    'django_analyze',
    'django_analyze.tests',
]
MEDIA_ROOT = os.path.join(PROJECT_DIR, 'media')
SOUTH_TESTS_MIGRATE = False
USE_TZ = True

AUTH_USER_MODEL = 'auth.User'

SECRET_KEY = '-a+nku-@gozg9_%er2_o+fabjf-knwaenjwfs@rt!z^ox=7$2d'