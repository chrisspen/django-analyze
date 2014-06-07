import re
import sys
import time
from datetime import datetime, timedelta
import commands
from pprint import pprint

import django
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db.models import Q
from django.core import serializers
from django.db.models.loading import get_model
from django.db import transaction, connection
from django.utils import simplejson

from optparse import make_option

from django_analyze.models import Genome

class Command(BaseCommand):
    args = '<genome_id>'
    help = ''
    option_list = BaseCommand.option_list + (
#        make_option('--force', action='store_true', default=False),
#        make_option('--delete-existing', action='store_true', default=False),
    )

    def handle(self, genome_id, **options):
        g = Genome.objects.get(id=int(genome_id))
        