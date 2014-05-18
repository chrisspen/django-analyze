import sys
import warnings

from multiprocessing import Process

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from optparse import make_option

from django_analyze import models

safe_backends = ['django.core.cache.backends.locmem.LocMemCache']

class Command(BaseCommand):
    args = ''
    help = 'Prepares genomes for production use.'
    option_list = BaseCommand.option_list + (
        make_option('--processes', default=1, help='The number of processes to use for evaluating.'),
        make_option('--genomes', default='')
    )

    def handle(self, *args, **options):
        ids = [int(_) for _ in options['genomes'].split(',')]
        q = models.Genome.objects.all().only('id')
        if ids:
            q = q.filter(id__in=ids)
        total = q.count()
        print '%i genomes found.' % total
        i = 0
        
        if options['processes'] and \
        settings.CACHES['default']['BACKEND'] not in safe_backends:
            warnings.warn('You are using multiprocessing but you are not '
                'using a multiprocessing safe cache backend. This may cause '
                'corruption.')
        
        def cmp_genomes(a, b):
            b_dependents = [_.id for _ in b.get_dependent_genomes()]
            a_dependents = [_.id for _ in a.get_dependent_genomes()]
            if a.id in b_dependents:
                return -1
            elif b.id in a_dependents:
                return +1
            return 0
        
        for genome in sorted(q, cmp=cmp_genomes):
            i += 1
            print genome
            genome.production_evaluate()
            
            