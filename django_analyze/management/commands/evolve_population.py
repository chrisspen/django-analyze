import os
import sys
import warnings

from multiprocessing import Process

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from optparse import make_option

from django_analyze import models
from django_analyze import utils

safe_backends = ['django.core.cache.backends.locmem.LocMemCache']

class Command(BaseCommand):
    args = ''
    help = 'Manages the genetic evolution of one or more genomes.'
    option_list = BaseCommand.option_list + (
        make_option('--genotype', default=0),
        make_option('--genome', default=''),
        make_option('--populate', action='store_true', default=False),
        make_option('--population', default=0),
        make_option('--evaluate', action='store_true', default=False),
        make_option('--epoches', default=0),
        make_option('--force-reset', action='store_true', default=False),
        make_option('--no-cleanup', action='store_true', default=False),
        make_option('--no-clear', action='store_true', default=False),
        make_option('--continuous', action='store_true', default=False),
        make_option('--processes', default=-1, help='The number of processes to use for evaluating.'),
    )

    def handle(self, *args, **options):
        
        if os.path.isfile(utils.WAIT_FOR_STALE_ERROR_FN):
            os.remove(utils.WAIT_FOR_STALE_ERROR_FN)
        
        ids = [int(_) for _ in options['genome']]
        q = models.Genome.objects.all().only('id')
        if ids:
            q = q.filter(id__in=ids)
            
        genotype_id = int(options.get('genotype', 0))
        del options['genotype']
        if genotype_id:
            q = q.filter(genotypes__id=genotype_id)
            
        total = q.count()
        print '%i genomes found.' % total
        i = 0
        
        if options['processes'] and \
        settings.CACHES['default']['BACKEND'] not in safe_backends:
            warnings.warn('You are using multiprocessing but you are not '
                'using a multiprocessing safe cache backend. This may cause '
                'corruption.')
        
        for genome in list(q):
            i += 1
            self.evolve(genome.id, genotype_id=genotype_id, **options)
            
    def evolve(self, genome_id, genotype_id=None, **kwargs):
        populate = kwargs['populate']
        evaluate = kwargs['evaluate']
        force_reset = kwargs['force_reset']
        population = int(kwargs['population'])
        genome = models.Genome.objects.get(id=genome_id)
        print 'Evolving genome %s.' % (genome.name,)
        genome.evolve(
            genotype_id=genotype_id,
            populate=populate,
            population=population,
            evaluate=evaluate,
            force_reset=force_reset,
            cleanup=not kwargs['no_cleanup'],
            continuous=kwargs['continuous'],
            clear=not kwargs['no_clear'],
            epoches=int(kwargs['epoches']),
            processes=int(kwargs['processes']),
        )
        