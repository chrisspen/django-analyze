import sys

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.db.transaction import commit_manually

from optparse import make_option

from django_analyze import models

class Command(BaseCommand):
    args = ''
    help = 'Deletes and adds gene values from genotypes based on gene rules.'
    option_list = BaseCommand.option_list + (
        make_option('--genotypes', default=''),
        make_option('--genomes', default=''),
        make_option('--dryrun', action='store_true', default=False),
    )

    @commit_manually
    def handle(self, **options):
        try:
            dryrun = options['dryrun']
            
            q = models.Genome.objects.only('id').all()
            
            genome_ids = [int(_) for _ in options['genomes'].split(',') if _.isdigit()]
            if genome_ids:
                q = q.filter(id__in=genome_ids)
            
            genotype_ids = [int(_) for _ in options['genotypes'].split(',') if _.isdigit()]
            if genotype_ids:
                q = q.filter(genotypes__id__in=genotype_ids)
            
            q = q.distinct()
            total = q.count()
            i = 0
            for genome in q.iterator():
                i += 1
                genome.cleanup(genotype_ids=genotype_ids)
            
            if dryrun:
                transaction.rollback()
            else:
                transaction.commit()
                
        except Exception, e:
            transaction.rollback()
            raise
            