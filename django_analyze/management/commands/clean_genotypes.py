import sys

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from optparse import make_option

from django_analyze import models

class Command(BaseCommand):
    args = '<genome ids>'
    help = 'Deletes and adds gene values from genotypes based on gene rules.'
    option_list = BaseCommand.option_list + (
        make_option('--genotype', default=0),
    )

    def handle(self, *args, **options):
        ids = [int(_) for _ in args]
        q = models.Genome.objects.all()
        if ids:
            q = q.filter(id__in=ids)
        genotype_id = int(options.get('genotype', 0))
        del options['genotype']
        if genotype_id:
            q = q.filter(genotypes__id=genotype_id)
            
        total = q.count()
        i = 0
        for genome in q.iterator():
            i += 1
            genome.delete_corrupt(save=False)
            genome.add_missing_genes(save=False)
        
        print
        models.Genotype.freshen_fingerprints()
        