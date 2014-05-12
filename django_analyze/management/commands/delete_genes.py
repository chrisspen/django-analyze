import sys

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from optparse import make_option

from django_analyze import models

class Command(BaseCommand):
    args = '<genome>'
    help = 'Bulk deletes genes matching a given pattern.'
    option_list = BaseCommand.option_list + (
        make_option('--gene__name__icontains', default=''),
        #make_option('--genomes', default=''),
    )

    def handle(self, *args, **options):
        
        gene__name__icontains = options['gene__name__icontains'].strip()
        assert gene__name__icontains
        
        q = models.Genome.objects.only('id').filter(id__in=map(int, args))
        total = q.count()
        i = 0
        for genome in q.iterator():
            i += 1
            q2 = genome.genes.filter(name__icontains=gene__name__icontains)
            total2 = q2.count()
            i2 = 0
            for gene in q2.iterator():
                i2 += 1
                print('Deleting gene %s (%i of %i %.02f%%).' % (gene.name, i2, total2, i2/float(total2)*100))
                gene.delete()
                