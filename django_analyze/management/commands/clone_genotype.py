import sys

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db.transaction import commit_on_success

from optparse import make_option

from django_analyze import models

class Command(BaseCommand):
    args = '<genotype_id>'
    help = 'Makes a copy of a genotype.'
    option_list = BaseCommand.option_list + (
        #make_option('--genomes', default=''),
    )

    @commit_on_success
    def handle(self, *args, **options):
        old_genotype_id = int(args[0])
        
        new_genotype = models.Genotype.objects.get(id=old_genotype_id)
        new_genotype.id = None
        new_genotype.fingerprint = None
        new_genotype.fingerprint_fresh = False
        new_genotype.save(check_fingerprint=False)
        
        old_genotype = models.Genotype.objects.get(id=old_genotype_id)
        genes = old_genotype.genes.all()
        total = genes.count()
        i = 0
        for gene in genes:
            i += 1
            print '\rCopying gene %i of %i.' % (i, total),
            gene.id = None
            gene.genotype = new_genotype
            gene.save()
        print
        print 'Created clone genotype %i.' % (new_genotype.id,)
        