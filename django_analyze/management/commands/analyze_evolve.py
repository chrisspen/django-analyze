from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from optparse import make_option

from django_analyze import models

class Command(BaseCommand):
    args = '<genome ids>'
    help = ''
    option_list = BaseCommand.option_list + (
        make_option('--genotype_id', default=0),
        make_option('--no-populate', action='store_true', default=False),
        make_option('--no-evaluate', action='store_true', default=False),
        #make_option('--production', action='store_true', default=False),
    )

    def handle(self, *args, **options):
        ids = [int(_) for _ in args]
        q = models.Genome.objects.filter(id__in=ids)
        genotype_id = int(options.get('genotype_id', 0))
        no_populate = options['no_populate']
        no_evaluate = options['no_evaluate']
        if genotype_id:
            q = q.filter(genotypes__id=genotype_id)
        total = q.count()
        i = 0
        for genome in q.iterator():
            i += 1
            print 'Evolving genome %s (%i of %i)...' % (genome.name, i, total)
            genome.evolve(
                genotype_id=genotype_id,
                populate=not no_populate,
                evaluate=not no_evaluate)
            