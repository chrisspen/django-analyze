import sys

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from optparse import make_option

from django_analyze import models

class Command(BaseCommand):
    args = '<genome ids>'
    help = 'Manages the genetic evolution of one or more genomes.'
    option_list = BaseCommand.option_list + (
        make_option('--genotype-id', default=0),
        make_option('--no-populate', action='store_true', default=False),
        make_option('--no-evaluate', action='store_true', default=False),
        #make_option('--production', action='store_true', default=False),
    )

    def handle(self, *args, **options):
        ids = [int(_) for _ in args]
        q = models.Genome.objects.filter(id__in=ids)
        genotype_id = int(options.get('genotype_id', 0))
        del options['genotype_id']
        if genotype_id:
            q = q.filter(genotypes__id=genotype_id)
        total = q.count()
        i = 0
        #TODO:multiprocessing, warning this may corrupt django's cache
        for genome in q.iterator():
            i += 1
            self.evolve(genome.id, genotype_id=genotype_id, **options)
            
    def evolve(self, genome_id, genotype_id=None, **kwargs):
        no_populate = kwargs['no_populate']
        no_evaluate = kwargs['no_evaluate']
        
        genome = models.Genome.objects.get(id=genome_id)
        print 'Evolving genome %s.' % (genome.name,)
#            genes = genome.genes.all()
#            genes_total = genes.count()
#            genes_i = 0
#            for gene in genes.iterator():
#                genes_i += 1
#                print '\rgenes %i of %i' % (genes_i, genes_total),
#                sys.stdout.flush()
#                gene.update_coverage()
#            return
        genome.evolve(
            genotype_id=genotype_id,
            populate=not no_populate,
            evaluate=not no_evaluate,
        )