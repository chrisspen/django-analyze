import re
import sys

import django
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db.models import Q
from django.core import serializers

from optparse import make_option

from django_analyze import models

class Command(BaseCommand):
    args = '<genome id>'
    help = 'Efficiently exports the given genome to JSON.'
    option_list = BaseCommand.option_list + (
#        make_option('--force', action='store_true', default=False),
        make_option(
            '--exclude-genotypes',
            action='store_true',
            default=False),
        make_option(
            '--genotypes',
            default=''),
    )

    def handle(self, genome_id, **options):
        genome_id = int(genome_id)
        genotype_ids = [
            int(_)
            for _ in options['genotypes'].split(',')
            if _.strip().isdigit()]
        queries = []
        
        def to_json_str(obj):
            json_str = serializers.serialize('json', [obj],
#                use_natural_foreign_keys=True,
#                use_natural_primary_keys=True,
                use_natural_keys=True,
                indent=4,
            )
            json_str = re.sub(
                '"pk":\s+[0-9]+', '"pk": null', json_str, re.DOTALL|re.I)
            json_str = json_str.strip()[1:-1].strip()
            return json_str
        
        # Genomes.
        queries.append(models.Genome.objects.filter(id=genome_id))
        
        # Epoches.
        queries.append(models.Epoche.objects.filter(genome__id=genome_id))
        
        # Genes.
        queries.append(models.Gene.objects.filter(genome__id=genome_id))
        
        # Gene dependencies.
        queries.append(models.GeneDependency.objects.filter(gene__genome__id=genome_id))
        
        # Species.
        queries.append(models.Species.objects.filter(genome__id=genome_id))
        
        if not options['exclude_genotypes']:
            
            # Genotypes.
            genotype_q = models.Genotype.objects.filter(
                genome__id=genome_id)
            if genotype_ids:
                genotype_q = genotype_q.filter(id__in=genotype_ids)
            queries.append(genotype_q)
        
            # Genotype gene values.
            queries.append(models.GenotypeGene.objects.filter(
                genotype__in=genotype_q))
        
        total = sum([_q.count() for _q in queries])
        i = 0
        print '[\n'
        for q in queries:
            for obj in q.iterator():
                i += 1
                if i == 1 or not i % 10 or i == total:
                    print>>sys.stderr, '\r%i of %i (%.02f%%)' \
                        % (i, total, float(i)/total*100),
                    sys.stderr.flush()
                
                # In certain cases where there's a cyclic reference on a model
                # we're not including, we must temporarily break the reference.
                if isinstance(obj, models.Genome):
                    obj.production_genotype = None
                    obj._epoche = None
                    
                first = i == 1
                last = i == total
                json_str = to_json_str(obj)
                if not last:
                    json_str = json_str+','
                print json_str
        print ']'
        print>>sys.stderr, '\r%i of %i' % (i, total),
        print>>sys.stderr, '\nDone!'
        sys.stderr.flush()
        