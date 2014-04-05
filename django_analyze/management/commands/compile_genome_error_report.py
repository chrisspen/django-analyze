import sys

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from optparse import make_option

from django_analyze import models

class Command(BaseCommand):
    args = '<genome ids>'
    help = 'Regenerates the error report for the given genomes.'
    option_list = BaseCommand.option_list + (
#        make_option('--genotype-id', default=0),
#        make_option('--no-populate', action='store_true', default=False),
#        make_option('--no-evaluate', action='store_true', default=False),
        #make_option('--production', action='store_true', default=False),
    )

    def handle(self, *args, **options):
        ids = [int(_) for _ in args]
        q = models.Genome.objects.filter(id__in=ids)
        total = q.count()
        i = 0
        for genome in q.iterator():
            i += 1
            print '\rProcessing genome %s (%i of %i)...' % (genome.name, i, total),
            sys.stdout.flush()
            genome.generate_error_report(force=True)
        print
        print 'Done.'
        