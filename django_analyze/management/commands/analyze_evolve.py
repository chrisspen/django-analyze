from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from optparse import make_option

from django_analyze import models

class Command(BaseCommand):
    args = '<genome ids>'
    help = ''
    option_list = BaseCommand.option_list + (
        #make_option('--user', default=1),
        #make_option('--subject', default='test subject'),
        #make_option('--recipient_list', default=None),
    )

    def handle(self, *args, **options):
        ids = [int(_) for _ in args]
        q = models.Genome.objects.filter(id__in=ids)
        total = q.count()
        i = 0
        for genome in q.iterator():
            i += 1
            print 'Evolving %s (%i of %i)...' % (genome.name, i, total)
            genome.evolve()
            