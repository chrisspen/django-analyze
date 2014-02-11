from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from optparse import make_option

from django_analyze import models

class Command(BaseCommand):
    args = '<genome ids>'
    help = 'Manages the genetic evolution of one or more genomes.'
    option_list = BaseCommand.option_list + (
        make_option('--all', action='store_true', default=False),
    )

    def handle(self, genome_id, **options):
        genome = models.Genome.objects.get(id=int(genome_id))
        genome.organize_species(**options)
        