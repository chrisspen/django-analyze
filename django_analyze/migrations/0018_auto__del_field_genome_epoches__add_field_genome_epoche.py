# -*- coding: utf-8 -*-
from south.utils import datetime_utils as datetime
from south.db import db
from south.v2 import SchemaMigration
from django.db import models


class Migration(SchemaMigration):

    def forwards(self, orm):
        # Deleting field 'Genome.epoches'
        db.delete_column(u'django_analyze_genome', 'epoches')

        # Adding field 'Genome.epoche'
        db.add_column(u'django_analyze_genome', 'epoche',
                      self.gf('django.db.models.fields.PositiveIntegerField')(default=0),
                      keep_default=False)


    def backwards(self, orm):
        # Adding field 'Genome.epoches'
        db.add_column(u'django_analyze_genome', 'epoches',
                      self.gf('django.db.models.fields.PositiveIntegerField')(default=0),
                      keep_default=False)

        # Deleting field 'Genome.epoche'
        db.delete_column(u'django_analyze_genome', 'epoche')


    models = {
        'django_analyze.gene': {
            'Meta': {'ordering': "('-dependee_gene__id', 'name')", 'unique_together': "(('genome', 'name'),)", 'object_name': 'Gene'},
            'default': ('django.db.models.fields.CharField', [], {'max_length': '1000', 'null': 'True', 'blank': 'True'}),
            'dependee_gene': ('django.db.models.fields.related.ForeignKey', [], {'blank': 'True', 'related_name': "'dependent_genes'", 'null': 'True', 'to': "orm['django_analyze.Gene']"}),
            'dependee_value': ('django.db.models.fields.CharField', [], {'max_length': '1000', 'null': 'True', 'blank': 'True'}),
            'genome': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'genes'", 'to': "orm['django_analyze.Genome']"}),
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'max_increment': ('django.db.models.fields.CharField', [], {'max_length': '100', 'null': 'True', 'blank': 'True'}),
            'max_value': ('django.db.models.fields.CharField', [], {'max_length': '100', 'null': 'True', 'blank': 'True'}),
            'min_value': ('django.db.models.fields.CharField', [], {'max_length': '100', 'null': 'True', 'blank': 'True'}),
            'name': ('django.db.models.fields.CharField', [], {'max_length': '100'}),
            'type': ('django.db.models.fields.CharField', [], {'max_length': '100'}),
            'values': ('django.db.models.fields.TextField', [], {'null': 'True', 'blank': 'True'})
        },
        'django_analyze.genome': {
            'Meta': {'object_name': 'Genome'},
            'delete_inferiors': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'epoche': ('django.db.models.fields.PositiveIntegerField', [], {'default': '0'}),
            'evaluation_timeout': ('django.db.models.fields.PositiveIntegerField', [], {'default': '0'}),
            'evaluator': ('django.db.models.fields.CharField', [], {'max_length': '1000', 'null': 'True', 'blank': 'True'}),
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'max_fitness': ('django.db.models.fields.FloatField', [], {'null': 'True', 'blank': 'True'}),
            'maximum_population': ('django.db.models.fields.PositiveIntegerField', [], {'default': '1000'}),
            'min_fitness': ('django.db.models.fields.FloatField', [], {'null': 'True', 'blank': 'True'}),
            'mutation_rate': ('django.db.models.fields.FloatField', [], {'default': '0.1'}),
            'name': ('django.db.models.fields.CharField', [], {'unique': 'True', 'max_length': '100'}),
            'production_genotype': ('django.db.models.fields.related.ForeignKey', [], {'blank': 'True', 'related_name': "'production_genomes'", 'null': 'True', 'on_delete': 'models.SET_NULL', 'to': "orm['django_analyze.Genotype']"}),
            'production_genotype_auto': ('django.db.models.fields.BooleanField', [], {'default': 'False'})
        },
        'django_analyze.genotype': {
            'Meta': {'ordering': "('-fitness',)", 'unique_together': "(('genome', 'fingerprint'),)", 'object_name': 'Genotype'},
            'created': ('django.db.models.fields.DateTimeField', [], {'auto_now_add': 'True', 'blank': 'True'}),
            'fingerprint': ('django.db.models.fields.CharField', [], {'db_index': 'True', 'max_length': '700', 'null': 'True', 'db_column': "'fingerprint'", 'blank': 'True'}),
            'fingerprint_fresh': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'fitness': ('django.db.models.fields.FloatField', [], {'null': 'True', 'blank': 'True'}),
            'fitness_evaluation_datetime': ('django.db.models.fields.DateTimeField', [], {'null': 'True', 'blank': 'True'}),
            'fresh': ('django.db.models.fields.BooleanField', [], {'default': 'False', 'db_index': 'True'}),
            'fresh_datetime': ('django.db.models.fields.DateTimeField', [], {'null': 'True', 'blank': 'True'}),
            'gene_count': ('django.db.models.fields.PositiveIntegerField', [], {'null': 'True', 'blank': 'True'}),
            'generation': ('django.db.models.fields.PositiveIntegerField', [], {'default': '1'}),
            'genome': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'genotypes'", 'to': "orm['django_analyze.Genome']"}),
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'mean_absolute_error': ('django.db.models.fields.FloatField', [], {'db_index': 'True', 'null': 'True', 'blank': 'True'}),
            'mean_evaluation_seconds': ('django.db.models.fields.PositiveIntegerField', [], {'null': 'True', 'blank': 'True'})
        },
        'django_analyze.genotypegene': {
            'Meta': {'ordering': "('gene__name',)", 'unique_together': "(('genotype', 'gene'),)", 'object_name': 'GenotypeGene'},
            '_value': ('django.db.models.fields.CharField', [], {'max_length': '1000', 'db_column': "'value'"}),
            'gene': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'genes'", 'to': "orm['django_analyze.Gene']"}),
            'genotype': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'genes'", 'to': "orm['django_analyze.Genotype']"}),
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'})
        }
    }

    complete_apps = ['django_analyze']