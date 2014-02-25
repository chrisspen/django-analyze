# -*- coding: utf-8 -*-
from south.utils import datetime_utils as datetime
from south.db import db
from south.v2 import SchemaMigration
from django.db import models


class Migration(SchemaMigration):

    def forwards(self, orm):
        # Adding field 'GenotypeGene._value_genome'
        db.add_column(u'django_analyze_genotypegene', '_value_genome',
                      self.gf('django.db.models.fields.related.ForeignKey')(to=orm['django_analyze.Genome'], null=True, blank=True),
                      keep_default=False)


    def backwards(self, orm):
        # Deleting field 'GenotypeGene._value_genome'
        db.delete_column(u'django_analyze_genotypegene', '_value_genome_id')


    models = {
        'django_analyze.gene': {
            'Meta': {'ordering': "('-dependee_gene__id', 'name')", 'unique_together': "(('genome', 'name'),)", 'object_name': 'Gene'},
            'coverage_ratio': ('django.db.models.fields.FloatField', [], {'db_index': 'True', 'null': 'True', 'blank': 'True'}),
            'default': ('django.db.models.fields.CharField', [], {'max_length': '1000', 'null': 'True'}),
            'dependee_gene': ('django.db.models.fields.related.ForeignKey', [], {'blank': 'True', 'related_name': "'dependent_genes'", 'null': 'True', 'to': "orm['django_analyze.Gene']"}),
            'dependee_value': ('django.db.models.fields.CharField', [], {'max_length': '1000', 'null': 'True', 'blank': 'True'}),
            'description': ('django.db.models.fields.TextField', [], {'default': "''", 'blank': 'True'}),
            'exploration_priority': ('django.db.models.fields.PositiveIntegerField', [], {'default': '1', 'db_index': 'True'}),
            'genome': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'genes'", 'to': "orm['django_analyze.Genome']"}),
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'max_increment': ('django.db.models.fields.CharField', [], {'max_length': '100', 'null': 'True', 'blank': 'True'}),
            'max_value': ('django.db.models.fields.CharField', [], {'max_length': '100', 'null': 'True', 'blank': 'True'}),
            'max_value_observed': ('django.db.models.fields.CharField', [], {'max_length': '100', 'null': 'True', 'blank': 'True'}),
            'min_value': ('django.db.models.fields.CharField', [], {'max_length': '100', 'null': 'True', 'blank': 'True'}),
            'min_value_observed': ('django.db.models.fields.CharField', [], {'max_length': '100', 'null': 'True', 'blank': 'True'}),
            'mutation_weight': ('django.db.models.fields.FloatField', [], {'db_index': 'True', 'null': 'True', 'blank': 'True'}),
            'name': ('django.db.models.fields.CharField', [], {'max_length': '1000'}),
            'type': ('django.db.models.fields.CharField', [], {'max_length': '100'}),
            'values': ('django.db.models.fields.TextField', [], {'null': 'True', 'blank': 'True'})
        },
        'django_analyze.genedependency': {
            'Meta': {'unique_together': "(('gene', 'dependee_gene', 'dependee_value'),)", 'object_name': 'GeneDependency'},
            'dependee_gene': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'dependents'", 'to': "orm['django_analyze.Gene']"}),
            'dependee_value': ('django.db.models.fields.CharField', [], {'max_length': '1000'}),
            'gene': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'dependencies'", 'to': "orm['django_analyze.Gene']"}),
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'positive': ('django.db.models.fields.BooleanField', [], {'default': 'True'})
        },
        'django_analyze.genome': {
            'Meta': {'object_name': 'Genome'},
            'delete_inferiors': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'epoche': ('django.db.models.fields.PositiveIntegerField', [], {'default': '0'}),
            'epoche_stall': ('django.db.models.fields.PositiveIntegerField', [], {'default': '10'}),
            'epoches_since_improvement': ('django.db.models.fields.PositiveIntegerField', [], {'default': '0'}),
            'evaluating_part': ('django.db.models.fields.PositiveIntegerField', [], {'default': '0'}),
            'evaluation_timeout': ('django.db.models.fields.PositiveIntegerField', [], {'default': '300'}),
            'evaluator': ('django.db.models.fields.CharField', [], {'max_length': '1000', 'null': 'True', 'blank': 'True'}),
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'max_fitness': ('django.db.models.fields.FloatField', [], {'null': 'True', 'blank': 'True'}),
            'max_species': ('django.db.models.fields.PositiveIntegerField', [], {'default': '10'}),
            'maximum_evaluated_population': ('django.db.models.fields.PositiveIntegerField', [], {'default': '1000'}),
            'maximum_population': ('django.db.models.fields.PositiveIntegerField', [], {'default': '10'}),
            'min_fitness': ('django.db.models.fields.FloatField', [], {'null': 'True', 'blank': 'True'}),
            'mutation_rate': ('django.db.models.fields.FloatField', [], {'default': '0.1'}),
            'name': ('django.db.models.fields.CharField', [], {'unique': 'True', 'max_length': '100'}),
            'production_genotype': ('django.db.models.fields.related.ForeignKey', [], {'blank': 'True', 'related_name': "'production_genomes'", 'null': 'True', 'on_delete': 'models.SET_NULL', 'to': "orm['django_analyze.Genotype']"}),
            'production_genotype_auto': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'ratio_evaluated': ('django.db.models.fields.FloatField', [], {'null': 'True', 'blank': 'True'})
        },
        'django_analyze.genotype': {
            'Meta': {'ordering': "('-fitness',)", 'unique_together': "(('genome', 'fingerprint'),)", 'object_name': 'Genotype', 'index_together': "(('valid', 'fresh', 'fitness'), ('genome', 'fresh'))"},
            'created': ('django.db.models.fields.DateTimeField', [], {'auto_now_add': 'True', 'blank': 'True'}),
            'epoche_of_evaluation': ('django.db.models.fields.PositiveIntegerField', [], {'db_index': 'True', 'null': 'True', 'blank': 'True'}),
            'error': ('django.db.models.fields.TextField', [], {'null': 'True', 'blank': 'True'}),
            'evaluating': ('django.db.models.fields.BooleanField', [], {'default': 'False', 'db_index': 'True'}),
            'evaluating_pid': ('django.db.models.fields.IntegerField', [], {'null': 'True', 'blank': 'True'}),
            'fingerprint': ('django.db.models.fields.CharField', [], {'db_index': 'True', 'max_length': '700', 'null': 'True', 'db_column': "'fingerprint'", 'blank': 'True'}),
            'fingerprint_fresh': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'fitness': ('django.db.models.fields.FloatField', [], {'null': 'True', 'blank': 'True'}),
            'fitness_evaluation_datetime': ('django.db.models.fields.DateTimeField', [], {'null': 'True', 'blank': 'True'}),
            'fitness_evaluation_datetime_start': ('django.db.models.fields.DateTimeField', [], {'null': 'True', 'blank': 'True'}),
            'fresh': ('django.db.models.fields.BooleanField', [], {'default': 'False', 'db_index': 'True'}),
            'gene_count': ('django.db.models.fields.PositiveIntegerField', [], {'null': 'True', 'blank': 'True'}),
            'generation': ('django.db.models.fields.PositiveIntegerField', [], {'default': '1'}),
            'genome': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'genotypes'", 'to': "orm['django_analyze.Genome']"}),
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'mean_absolute_error': ('django.db.models.fields.FloatField', [], {'db_index': 'True', 'null': 'True', 'blank': 'True'}),
            'mean_evaluation_seconds': ('django.db.models.fields.PositiveIntegerField', [], {'null': 'True', 'blank': 'True'}),
            'ontime_parts': ('django.db.models.fields.PositiveIntegerField', [], {'null': 'True', 'blank': 'True'}),
            'ontime_ratio': ('django.db.models.fields.FloatField', [], {'db_index': 'True', 'null': 'True', 'blank': 'True'}),
            'species': ('django.db.models.fields.related.ForeignKey', [], {'blank': 'True', 'related_name': "'genotypes'", 'null': 'True', 'on_delete': 'models.SET_NULL', 'to': "orm['django_analyze.Species']"}),
            'success_parts': ('django.db.models.fields.PositiveIntegerField', [], {'null': 'True', 'blank': 'True'}),
            'success_ratio': ('django.db.models.fields.FloatField', [], {'db_index': 'True', 'null': 'True', 'blank': 'True'}),
            'total_evaluation_seconds': ('django.db.models.fields.PositiveIntegerField', [], {'null': 'True', 'blank': 'True'}),
            'total_parts': ('django.db.models.fields.PositiveIntegerField', [], {'null': 'True', 'blank': 'True'}),
            'valid': ('django.db.models.fields.BooleanField', [], {'default': 'True', 'db_index': 'True'})
        },
        'django_analyze.genotypegene': {
            'Meta': {'ordering': "('gene__name',)", 'unique_together': "(('genotype', 'gene'),)", 'object_name': 'GenotypeGene'},
            '_value': ('django.db.models.fields.CharField', [], {'max_length': '1000', 'db_column': "'value'"}),
            '_value_genome': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['django_analyze.Genome']", 'null': 'True', 'blank': 'True'}),
            'gene': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'gene_values'", 'to': "orm['django_analyze.Gene']"}),
            'genotype': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'genes'", 'to': "orm['django_analyze.Genotype']"}),
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'})
        },
        u'django_analyze.genotypegeneillegal': {
            'Meta': {'object_name': 'GenotypeGeneIllegal', 'managed': 'False'},
            'dependee_name': ('django.db.models.fields.CharField', [], {'max_length': '1000'}),
            'dependee_value': ('django.db.models.fields.CharField', [], {'max_length': '1000'}),
            'gene_value': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['django_analyze.GenotypeGene']", 'on_delete': 'models.DO_NOTHING', 'primary_key': 'True', 'db_column': "'illegal_genotypegene_id'"}),
            'genotype': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'illegal_gene_values'", 'on_delete': 'models.DO_NOTHING', 'db_column': "'illegal_genotype_id'", 'to': "orm['django_analyze.Genotype']"}),
            'illegal_gene_name': ('django.db.models.fields.CharField', [], {'max_length': '1000'}),
            'illegal_value': ('django.db.models.fields.CharField', [], {'max_length': '1000'})
        },
        u'django_analyze.genotypegenemissing': {
            'Meta': {'object_name': 'GenotypeGeneMissing', 'managed': 'False'},
            'default': ('django.db.models.fields.CharField', [], {'max_length': '1000'}),
            'dependee_gene': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'missing_dependents'", 'on_delete': 'models.DO_NOTHING', 'primary_key': 'True', 'db_column': "'dependee_gene_id'", 'to': "orm['django_analyze.Gene']"}),
            'gene': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['django_analyze.Gene']", 'on_delete': 'models.DO_NOTHING', 'primary_key': 'True', 'db_column': "'gene_id'"}),
            'gene_name': ('django.db.models.fields.CharField', [], {'max_length': '1000'}),
            'genotype': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'missing_gene_values'", 'on_delete': 'models.DO_NOTHING', 'db_column': "'genotype_id'", 'to': "orm['django_analyze.Genotype']"})
        },
        'django_analyze.species': {
            'Meta': {'ordering': "('genome', 'index')", 'unique_together': "(('genome', 'index'),)", 'object_name': 'Species', 'index_together': "(('genome', 'index'),)"},
            'centroid': ('picklefield.fields.PickledObjectField', [], {'null': 'True', 'blank': 'True'}),
            'genome': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'species'", 'to': "orm['django_analyze.Genome']"}),
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'index': ('django.db.models.fields.PositiveIntegerField', [], {'default': '0', 'db_index': 'True'}),
            'population': ('django.db.models.fields.PositiveIntegerField', [], {'default': '0'})
        }
    }

    complete_apps = ['django_analyze']