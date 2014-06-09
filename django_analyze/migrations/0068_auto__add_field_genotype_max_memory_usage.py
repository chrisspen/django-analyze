# -*- coding: utf-8 -*-
from south.utils import datetime_utils as datetime
from south.db import db
from south.v2 import SchemaMigration
from django.db import models


class Migration(SchemaMigration):

    def forwards(self, orm):
        # Adding field 'Genotype.max_memory_usage'
        db.add_column(u'django_analyze_genotype', 'max_memory_usage',
                      self.gf('django.db.models.fields.PositiveIntegerField')(null=True, blank=True),
                      keep_default=False)


    def backwards(self, orm):
        # Deleting field 'Genotype.max_memory_usage'
        db.delete_column(u'django_analyze_genotype', 'max_memory_usage')


    models = {
        'django_analyze.epoche': {
            'Meta': {'ordering': "('genome', '-index')", 'unique_together': "(('genome', 'index'),)", 'object_name': 'Epoche'},
            'genome': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'epoches'", 'to': "orm['django_analyze.Genome']"}),
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'index': ('django.db.models.fields.PositiveIntegerField', [], {'default': '1', 'db_index': 'True'}),
            'max_fitness': ('django.db.models.fields.FloatField', [], {'db_index': 'True', 'null': 'True', 'blank': 'True'}),
            'mean_fitness': ('django.db.models.fields.FloatField', [], {'db_index': 'True', 'null': 'True', 'blank': 'True'}),
            'min_fitness': ('django.db.models.fields.FloatField', [], {'db_index': 'True', 'null': 'True', 'blank': 'True'}),
            'oldest_epoche_of_creation': ('django.db.models.fields.PositiveIntegerField', [], {'null': 'True', 'blank': 'True'})
        },
        'django_analyze.gene': {
            'Meta': {'ordering': "('name',)", 'unique_together': "(('genome', 'name'),)", 'object_name': 'Gene'},
            'coverage_ratio': ('django.db.models.fields.FloatField', [], {'db_index': 'True', 'null': 'True', 'blank': 'True'}),
            'default': ('django.db.models.fields.CharField', [], {'max_length': '1000', 'null': 'True'}),
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
        'django_analyze.genestatistics': {
            'Meta': {'ordering': "('genome', 'gene', '-mean_fitness')", 'object_name': 'GeneStatistics', 'managed': 'False'},
            'gene': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['django_analyze.Gene']", 'on_delete': 'models.DO_NOTHING', 'db_column': "'gene_id'"}),
            'genome': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'gene_statistics'", 'on_delete': 'models.DO_NOTHING', 'db_column': "'genome_id'", 'to': "orm['django_analyze.Genome']"}),
            'genotype_count': ('django.db.models.fields.PositiveIntegerField', [], {}),
            'id': ('django.db.models.fields.CharField', [], {'max_length': '1000', 'primary_key': 'True'}),
            'max_fitness': ('django.db.models.fields.FloatField', [], {'null': 'True', 'blank': 'True'}),
            'mean_fitness': ('django.db.models.fields.FloatField', [], {'null': 'True', 'blank': 'True'}),
            'min_fitness': ('django.db.models.fields.FloatField', [], {'null': 'True', 'blank': 'True'}),
            'value': ('django.db.models.fields.CharField', [], {'max_length': '1000'})
        },
        'django_analyze.genome': {
            'Meta': {'object_name': 'Genome'},
            '_epoche': ('django.db.models.fields.related.ForeignKey', [], {'blank': 'True', 'related_name': "'current_genome'", 'null': 'True', 'on_delete': 'models.SET_NULL', 'to': "orm['django_analyze.Epoche']"}),
            'delete_inferiors': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'elite_ratio': ('django.db.models.fields.FloatField', [], {'default': '0.1'}),
            'epoche': ('django.db.models.fields.PositiveIntegerField', [], {'default': '1'}),
            'epoche_stall': ('django.db.models.fields.PositiveIntegerField', [], {'default': '10'}),
            'epoches_since_improvement': ('django.db.models.fields.PositiveIntegerField', [], {'default': '0'}),
            'error_report': ('django.db.models.fields.TextField', [], {'null': 'True', 'blank': 'True'}),
            'evaluating_part': ('django.db.models.fields.PositiveIntegerField', [], {'default': '0'}),
            'evaluation_timeout': ('django.db.models.fields.PositiveIntegerField', [], {'default': '300'}),
            'evaluator': ('django.db.models.fields.CharField', [], {'max_length': '1000', 'null': 'True', 'blank': 'True'}),
            'evolution_start_datetime': ('django.db.models.fields.DateTimeField', [], {'null': 'True', 'blank': 'True'}),
            'evolving': ('django.db.models.fields.BooleanField', [], {'default': 'False', 'db_index': 'True'}),
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'max_fitness': ('django.db.models.fields.FloatField', [], {'null': 'True', 'blank': 'True'}),
            'max_species': ('django.db.models.fields.PositiveIntegerField', [], {'default': '10'}),
            'maximum_evaluated_population': ('django.db.models.fields.PositiveIntegerField', [], {'default': '1000'}),
            'maximum_population': ('django.db.models.fields.PositiveIntegerField', [], {'default': '10'}),
            'min_fitness': ('django.db.models.fields.FloatField', [], {'null': 'True', 'blank': 'True'}),
            'mutation_rate': ('django.db.models.fields.FloatField', [], {'default': '0.1'}),
            'name': ('django.db.models.fields.CharField', [], {'unique': 'True', 'max_length': '100'}),
            'production_at_best': ('django.db.models.fields.BooleanField', [], {'default': 'False', 'db_index': 'True'}),
            'production_evaluation_timeout': ('django.db.models.fields.PositiveIntegerField', [], {'default': '0', 'null': 'True', 'blank': 'True'}),
            'production_genotype': ('django.db.models.fields.related.ForeignKey', [], {'blank': 'True', 'related_name': "'production_genomes'", 'null': 'True', 'on_delete': 'models.SET_NULL', 'to': "orm['django_analyze.Genotype']"}),
            'production_genotype_auto': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'ratio_evaluated': ('django.db.models.fields.FloatField', [], {'null': 'True', 'blank': 'True'}),
            'version': ('django.db.models.fields.PositiveIntegerField', [], {'default': '1'})
        },
        'django_analyze.genotype': {
            'Meta': {'ordering': "('-fitness',)", 'unique_together': "(('genome', 'fingerprint'),)", 'object_name': 'Genotype', 'index_together': "(('valid', 'fresh', 'fitness'), ('genome', 'fresh'))"},
            'accuracy': ('django.db.models.fields.FloatField', [], {'db_index': 'True', 'null': 'True', 'blank': 'True'}),
            'complete_parts': ('django.db.models.fields.PositiveIntegerField', [], {'null': 'True', 'blank': 'True'}),
            'complete_ratio': ('django.db.models.fields.FloatField', [], {'db_index': 'True', 'null': 'True', 'blank': 'True'}),
            'created': ('django.db.models.fields.DateTimeField', [], {'auto_now_add': 'True', 'blank': 'True'}),
            'description': ('django.db.models.fields.CharField', [], {'max_length': '500', 'null': 'True', 'blank': 'True'}),
            'epoche': ('django.db.models.fields.related.ForeignKey', [], {'blank': 'True', 'related_name': "'genotypes'", 'null': 'True', 'on_delete': 'models.SET_NULL', 'to': "orm['django_analyze.Epoche']"}),
            'epoche_of_creation': ('django.db.models.fields.PositiveIntegerField', [], {'db_index': 'True', 'null': 'True', 'blank': 'True'}),
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
            'immortal': ('django.db.models.fields.BooleanField', [], {'default': 'False', 'db_index': 'True'}),
            'max_memory_usage': ('django.db.models.fields.PositiveIntegerField', [], {'null': 'True', 'blank': 'True'}),
            'mean_absolute_error': ('django.db.models.fields.FloatField', [], {'db_index': 'True', 'null': 'True', 'blank': 'True'}),
            'mean_evaluation_seconds': ('django.db.models.fields.FloatField', [], {'null': 'True', 'blank': 'True'}),
            'mean_memory_usage': ('django.db.models.fields.PositiveIntegerField', [], {'null': 'True', 'blank': 'True'}),
            'memory_usage_samples': ('picklefield.fields.PickledObjectField', [], {'null': 'True', 'blank': 'True'}),
            'ontime_parts': ('django.db.models.fields.PositiveIntegerField', [], {'null': 'True', 'blank': 'True'}),
            'ontime_ratio': ('django.db.models.fields.FloatField', [], {'db_index': 'True', 'null': 'True', 'blank': 'True'}),
            'production_complete_parts': ('django.db.models.fields.PositiveIntegerField', [], {'null': 'True', 'blank': 'True'}),
            'production_complete_ratio': ('django.db.models.fields.FloatField', [], {'db_index': 'True', 'null': 'True', 'blank': 'True'}),
            'production_error': ('django.db.models.fields.TextField', [], {'null': 'True', 'blank': 'True'}),
            'production_evaluating': ('django.db.models.fields.BooleanField', [], {'default': 'False', 'db_index': 'True'}),
            'production_evaluating_pid': ('django.db.models.fields.IntegerField', [], {'null': 'True', 'blank': 'True'}),
            'production_evaluation_end_datetime': ('django.db.models.fields.DateTimeField', [], {'db_index': 'True', 'null': 'True', 'blank': 'True'}),
            'production_evaluation_start_datetime': ('django.db.models.fields.DateTimeField', [], {'db_index': 'True', 'null': 'True', 'blank': 'True'}),
            'production_fresh': ('django.db.models.fields.BooleanField', [], {'default': 'False', 'db_index': 'True'}),
            'production_ontime_parts': ('django.db.models.fields.PositiveIntegerField', [], {'null': 'True', 'blank': 'True'}),
            'production_ontime_ratio': ('django.db.models.fields.FloatField', [], {'db_index': 'True', 'null': 'True', 'blank': 'True'}),
            'production_success_parts': ('django.db.models.fields.PositiveIntegerField', [], {'null': 'True', 'blank': 'True'}),
            'production_success_ratio': ('django.db.models.fields.FloatField', [], {'db_index': 'True', 'null': 'True', 'blank': 'True'}),
            'production_total_parts': ('django.db.models.fields.PositiveIntegerField', [], {'null': 'True', 'blank': 'True'}),
            'production_valid': ('django.db.models.fields.BooleanField', [], {'default': 'True', 'db_index': 'True'}),
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
            '_value_genome': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['django_analyze.Genome']", 'null': 'True', 'on_delete': 'models.SET_NULL', 'blank': 'True'}),
            'gene': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'gene_values'", 'to': "orm['django_analyze.Gene']"}),
            'genotype': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'genes'", 'to': "orm['django_analyze.Genotype']"}),
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'})
        },
        u'django_analyze.genotypegeneillegal': {
            'Meta': {'object_name': 'GenotypeGeneIllegal', 'managed': 'False'},
            'gene_value': ('django.db.models.fields.related.ForeignKey', [], {'to': "orm['django_analyze.GenotypeGene']", 'on_delete': 'models.DO_NOTHING', 'primary_key': 'True', 'db_column': "'illegal_genotypegene_id'"}),
            'genotype': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'illegal_gene_values'", 'on_delete': 'models.DO_NOTHING', 'db_column': "'illegal_genotype_id'", 'to': "orm['django_analyze.Genotype']"}),
            'illegal_gene_name': ('django.db.models.fields.CharField', [], {'max_length': '1000'})
        },
        u'django_analyze.genotypegenemissing': {
            'Meta': {'object_name': 'GenotypeGeneMissing', 'managed': 'False'},
            'default': ('django.db.models.fields.CharField', [], {'max_length': '1000'}),
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