# -*- coding: utf-8 -*-
from south.utils import datetime_utils as datetime
from south.db import db
from south.v2 import SchemaMigration
from django.db import models


class Migration(SchemaMigration):

    def forwards(self, orm):
        # Adding model 'Genome'
        db.create_table(u'django_analyze_genome', (
            (u'id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('name', self.gf('django.db.models.fields.CharField')(unique=True, max_length=100)),
            ('maximum_population', self.gf('django.db.models.fields.PositiveIntegerField')(default=1000)),
            ('delete_inferiors', self.gf('django.db.models.fields.BooleanField')(default=False)),
            ('mutation_rate', self.gf('django.db.models.fields.FloatField')(default=0.1)),
            ('epoches', self.gf('django.db.models.fields.PositiveIntegerField')(default=0)),
        ))
        db.send_create_signal(u'django_analyze', ['Genome'])

        # Adding model 'Gene'
        db.create_table(u'django_analyze_gene', (
            (u'id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('genome', self.gf('django.db.models.fields.related.ForeignKey')(related_name='genes', to=orm['django_analyze.Genome'])),
            ('name', self.gf('django.db.models.fields.CharField')(max_length=100)),
            ('type', self.gf('django.db.models.fields.CharField')(max_length=100)),
            ('values', self.gf('django.db.models.fields.TextField')(null=True, blank=True)),
            ('default', self.gf('django.db.models.fields.CharField')(max_length=1000)),
            ('min_value', self.gf('django.db.models.fields.CharField')(max_length=100, null=True, blank=True)),
            ('max_value', self.gf('django.db.models.fields.CharField')(max_length=100, null=True, blank=True)),
        ))
        db.send_create_signal(u'django_analyze', ['Gene'])

        # Adding unique constraint on 'Gene', fields ['genome', 'name']
        db.create_unique(u'django_analyze_gene', ['genome_id', 'name'])

        # Adding model 'Genotype'
        db.create_table(u'django_analyze_genotype', (
            (u'id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('genome', self.gf('django.db.models.fields.related.ForeignKey')(related_name='genotypes', to=orm['django_analyze.Genome'])),
            ('fingerprint', self.gf('django.db.models.fields.CharField')(db_index=True, max_length=700, null=True, db_column='fingerprint', blank=True)),
            ('fingerprint_fresh', self.gf('django.db.models.fields.BooleanField')(default=False)),
            ('created', self.gf('django.db.models.fields.DateTimeField')(auto_now_add=True, blank=True)),
            ('fitness', self.gf('django.db.models.fields.FloatField')(null=True, blank=True)),
            ('fitness_evaluation_datetime', self.gf('django.db.models.fields.DateTimeField')(null=True, blank=True)),
        ))
        db.send_create_signal(u'django_analyze', ['Genotype'])

        # Adding model 'GenotypeGene'
        db.create_table(u'django_analyze_genotypegene', (
            (u'id', self.gf('django.db.models.fields.AutoField')(primary_key=True)),
            ('genotype', self.gf('django.db.models.fields.related.ForeignKey')(related_name='genes', to=orm['django_analyze.Genotype'])),
            ('gene', self.gf('django.db.models.fields.related.ForeignKey')(related_name='genes', to=orm['django_analyze.Gene'])),
            ('_value', self.gf('django.db.models.fields.CharField')(max_length=1000, db_column='value')),
        ))
        db.send_create_signal(u'django_analyze', ['GenotypeGene'])

        # Adding unique constraint on 'GenotypeGene', fields ['genotype', 'gene']
        db.create_unique(u'django_analyze_genotypegene', ['genotype_id', 'gene_id'])


    def backwards(self, orm):
        # Removing unique constraint on 'GenotypeGene', fields ['genotype', 'gene']
        db.delete_unique(u'django_analyze_genotypegene', ['genotype_id', 'gene_id'])

        # Removing unique constraint on 'Gene', fields ['genome', 'name']
        db.delete_unique(u'django_analyze_gene', ['genome_id', 'name'])

        # Deleting model 'Genome'
        db.delete_table(u'django_analyze_genome')

        # Deleting model 'Gene'
        db.delete_table(u'django_analyze_gene')

        # Deleting model 'Genotype'
        db.delete_table(u'django_analyze_genotype')

        # Deleting model 'GenotypeGene'
        db.delete_table(u'django_analyze_genotypegene')


    models = {
        u'django_analyze.gene': {
            'Meta': {'unique_together': "(('genome', 'name'),)", 'object_name': 'Gene'},
            'default': ('django.db.models.fields.CharField', [], {'max_length': '1000'}),
            'genome': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'genes'", 'to': u"orm['django_analyze.Genome']"}),
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'max_value': ('django.db.models.fields.CharField', [], {'max_length': '100', 'null': 'True', 'blank': 'True'}),
            'min_value': ('django.db.models.fields.CharField', [], {'max_length': '100', 'null': 'True', 'blank': 'True'}),
            'name': ('django.db.models.fields.CharField', [], {'max_length': '100'}),
            'type': ('django.db.models.fields.CharField', [], {'max_length': '100'}),
            'values': ('django.db.models.fields.TextField', [], {'null': 'True', 'blank': 'True'})
        },
        u'django_analyze.genome': {
            'Meta': {'object_name': 'Genome'},
            'delete_inferiors': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'epoches': ('django.db.models.fields.PositiveIntegerField', [], {'default': '0'}),
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'}),
            'maximum_population': ('django.db.models.fields.PositiveIntegerField', [], {'default': '1000'}),
            'mutation_rate': ('django.db.models.fields.FloatField', [], {'default': '0.1'}),
            'name': ('django.db.models.fields.CharField', [], {'unique': 'True', 'max_length': '100'})
        },
        u'django_analyze.genotype': {
            'Meta': {'unique_together': '()', 'object_name': 'Genotype'},
            'created': ('django.db.models.fields.DateTimeField', [], {'auto_now_add': 'True', 'blank': 'True'}),
            'fingerprint': ('django.db.models.fields.CharField', [], {'db_index': 'True', 'max_length': '700', 'null': 'True', 'db_column': "'fingerprint'", 'blank': 'True'}),
            'fingerprint_fresh': ('django.db.models.fields.BooleanField', [], {'default': 'False'}),
            'fitness': ('django.db.models.fields.FloatField', [], {'null': 'True', 'blank': 'True'}),
            'fitness_evaluation_datetime': ('django.db.models.fields.DateTimeField', [], {'null': 'True', 'blank': 'True'}),
            'genome': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'genotypes'", 'to': u"orm['django_analyze.Genome']"}),
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'})
        },
        u'django_analyze.genotypegene': {
            'Meta': {'ordering': "('gene__name',)", 'unique_together': "(('genotype', 'gene'),)", 'object_name': 'GenotypeGene'},
            '_value': ('django.db.models.fields.CharField', [], {'max_length': '1000', 'db_column': "'value'"}),
            'gene': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'genes'", 'to': u"orm['django_analyze.Gene']"}),
            'genotype': ('django.db.models.fields.related.ForeignKey', [], {'related_name': "'genes'", 'to': u"orm['django_analyze.Genotype']"}),
            u'id': ('django.db.models.fields.AutoField', [], {'primary_key': 'True'})
        }
    }

    complete_apps = ['django_analyze']