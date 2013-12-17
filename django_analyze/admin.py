from django.contrib import admin

import models

import admin_steroids
from admin_steroids.utils import view_related_link
from admin_steroids.filters import NullListFilter

class PredictorAdmin(admin.ModelAdmin):
    
    list_display = (
        'id',
        'algorithm',
        'trained_datetime',
        'testing_mean_absolute_error_str',
        'training_seconds',
        'fresh',
    )
    list_filter = (
        'fresh',
    )
    readonly_fields = (
        'testing_mean_absolute_error_str',
    )
    
    def testing_mean_absolute_error_str(self, obj):
        if not obj or obj.testing_mean_absolute_error is None:
            return ''
        return '%.4f' % obj.testing_mean_absolute_error
    testing_mean_absolute_error_str.short_description = 'testing mean absolute error'
    testing_mean_absolute_error_str.admin_order_field = 'testing_mean_absolute_error'

class GeneInline(admin.TabularInline):
    model = models.Gene
    extra = 0
    #max_num = 1
    
    fields = (
        'name',
        'type',
        'values',
        'default',
        'min_value',
        'max_value',
    )
    
    readonly_fields = (
    )

class GenomeAdmin(admin.ModelAdmin):
    inlines = (
        GeneInline,
    )
    
    readonly_fields = (
        'genotypes_link',
        'max_fitness',
        'min_fitness',
    )
    
    def genotypes_link(self, obj=None):
        if not obj:
            return ''
        return view_related_link(obj, 'genotypes')
    genotypes_link.allow_tags = True
    genotypes_link.short_description = 'genotypes'

admin.site.register(models.Genome, GenomeAdmin)

class GenotypeGeneInline(admin.TabularInline):
    model = models.GenotypeGene
    extra = 0
    #max_num = 1
    
    fields = (
        'gene',
        '_value',
    )
    
    readonly_fields = (
        #'_value',
    )

class GenotypeAdmin(admin_steroids.BetterRawIdFieldsModelAdmin,):
    inlines = (
        GenotypeGeneInline,
    )
    
    list_display = (
        'id',
        'genome',
        'fitness',
        'fingerprint',
    )
    
    list_filter = (
        ('fitness', NullListFilter),
    )
    
    raw_id_fields = (
        'genome',
    )
    
    readonly_fields = (
        #'genotypes_link',
    )
    
    def genotypes_link(self, obj=None):
        if not obj:
            return ''
        return view_related_link(obj, 'genotypes')

admin.site.register(models.Genotype, GenotypeAdmin)
