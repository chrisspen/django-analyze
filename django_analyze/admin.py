from django.http import HttpResponse, HttpResponseRedirect
from django.contrib import admin

import models

import admin_steroids
from admin_steroids.utils import view_related_link, classproperty
from admin_steroids.filters import NullListFilter

class PredictorAdmin(admin.ModelAdmin):
    
    list_display = (
        'id',
        'algorithm',
        'trained_datetime',
        'testing_mean_absolute_error_str',
        'training_seconds',
        'training_ontime',
        'fresh',
    )
    list_filter = (
        'fresh',
        'training_ontime',
        ('training_seconds', NullListFilter),
        ('testing_mean_absolute_error', NullListFilter),
    )
    
    readonly_fields = (
        'testing_mean_absolute_error_str',
    )
    
    actions = (
        'refresh',
    )
    
    def refresh(self, request, queryset):
        for obj in queryset:
            obj.fresh = False
            obj.save()
        return HttpResponseRedirect(request.META['HTTP_REFERER'])
    refresh.short_description = 'Refresh selected %(verbose_name_plural)s'
    
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

_callbacks = {}

def register_callback(modeladmin, callback):
    _callbacks.setdefault(modeladmin, [])
    _callbacks[modeladmin].append(callback)

class GenotypeAdmin(admin_steroids.BetterRawIdFieldsModelAdmin):
    
    modeladmin_callbacks = set()
    
    inlines = (
        GenotypeGeneInline,
    )
    
    list_display = (
        'id',
        'genome',
        'fitness',
        'mean_absolute_error',
        'mean_evaluation_seconds',
        'fresh',
        #'fingerprint',
    )
    
    list_filter = (
        'fresh',
        ('fitness', NullListFilter),
    )
    
    search_fields = (
        'genes___value',
    )
    
    raw_id_fields = (
        'genome',
    )
    
    readonly_fields = [
        'id',
        'fitness',
        'fitness_evaluation_datetime',
        'mean_evaluation_seconds',
        'mean_absolute_error',
        'gene_count',
    ]
    
    actions = (
        'refresh'
    )
    
    def refresh(self, request, queryset):
        for obj in queryset:
            obj.fresh = False
            obj.save()
        return HttpResponseRedirect(request.META['HTTP_REFERER'])
    refresh.short_description = 'Refresh selected %(verbose_name_plural)s'
    
    def __init__(self, *args, **kwargs):
        super(GenotypeAdmin, self).__init__(*args, **kwargs)
    
    def get_fieldsets(self, request, obj=None):
        fieldsets = [
            (None, {
                'fields': [
                    'id',
                    'genome',
                    'fresh',
                    'fitness',
                    'fitness_evaluation_datetime',
                    'mean_evaluation_seconds',
                    'mean_absolute_error',
                    'gene_count',
                ]
            }),
        ]
        for method in models._modeladmin_extenders.itervalues():
            method(self, request, obj, fieldsets=fieldsets)
        return fieldsets
    
    def genotypes_link(self, obj=None):
        if not obj:
            return ''
        return view_related_link(obj, 'genotypes')

admin.site.register(models.Genotype, GenotypeAdmin)
