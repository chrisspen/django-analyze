from django.http import HttpResponse, HttpResponseRedirect
from django.contrib import admin, messages
from django.core.exceptions import ValidationError

import models

import admin_steroids
from admin_steroids.utils import view_related_link, classproperty
from admin_steroids.filters import NullListFilter

class BaseModelAdmin(admin_steroids.BetterRawIdFieldsModelAdmin):
    pass

class PredictorAdmin(BaseModelAdmin):
    
    list_display = [
        'id',
        'algorithm',
        'trained_datetime',
        'testing_mean_absolute_error_str',
        'training_seconds',
        'training_ontime',
        'fresh',
    ]
    list_filter = (
        'fresh',
        'training_ontime',
        ('training_seconds', NullListFilter),
        ('testing_mean_absolute_error', NullListFilter),
    )
    
    readonly_fields = [
        'testing_mean_absolute_error_str',
    ]
    
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

class GeneInline(
    #admin.TabularInline
    admin_steroids.BetterRawIdFieldsTabularInline):
    
    model = models.Gene
    extra = 0
    #max_num = 1
    
    fields = (
        'name',
        'type',
#        'dependee_gene',
#        'dependee_value',
        'values',
        'default',
        'min_value',
        'max_value',
        'max_increment',
    )
    
    readonly_fields = (
    )
    
    raw_id_fields = (
        'dependee_gene',
    )

class GeneAdmin(BaseModelAdmin):
    
    list_display = (
        'name',
        'genome',
        'type',
        'values_str',
        'default',
        'dependee_gene',
        'dependee_value',
        'coverage_ratio',
        'exploration_priority',
        'mutation_weight',
    )
    
    raw_id_fields = (
        'genome',
        'dependee_gene',
    )
    
    search_fields = (
        'name',
    )
    
    list_filter = (
        'type',
    )
    
    readonly_fields = (
        'values_str',
    )
    
    def values_str(self, obj=None):
        if not obj:
            return
        return (obj.values or '').replace(',', ', ')
    values_str.short_description = 'values'
    values_str.admin_order_field = 'values'
    
admin.site.register(models.Gene, GeneAdmin)

class SpeciesAdmin(admin.ModelAdmin):
    list_display = (
        'letter',
        'genome',
        'population',
    )
    readonly_fields = (
        'letter',
        'genome',
        'population',
    )
    
admin.site.register(models.Species, SpeciesAdmin)

class GenomeAdmin(admin.ModelAdmin):
    inlines = (
        #GeneInline,
    )
    
    list_display = (
        'id',
        'name',
        'min_fitness',
        'max_fitness',
    )
    
    readonly_fields = (
        'total_possible_genotypes',
        'genotypes_link',
        'genes_link',
        'species_link',
        'max_fitness',
        'min_fitness',
    )
    
    actions = (
        'organize_species',
    )
    
    def organize_species(self, request, queryset):
        i = 0
        for obj in queryset.iterator():
            i += 1
            obj.organize_species()
        messages.success(request, 'Organized species for %i genomes.' % i)
        return HttpResponseRedirect(request.META['HTTP_REFERER'])
    organize_species.short_description = 'Organize species for %(verbose_name_plural)s'
    
    def species_link(self, obj=None):
        if not obj:
            return ''
        return view_related_link(obj, 'species')
    species_link.allow_tags = True
    species_link.short_description = 'species'
    
    def genotypes_link(self, obj=None):
        if not obj:
            return ''
        return view_related_link(obj, 'genotypes')
    genotypes_link.allow_tags = True
    genotypes_link.short_description = 'genotypes'
    
    def genes_link(self, obj=None):
        try:
            if not obj:
                return ''
            return view_related_link(obj, 'genes')
        except Exception, e:
            return str(e)
    genes_link.allow_tags = True
    genes_link.short_description = 'genes'

admin.site.register(models.Genome, GenomeAdmin)

class GenotypeGeneAdmin(BaseModelAdmin):
    
    list_display = (
        'gene',
        'genotype',
        '_value',
    )
    
    list_editable = (
        '_value',
    )
    
    search_fields = (
        'gene__name',
        '_value',
    )

admin.site.register(models.GenotypeGene, GenotypeGeneAdmin)

class GenotypeGeneInline(admin.TabularInline):
    model = models.GenotypeGene
    extra = 0
    max_num = 0
    
    has_delete = can_delete = 1
    
    fields = (
        'gene',
        '_value',
        'is_legal',
    )
    
    exclude = (
#        'dependee_gene',
#        'dependee_value',
    )
    
    readonly_fields = (
        'gene',
        #'_value',
        'is_legal',
    )
    
    #def gene_name

_callbacks = {}

def register_callback(modeladmin, callback):
    _callbacks.setdefault(modeladmin, [])
    _callbacks[modeladmin].append(callback)

class GenotypeAdmin(admin_steroids.BetterRawIdFieldsModelAdmin):
    
#    modeladmin_callbacks = set()
    
    inlines = (
#        GenotypeGeneInline,
    )
    
    list_display = (
        'id',
        'genome',
        'species',
        'fitness',
        'mean_absolute_error',
        'mean_evaluation_seconds',
        'total_evaluation_seconds',
        'success_ratio',
        'ontime_ratio',
        'generation',
        'fresh',
        'valid',
        'fingerprint_bool',
    )
    
    list_filter = (
        'fresh',
        'valid',
        ('fitness', NullListFilter),
        ('fingerprint', NullListFilter),
    )
    
    search_fields = (
        'genes___value',
    )
    
    raw_id_fields = (
        'genome',
        'species',
    )
    
    readonly_fields = [
        'id',
        'fitness',
        'species',
        'fitness_evaluation_datetime',
        'mean_evaluation_seconds',
        'mean_absolute_error',
        'total_evaluation_seconds',
        'gene_count',
        'fingerprint_bool',
        'fingerprint',
        'fresh',
        'valid',
        'total_parts',
        'success_parts',
        'ontime_parts',
        'success_ratio',
        'ontime_ratio',
        'error',
#        'fresh_str',
        'fingerprint_fresh',
        'genes_link',
    ]
    
#    exclude = (
#        'fresh',
#    )
    
    actions = (
        'refresh',
        'check_fingerprint',
        'reset',
        'refresh_fitness',
    )
    
    def refresh(self, request, queryset):
        i = 0
        for obj in queryset.iterator():
            i += 1
            obj.fresh = False
            obj.save()
        messages.success(request, '%i genotypes were refreshed.' % i)
        return HttpResponseRedirect(request.META['HTTP_REFERER'])
    refresh.short_description = 'Refresh selected %(verbose_name_plural)s'
    
    def reset(self, request, queryset):
        i = 0
        for obj in queryset.iterator():
            i += 1
            obj.reset()
        messages.success(request, '%i genotypes were reset.' % i)
        return HttpResponseRedirect(request.META['HTTP_REFERER'])
    reset.short_description = 'Reset selected %(verbose_name_plural)s'
    
    def refresh_fitness(self, request, queryset):
        i = 0
        for obj in queryset.iterator():
            i += 1
            obj.refresh_fitness()
        messages.success(request, '%i genotypes had their fitness refreshed.' % i)
        return HttpResponseRedirect(request.META['HTTP_REFERER'])
    refresh_fitness.short_description = 'Refresh fitness of selected %(verbose_name_plural)s'
    
    def check_fingerprint(self, request, queryset):
        """
        Tests to see if it can regenerate a fresh fingerprint.
        If it encounters a validation error, implying there's another genotype
        with the same fingerprint, it clears the fingerprint signalling the
        genotype for deletion.
        Checks unevaluated genotypes first, so if there's a duplicate, the
        previously evaluated genotype is left alone.
        """
        errors = 0
        for obj in queryset.order_by('-fitness'):
            obj.fingerprint_fresh = False
            try:
                obj.save()
            except ValidationError:
                errors += 1
                obj.fingerprint = None
                obj.save(check_fingerprint=False)
        if errors:
            messages.warning(request, 'Found %i potentially duplicate genotypes.' % errors)
        else:
            messages.success(request, 'All fingerprints look good.')
                
        return HttpResponseRedirect(request.META['HTTP_REFERER'])
    check_fingerprint.short_description = 'Check selected %(verbose_name_plural)s for fingerprint conflicts'
    
#    def __init__(self, *args, **kwargs):
#        super(GenotypeAdmin, self).__init__(*args, **kwargs)
    
    def get_fieldsets(self, request, obj=None):
        fieldsets = [
            (None, {
                'fields': [
                    'id',
                    'genome',
                    'species',
                    'fingerprint_bool',
                    'fresh',
                    'valid',
                    'error',
#                    'fresh_str',
                    'fingerprint_fresh',
                    'fitness',
                    'fitness_evaluation_datetime',
                    'mean_evaluation_seconds',
                    'total_evaluation_seconds',
                    'mean_absolute_error',
                    'gene_count',
                    'total_parts',
                    'success_parts',
                    'ontime_parts',
                    'success_ratio',
                    'ontime_ratio',
                    'fingerprint',
                    'genes_link',
                ]
            }),
        ]
        for method in models._modeladmin_extenders.itervalues():
            method(self, request, obj, fieldsets=fieldsets)
        return fieldsets
    
    def fresh_str(self, obj=None):
        if not obj:
            return ''
        return obj.fresh
    fresh_str.short_description = 'fresh'
    fresh_str.boolean = True
    
    def fingerprint_bool(self, obj=None):
        if not obj:
            return ''
        return bool(obj.fingerprint)
    fingerprint_bool.short_description = 'has fingerprint'
    fingerprint_bool.boolean = True
    
    def genotypes_link(self, obj=None):
        if not obj:
            return ''
        return view_related_link(obj, 'genotypes')
    
    def genes_link(self, obj=None):
        if not obj:
            return ''
        return view_related_link(obj, 'genes')
    genes_link.short_description = 'gene values'
    genes_link.allow_tags = True

admin.site.register(models.Genotype, GenotypeAdmin)

class LabelAdmin(admin_steroids.BetterRawIdFieldsModelAdmin):

    list_display = (
        'name',
        'duplicate_of',
    )
    
    search_fields = (
        'name',
    )
    
    list_filter = (
        ('duplicate_of', NullListFilter),
    )
    