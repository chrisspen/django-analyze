import sys
import traceback

from django.http import HttpResponse, HttpResponseRedirect
from django.contrib import admin, messages
from django.core.exceptions import ValidationError

import constants as c
import models
import utils

import admin_steroids
import admin_steroids.options
from admin_steroids.utils import (
    view_related_link, view_link, classproperty,
    get_admin_changelist_url,
)
from admin_steroids.filters import NullListFilter

class BaseModelAdmin(admin_steroids.options.BetterRawIdFieldsModelAdmin):
    pass

class PredictorAdmin(BaseModelAdmin):
    
    list_display = [
        'id',
        #'algorithm',
        'trained_datetime',
        'training_mean_absolute_error_str',
        'training_accuracy',
        'training_seconds',
        'training_ontime',
        'testing_mean_absolute_error_str',
        'testing_accuracy',
        'predicted_value',
        'predicted_prob',
        'reference_difference',
        'testing_r2',
        'predicted_score',
        'evaluating',
        'test',
        'fresh',
        'valid',
    ]
    list_filter = (
        'evaluating',
        'test',
        'fresh',
        'valid',
        'training_ontime',
        ('training_seconds', NullListFilter),
        ('testing_mean_absolute_error', NullListFilter),
    )
    
    readonly_fields = [
        'test',
        'fresh',
        'valid',
        'predicted_value',
        'expected_value',
        'reference_value',
        'reference_difference',
        'predicted_prob',
        'testing_r2',
        'predicted_score',
        'training_accuracy',
        'testing_accuracy',
        'training_mean_absolute_error_str',
        'testing_mean_absolute_error_str',
    ]
    
    actions = (
        'refresh',
        'clear',
    )
    
    def refresh(self, request, queryset):
        queryset.update(fresh=False)
        return HttpResponseRedirect(request.META['HTTP_REFERER'])
    refresh.short_description = 'Mark selected %(verbose_name_plural)s as stale'
    
    def clear(self, request, queryset):
        queryset.update(
            training_mean_squared_error = None,
            training_mean_absolute_error = None,
            testing_mean_absolute_error = None,
            testing_mean_squared_error = None,
            predicted_score = None,
            predicted_prob = None,
            reference_difference = None,
            reference_value = None,
            expected_value = None,
            predicted_value = None,
            fresh = False,
        )
        return HttpResponseRedirect(request.META['HTTP_REFERER'])
    clear.short_description = 'Clear metrics on selected %(verbose_name_plural)s'
    
    def testing_mean_absolute_error_str(self, obj):
        if not obj or obj.testing_mean_absolute_error is None:
            return ''
        return '%.4f' % obj.testing_mean_absolute_error
    testing_mean_absolute_error_str.short_description = 'testing mean absolute error'
    testing_mean_absolute_error_str.admin_order_field = 'testing_mean_absolute_error'
    
    def training_mean_absolute_error_str(self, obj):
        if not obj or obj.training_mean_absolute_error is None:
            return ''
        return '%.4f' % obj.training_mean_absolute_error
    training_mean_absolute_error_str.short_description = 'training mean absolute error'
    training_mean_absolute_error_str.admin_order_field = 'training_mean_absolute_error'

class GeneInline(
    #admin.TabularInline
    admin_steroids.options.BetterRawIdFieldsTabularInline):
    
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
#        'dependee_gene',
    )

class GeneDependencyInline(
    #admin.TabularInline
    admin_steroids.options.BetterRawIdFieldsTabularInline):
    
    model = models.GeneDependency
    extra = 1
    fk_name = 'gene'
    
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
#        'dependee_gene',
#        'dependee_value',
        'coverage_ratio',
        'exploration_priority',
        'mutation_weight',
    )
    
    raw_id_fields = (
        'genome',
#        'dependee_gene',
    )
    
    search_fields = (
        'name',
        'values',
    )
    
    list_filter = (
        'type',
#        ('dependee_gene', NullListFilter),
        ('dependencies', NullListFilter),
        'genome',
    )
    
    readonly_fields = (
        'values_str',
    )
    
    inlines = (
        GeneDependencyInline,
    )
    
    def values_str(self, obj=None):
        if not obj:
            return
        #return (obj.values or '').replace(',', ', ')
        _lst = obj.get_values_list() or []
        lst = []
        for _ in _lst:
            if isinstance(_, models.Genotype):
                lst.append('%i:%i' % (_.genome.id, _.id))
            else:
                lst.append(str(_))
        return ', '.join(lst)
    values_str.short_description = 'values'
    values_str.admin_order_field = 'values'
    
admin.site.register(models.Gene, GeneAdmin)

class SpeciesAdmin(BaseModelAdmin):
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

class GenomeAdmin(BaseModelAdmin):
    inlines = (
        #GeneInline,
    )
    
    list_display = (
        'id',
        'name',
        'epoche',
        'epoches_since_improvement',
        
        'total_genotype_count',
        'pending_genotype_count',
        'evaluating_genotype_count',
        'complete_genotype_count',
        'invalid_genotype_count',
        
        'min_fitness',
        'max_fitness',
        'improving',
        'evolving',
        #'evolution_start_datetime',
        
        'production_at_best',
        'is_production_ready_bool',
    )
    
    list_display_links = (
        'id',
        'name',
    )
    
    list_filter = (
        'evolving',
        'production_at_best',
        #'improving',
    )
    
    search_fields = (
        'name',
    )
    
    raw_id_fields = (
        'production_genotype',
    )
    
    readonly_fields = (
        'genes_link',
        'species_link',
        'epoches_link',
        'max_fitness',
        'min_fitness',
        'version',
        #'epoches_since_improvement',
        'improving',
        'total_possible_genotypes_sci',
        'genotypes_link',
        'genestats_link',
        'evolution_start_datetime',
        'production_at_best',
        'is_production_ready_bool',
        'error_report_str',
        
        'total_genotype_count',
        'pending_genotype_count',
        'evaluating_genotype_count',
        'complete_genotype_count',
        'invalid_genotype_count',
        
    )
    
    actions = (
        'organize_species',
    )
    
    fieldsets = (
        (None, {
            'fields': (
                'name',
                'evaluator',
                'evolving',
                'evolution_start_datetime',
                'total_possible_genotypes_sci',
                'version',
            )
        }),
        ('Related models', {
            'fields': (
                'genes_link',
                'genestats_link',
                'species_link',
                'genotypes_link',
                'epoches_link',
            )
        }),
        ('Progress', {
            'fields': (
                'improving',
                'epoche',
                'epoches_since_improvement',
                'min_fitness',
                'max_fitness',
                'evaluating_part',
                'ratio_evaluated',
                'error_report_str',
            )
        }),
        ('Production', {
            'fields': (
                'production_at_best',
                'is_production_ready_bool',
                'production_genotype_auto',
                'production_genotype',
                'production_evaluation_timeout',
            )
        }),
        ('Options', {
            'fields': (
                'maximum_population',
                'maximum_evaluated_population',
                'mutation_rate',
                'evaluation_timeout',
                'epoche_stall',
                'max_species',
                'delete_inferiors',
                'elite_ratio',
                'max_memory_usage_ratio',
            )
        }),
    )
    
    def is_production_ready_bool(self, obj=None):
        if not obj:
            return
        ret = obj.is_production_ready(as_bool=True)
        return ret
    is_production_ready_bool.boolean = True
    is_production_ready_bool.short_description = 'production ready'
    
    def error_report_str(self, obj=None):
        try:
            if not obj or not obj.error_report:
                return ''
            return obj.error_report
        except Exception, e:
            return str(e)
    error_report_str.short_description = 'error report'
    error_report_str.allow_tags = True
    
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
    
    def genestats_link(self, obj=None):
        if not obj:
            return ''
        return view_related_link(obj, 'gene_statistics')
    genestats_link.allow_tags = True
    genestats_link.short_description = 'gene statistics'
    
    def epoches_link(self, obj=None):
        if not obj:
            return ''
        return view_related_link(obj, 'epoches')
    epoches_link.allow_tags = True
    epoches_link.short_description = 'epoches'
    
    def genotypes_link(self, obj=None):
        try:
            if not obj:
                return ''
            return view_related_link(
                obj,
                'genotypes',
                template='{count} total') + '&nbsp;' + \
            view_related_link(
                obj,
                'pending_genotypes',
                extra='&fresh__exact=0&evaluating__exact=0',
                template='{count} pending') + '&nbsp;' + \
            view_related_link(
                obj,
                'evaluating_genotypes',
                extra='&evaluating__exact=1',
                template='{count} evaluating') + '&nbsp;' + \
            view_related_link(
                obj,
                'complete_genotypes',
                extra='&fitness__isnull=False&fresh__exact=1&valid__exact=1',
                template='{count} complete') + '&nbsp;' + \
            view_related_link(
                obj,
                'invalid_genotypes',
                extra='&valid__exact=0',
                template='{count} invalid') + '&nbsp;' + \
            ('<a href="%s" class="button" target="_blank">Add</a>' % (get_admin_changelist_url(models.Genotype)+'add/?genome='+str(obj.id),))
        except Exception, e:
            traceback.print_exc(file=sys.stdout)
            return str(e)
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
    
    def get_readonly_fields(self, request, obj=None, check_fieldsets=True):
        readonly_fields = list(self.readonly_fields)
        if obj and obj.evolving:
            readonly_fields.extend([f.name for f in self.model._meta.fields if f.name not in ('evolving',)])
        return readonly_fields

admin.site.register(models.Genome, GenomeAdmin)

class GenotypeGeneAdmin(BaseModelAdmin):
    
    list_display = (
        'gene',
        'genotype',
        '_value',
        'reference_link',
    )
    
    raw_id_fields = (
        'gene',
        'genotype',
    )
    
    list_editable = (
        '_value',
    )
    
    search_fields = (
        'gene__name',
        '_value',
    )
    
    readonly_fields = (
        'reference_link',
    )
    
    def reference_link(self, obj=None):
        try:
            if not obj:
                return ''
            elif obj.gene.type == c.GENE_TYPE_GENOME:
                return view_link(obj.value)
            else:
                return ''
        except Exception, e:
            return str(e)
    reference_link.allow_tags = True

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

class GenotypeAdmin(admin_steroids.options.BetterRawIdFieldsModelAdmin):
    
#    modeladmin_callbacks = set()
    
    inlines = (
#        GenotypeGeneInline,
    )
    
    list_display = (
        'id',
        'fresh',
        'valid',
        'immortal',
        'export',
        'description',
        'genome',
        'status',
        'species',
        'fitness',
#        'mean_absolute_error',
#        'accuracy',
        'mean_memory_usage_str',
        'max_memory_usage_str',
        'mean_evaluation_seconds',
        'total_evaluation_seconds',
        'success_ratio',
        'ontime_ratio',
        'complete_percent',
        'production_complete_percent2',
        'generation',
        'epoche_of_creation',
        'epoche_of_evaluation',
        #'fitness_evaluation_datetime',
        #'fingerprint_bool',
    )
    
    list_filter = (
        'genome',
        'fresh',
        'valid',
        'evaluating',
        'immortal',
        'export',
        ('fitness', NullListFilter),
        ('fingerprint', NullListFilter),
        'fingerprint_fresh',
    )
    
    search_fields = (
        'genes___value',
        #'error',
    )
    
    raw_id_fields = (
        'genome',
        'species',
    )
    
    readonly_fields = [
        'id',
        'genome',
        'genome_link',
        'status',
        'fitness',
        'species',
        'generation',
        'epoche_of_evaluation',
        'mean_evaluation_seconds',
        'mean_absolute_error',
        'fitness_evaluation_datetime_start',
        'fitness_evaluation_datetime',
        'total_evaluation_seconds',
        'gene_count',
        'fingerprint_bool',
        'mean_memory_usage_str',
        'max_memory_usage_str',
        'fingerprint',
        'fresh',
        'valid',
        'total_parts',
        'complete_parts',
        'success_parts',
        'ontime_parts',
        'success_ratio',
        'ontime_ratio',
        'error',
#        'fresh_str',
        'fingerprint_fresh',
        'genes_link',

        'complete_ratio',
        'complete_percent',
        'production_complete_ratio',
        'production_complete_percent',
        'production_complete_percent2',
                    
        'production_fresh',
        'production_valid',
        'production_error',
        'production_total_parts',
        'production_complete_parts',
        'production_success_parts',
        'production_ontime_parts',
        'production_success_ratio',
        'production_ontime_ratio',
        'production_evaluation_start_datetime',
        'production_evaluation_end_datetime',
        'production_evaluation_seconds',
        'production_evaluation_seconds_str',
    ]
    
    def get_readonly_fields(self, request, obj=None):
        lst = list(self.readonly_fields)
        if not obj:
            lst.remove('genome')
        return lst
    
#    exclude = (
#        'fresh',
#    )
    
    actions = (
        'check_fingerprint',
        'mark_fresh',
        'mark_stale',
        'mark_valid',
        'mark_invalid',
#        'refresh',
        'refresh_fitness',
        'reset',
    )
    
#    def refresh(self, request, queryset):
#        i = 0
#        for obj in queryset.iterator():
#            i += 1
#            obj.fresh = False
#            obj.save()
#        messages.success(request, '%i genotypes were refreshed.' % i)
#        return HttpResponseRedirect(request.META['HTTP_REFERER'])
#    refresh.short_description = 'Refresh selected %(verbose_name_plural)s'
    
    def reset(self, request, queryset):
        i = 0
        for obj in queryset.iterator():
            i += 1
            obj.reset()
        messages.success(request, '%i genotypes were reset.' % i)
        return HttpResponseRedirect(request.META['HTTP_REFERER'])
    reset.short_description = 'Reset selected %(verbose_name_plural)s'
    
    def mark_stale(self, request, queryset):
        queryset.update(fresh=False)
        genomes = set(queryset.values_list('genome', flat=True))
        for genome_id in genomes:
            genome = models.Genome.objects.only('id').get(id=genome_id)
            genome.mark_stale_function(queryset.filter(genome=genome))
        i = queryset.count()
        messages.success(request, '%i genotypes were marked as stale.' % i)
        return HttpResponseRedirect(request.META['HTTP_REFERER'])
    mark_stale.short_description = 'Mark selected %(verbose_name_plural)s as stale'
    
    def mark_fresh(self, request, queryset):
        queryset.update(fresh=True)
        i = queryset.count()
        messages.success(request, '%i genotypes were marked as fresh.' % i)
        return HttpResponseRedirect(request.META['HTTP_REFERER'])
    mark_fresh.short_description = 'Mark selected %(verbose_name_plural)s as fresh'
    
    def mark_valid(self, request, queryset):
        queryset.update(valid=True)
        i = queryset.count()
        messages.success(request, '%i genotypes were marked as valid.' % i)
        return HttpResponseRedirect(request.META['HTTP_REFERER'])
    mark_valid.short_description = 'Mark selected %(verbose_name_plural)s as valid'
    
    def mark_invalid(self, request, queryset):
        queryset.update(valid=False)
        i = queryset.count()
        messages.success(request, '%i genotypes were marked as invalid.' % i)
        return HttpResponseRedirect(request.META['HTTP_REFERER'])
    mark_invalid.short_description = 'Mark selected %(verbose_name_plural)s as invalid'
    
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
    
    def mean_memory_usage_str(self, obj=None):
        if not obj or not obj.mean_memory_usage:
            return ''
        return utils.sizeof_fmt(obj.mean_memory_usage)
    mean_memory_usage_str.short_description = 'mean memory usage'
    
    def max_memory_usage_str(self, obj=None):
        if not obj or not obj.max_memory_usage:
            return ''
        return utils.sizeof_fmt(obj.max_memory_usage)
    max_memory_usage_str.short_description = 'max memory usage'
    
    def genome_link(self, obj=None):
        if not obj:
            return ''
        return view_link(obj.genome)
    genome_link.short_description = 'genome'
    genome_link.allow_tags = True
    
    def get_fieldsets(self, request, obj=None):
        
        genome_field = 'genome_link' if obj else 'genome'
        
        fieldsets = [
            (None, {
                'fields': [
                    'id',
                    genome_field,
                    'description',
                    'immortal',
                    'export',
                    'fitness',
                    'genes_link',
                ]
            }),
            ('Details', {
                'classes': ('collapse',),
                'fields': [
                    'generation',
                    'fingerprint_fresh',
                    'fingerprint',
                    #'fingerprint_bool',
                    'epoche_of_evaluation',
                    'species',
                ]
            }),
            ('Test status', {
                'classes': ('collapse',),
                'fields': [
                    'complete_percent',
                    'fresh',
                    'valid',
                    'error',
                    'total_parts',
                    'complete_parts',
                    'success_parts',
                    'ontime_parts',
                    'success_ratio',
                    'ontime_ratio',
                ]
            }),
            ('Test results', {
                'classes': ('collapse',),
                'fields': [
                    'fitness_evaluation_datetime_start',
                    'fitness_evaluation_datetime',
                    'mean_memory_usage_str',
                    'max_memory_usage_str',
                    'mean_evaluation_seconds',
                    'total_evaluation_seconds',
                    'mean_absolute_error',
                    #'accuracy',
                ]
            }),
            ('Production status', {
                'classes': ('collapse',),
                'fields': [
                    'production_complete_percent',
                    'production_fresh',
                    'production_valid',
                    'production_error',
                    'production_total_parts',
                    'production_complete_parts',
                    'production_success_parts',
                    'production_ontime_parts',
                    'production_success_ratio',
                    'production_ontime_ratio',
                ]
            }),
            ('Production results', {
                'classes': ('collapse',),
                'fields': [
                    'production_evaluation_start_datetime',
                    'production_evaluation_end_datetime',
                    'production_evaluation_seconds',
                    'production_evaluation_seconds_str',
                ]
            }),
        ]
        for method in models._modeladmin_extenders.itervalues():
            evaluator_name = models.get_evaluator_name(method.im_self)
            if obj and obj.genome.evaluator == evaluator_name:
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
    
    def lookup_allowed(self, key, value=None):
        return True

admin.site.register(models.Genotype, GenotypeAdmin)

class EpocheAdmin(admin_steroids.options.BetterRawIdFieldsModelAdmin, admin_steroids.options.ReadonlyModelAdmin):
    
    list_display = (
        'index',
        'genome',
        'min_fitness',
        'mean_fitness',
        'max_fitness',
        'oldest_epoche_of_creation',
    )
    
    raw_id_fields = (
        'genome',
    )
    
admin.site.register(models.Epoche, EpocheAdmin)

class GeneStatisticsAdmin(admin_steroids.options.BetterRawIdFieldsModelAdmin, admin_steroids.options.ReadonlyModelAdmin):
    
    list_display = (
        'genome',
        'gene',
        'value',
        'min_fitness',
        'mean_fitness',
        'max_fitness',
        'genotype_count',
    )
    
    list_display_links = []
    
    list_filter = (
        'genome',
        'gene',
    )
    
    raw_id_fields = (
        'genome',
        'gene',
    )
    
    search_fields = (
        'gene__name',
    )
    
    actions = (
        'refresh',
    )
    
    def refresh(self, request, queryset):
        #queryset.update(fresh=False)
        return HttpResponseRedirect(request.META['HTTP_REFERER'])
    refresh.short_description = 'Refresh selected %(verbose_name_plural)s'
    
    def lookup_allowed(self, key, value=None):
        return True
    
admin.site.register(models.GeneStatistics, GeneStatisticsAdmin)


class LabelAdmin(admin_steroids.options.BetterRawIdFieldsModelAdmin):

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
    