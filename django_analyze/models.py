
from base64 import b64encode, b64decode
from datetime import timedelta
from StringIO import StringIO
import gc
import inspect
import importlib
import os
import random
import re
import sys
import tempfile
import time
import traceback
from collections import defaultdict

from picklefield.fields import PickledObjectField

import django
from django import forms
from django.conf import settings
from django.core.exceptions import ValidationError
from django.db import models, DatabaseError, connection, reset_queries
from django.db.models import Sum, Count, Max, Min, Q, F
from django.db.utils import IntegrityError
from django.utils import timezone
from django.utils.safestring import mark_safe
from django.utils.translation import ugettext, ugettext_lazy as _

from django_materialized_views.models import MaterializedView

try:
    from admin_steroids.utils import StringWithTitle
    APP_LABEL = StringWithTitle('django_analyze', 'Analyze')
except ImportError:
    APP_LABEL = 'django_analyze'

import constants as c
import utils

from sklearn.externals import joblib

from admin_steroids.utils import get_admin_change_url

str_to_type = {
    c.GENE_TYPE_INT:int,
    c.GENE_TYPE_FLOAT:float,
    c.GENE_TYPE_BOOL:(lambda v: True if v in (True, 'True', 1, '1') else False),
    c.GENE_TYPE_STR:(lambda v: str(v)),
}

def obj_to_hash(o):
    """
    Returns the 128-character SHA-512 hash of the given object's Pickle
    representation.
    """
    import hashlib
    import cPickle as pickle
    return hashlib.sha512(pickle.dumps(o)).hexdigest()

class BaseModel(models.Model):
    
#    created = models.DateTimeField(
#        auto_now_add=True,
#        blank=False,
#        db_index=True,
#        default=timezone.now,
#        editable=False,
#        null=False,
#        help_text="The date and time when this record was created.")
#        
#    updated = models.DateTimeField(
#        auto_now_add=True,
#        auto_now=True,
#        blank=True,
#        db_index=True,
#        default=timezone.now,
#        editable=False,
#        null=True,
#        help_text="The date and time when this record was last updated.")
#    
#    deleted = models.DateTimeField(
#        blank=True,
#        db_index=True,
#        null=True,
#        help_text="The date and time when this record was deleted.")
    
    class Meta:
        abstract = True
        
    def clean(self, *args, **kwargs):
        """
        Called to validate fields before saving.
        Override this to implement your own model validation
        for both inside and outside of admin. 
        """
        super(BaseModel, self).clean(*args, **kwargs)
        
    def full_clean(self, *args, **kwargs):
        return self.clean(*args, **kwargs)
    
    def save(self, *args, **kwargs):
        self.full_clean()
        super(BaseModel, self).save(*args, **kwargs)

class LabelManager(models.Manager):
    
    def get_by_natural_key(self, name):
        return self.get(name=name)

class Label(BaseModel):
    """
    A generic model for defining labels to manually and
    automatically classify records.
    """
    
    objects = LabelManager()

    name = models.CharField(
        max_length=1000,
        blank=False,
        unique=True,
        null=True)
    
    duplicate_of = models.ForeignKey(
        'self',
        blank=True,
        null=True,
        related_name='%(app_label)s_%(class)s_duplicate_labels',
        help_text=_('Indicates the record that this one duplicates.'))
    
    _logical = models.ForeignKey(
        'self',
        blank=True,
        null=True,
        db_column='logical',
        related_name='%(app_label)s_%(class)s_logical_labels',
        help_text=_('''Follows the duplicate_of chain and caches the first
            non-duplicate.'''))
    
    def get_logical(self):
        current = self
        while 1:
            if not current.duplicate_of:
                break
            current = current.duplicate_of
        return current

    class Meta:
        abstract = True
    
    def __unicode__(self):
        return self.name
    
    def natural_key(self):
        return (self.name,)
    natural_key.dependencies = []
    
    def save(self, *args, **kwargs):
        
#        old = None
#        if self.id:
#            old = type(self).objects.get(id=self.id)
        
        self.name = (self.name or '').strip().lower()
        
        self._logical = self.get_logical()
        
        super(Label, self).save(*args, **kwargs)
        
class Predictor(BaseModel, MaterializedView):
    """
    A general base class for creating an object that takes training samples
    and constructs a model for tagging future samples with one of several
    pre-defined classes.
    
    Questions you should answer before defining your own subclass:
    1. What type of classification do you want to do?
        Simple (boolean), multiclass, or multilabel classification?
        http://scikit-learn.org/stable/modules/multiclass.html
        http://scikit-learn.org/stable/auto_examples/plot_multilabel.html
    2. Should the predictor be trained in a batch or incrementally?
        * How you answer this will potentially have a huge impact on performance
        and which algorithms will be available to you.
        * Batch algorithms tend to be the most accurate but are also usually
        the most time-consuming. Ideal for small datasets.
        Scikits-learn implements this with its fit() method.
        * Sliding Window. This is essentially a batch algorithm that's
        only given the N most recent or most applicable samples of a data.
        * Online/incremental algorithms are often less inaccurate, or tend to
        "forget" patterns learned early on, but they can update their model
        extremely quickly. Ideal for large streaming datasets where new
        information must be assimilated close to real-time.
        * Mini-batch algorithms lie between batch and online/incremental
        algorithms. This is essentially an online algorithm that
        performs a batch training operation of a small set of data and then
        merges it into the prior model. Scikits-learn implements this as
        the partial_fit() method on many of its algorithms.
    3. What should be the minimum number of training samples and training
        accuracy required before the predictor can start making
        classifications?
    4. What kind of seed training data should be generated?
        e.g. Say you want have N people who you want to detect references
        of in M documents. If N is very large, it may be too burdensome to
        manually create a minimum level of training data.
        However, it may be trivial to generate some initial training data
        by finding documents containing each person's exact name.
        Of course, there will likely be many exceptions leading, but for most
        people, this seed data will probably generate a useable accuracy.
        As exceptions are found, they can be manually tagged and the predictor
        retrained accordingly.
    """
    
    include_in_batch = False
    
    max_i = 10
    
    # If true, old classifications will be updated.
    # If false, old classifications will never be updated.
    # Should be true for models that grow very slowly or are
    # of long-term interest.
    # Should be false for models that grow quickly or are short-term interest.
    revise_classifications = True
    
    # The fewest positive training samples needed FOR EACH CLASS
    # before the predictor can be trained and used for classifying.
    min_training_samples = 100
    
    # The minimum empirical classification accuracy needed before classifications
    # are published.
    min_empirical_accuracy = 0.5
    
    # The minimum number of records, ordered in reverse chronological order,
    # used for calculation of the empirical accuracy.
    # e.g. If this is 10 then the last 10 classifications that intersect with
    # manual classifications will be used to calculate the empirical accuracy.
    min_empirical_window = 10
    
    # The maximum number of most recent examples to use for training.
    max_training_window = 1000
    
    # The minimum certainty the guess must have before it's acted upon.
    min_certainty = 0.5 # [0.0:1.0]
    
    name = models.CharField(
        max_length=100,
        #choices=c.CLASSIFIER_CHOICES,
        #default=c.DEFAULT_CLASSIFIER_NAME,
        blank=True,
        null=True)
    
#    classifying_person = models.ForeignKey(
#        im_models.Person,
#        related_name='classifiers',
#        default=lambda: im_models.Person.objects.get(
#            user__username=settings.IM_DEFAULT_USERNAME),
#        blank=True,
#        null=True,
#        help_text='The person who will be associated with created records.')
    
    training_r2 = models.FloatField(
        blank=True,
        null=True,
        db_index=True,
        editable=False,
        help_text='The r2 measure recorded during training.')
    
    training_mean_squared_error = models.FloatField(
        blank=True,
        null=True,
        db_index=True,
        editable=False,
        help_text='The mean-squared-error measure recorded during training.')
    
    training_mean_absolute_error = models.FloatField(
        blank=True,
        null=True,
        db_index=True,
        editable=False,
        help_text='The mean-absolute-error measure recorded during training.')
    
    training_seconds = models.PositiveIntegerField(
        blank=True,
        null=True,
        help_text='The number of CPU seconds the algorithm spent training.')
    
    training_ontime = models.BooleanField(
        default=True,
        blank=False,
        null=False,
        help_text='''If false, indicates this predicator failed to train within
            the allotted time, and so it is not suitable for use.''',
    )
    
    testing_r2 = models.FloatField(
        blank=True,
        null=True,
        db_index=True,
        editable=False,
        help_text='The r2 measure recorded during testing.')
    
    testing_mean_squared_error = models.FloatField(
        blank=True,
        null=True,
        db_index=True,
        editable=False,
        help_text='The mean-squared-error measure recorded during testing.')
    
    testing_mean_absolute_error = models.FloatField(
        blank=True,
        null=True,
        db_index=True,
        editable=False,
        help_text='The mean-absolute-error measure recorded during testing.')
    
    training_accuracy = models.FloatField(
        blank=True,
        null=True,
        editable=False,
        help_text='The accuracy of the classifier predicting its training input.')
    
    min_training_accuracy = models.FloatField(
        blank=True,
        null=True,
        default=0.75,
        editable=True,
        help_text='''The minimum training accuracy allowed in order for this
            classifier to be used for production classification.''')
    
    training_classification_report = models.TextField(
        blank=True,
        null=True,
        help_text='''Arbitrary human-readable text summary of the last training session.''')
    
    trained_datetime = models.DateTimeField(
        blank=True,
        db_index=True,
        editable=False,
        null=True,
        help_text="The date and time when this classifier was last trained.")
    
    classified_datetime = models.DateTimeField(
        blank=True,
        db_index=True,
        editable=False,
        null=True,
        help_text="The date and time when this classifier last classified.")
    
    #classifier = PickledObjectField(blank=True, null=True)
    _predictor = models.TextField(
        blank=True,
        null=True,
        editable=False,
        db_column='classifier')
    
    @property
    def predictor(self):
        data = self._predictor
        if data:
            _, fn = tempfile.mkstemp()
            fout = open(fn, 'wb')
            fout.write(b64decode(data))
            fout.close()
            obj = joblib.load(fn)
            os.remove(fn)
            return obj
        else:
            return
        
    @predictor.setter
    def predictor(self, v):
        if v:
            _, fn = tempfile.mkstemp()
            joblib.dump(v, fn, compress=9)
            self._predictor = b64encode(open(fn, 'rb').read())
            os.remove(fn)
        else:
            self._predictor = None
    
    empirical_classification_count = models.PositiveIntegerField(
        default=0,
        editable=False,
        help_text='The total number of real classifications performed.')
    
    correct_empirical_classification_count = models.PositiveIntegerField(
        default=0,
        editable=False,
        help_text='The total number of real classifications performed that match the supervisor.')
    
    empirical_accuracy = models.FloatField(
        blank=True,
        null=True,
        editable=False,
        help_text='The empirical accuracy of the predictor predicting the supervisor.')
    
    predicted_value = models.FloatField(
        blank=True,
        null=True,
        db_index=True,
        help_text='The future value predicted by the algorithm.')
    
    predicted_prob = models.FloatField(
        blank=True,
        null=True,
        db_index=True,
        help_text='The future value probability by the algorithm.')
    
    predicted_score = models.FloatField(
        blank=True,
        null=True,
        db_index=True,
        help_text='How well this prediction compares to others.')
    
    fresh = models.NullBooleanField(
        editable=False,
        db_index=True,
        help_text='If true, indicates this predictor is ready to classify.')

    #@classmethod
    #def create(cls):

    class Meta:
        abstract = True
    
    def create(self, *args, **kwargs):
        raise NotImplementedError
    
    def populate(self):
        """
        Called to bulk create predictors for each target class.
        """
        raise NotImplementedError
    
    def label(self, cls, sample):
        """
        Upon successful classification, is called to record the label
        association to the sample.
        
        This should create:
        * the label association
        * a link to the record storing the predictor instance
        """
        raise NotImplementedError
    
    def train(self):
        todo
        
    def predict(self):
        todo

def get_class_that_defined_method(meth):
    if hasattr(meth, 'im_self'):
        return meth.im_self
    for cls in inspect.getmro(meth.im_class):
        print 'cls:',cls
        if meth.__name__ in cls.__dict__:
            return cls
    return None

def get_method_name(func):
    return get_class_that_defined_method(func).__name__ + '.' +func.func_name

_evaluators = {}
_modeladmin_extenders = {}

def register_evaluator(cls):
    assert hasattr(cls, 'evaluate_genotype'), \
        'Class %s does not implement `evaluate_genotype`.' % (cls.__name__,)
    assert hasattr(cls, 'reset_genotype'), \
        'Class %s does not implement `reset_genotype`.' % (cls.__name__,)
    assert hasattr(cls, 'calculate_genotype_fitness'), \
        'Class %s does not implement `calculate_genotype_fitness`.' % (cls.__name__,)
    #name = get_method_name(cls)
    name = cls.__module__ + "." + cls.__name__
    _evaluators[name] = cls

def register_modeladmin_extender(func):
    name = get_method_name(func)
    _modeladmin_extenders[name] = func

def get_evaluators():
    for name in sorted(_evaluators.iterkeys()):
        yield (name, name)

def get_extenders():
    for name in sorted(_modeladmin_extenders.iterkeys()):
        yield (name, name)

class SpeciesManager(models.Manager):

    def get_by_natural_key(self, index, *args, **kwargs):
        genome = Genome.objects.get_by_natural_key(*args, **kwargs)
        return self.get(index=index, genome=genome)
    
class Species(BaseModel):
    
    objects = SpeciesManager()
    
    genome = models.ForeignKey(
        'Genome',
        related_name='species')
    
    index = models.PositiveIntegerField(
        default=0,
        db_index=True)
    
    centroid = PickledObjectField(blank=True, null=True)
    
    population = models.PositiveIntegerField(default=0)

    class Meta:
        app_label = APP_LABEL
        verbose_name = _('species')
        verbose_name_plural = _('species')
        unique_together = (
            ('genome', 'index'),
        )
        index_together = (
            ('genome', 'index'),
        )
        ordering = ('genome', 'index')
    
    def __unicode__(self):
        return self.letter()
    
    def natural_key(self):
        return (self.index,) + self.genome.natural_key()
    natural_key.dependencies = ['genome']
    
    def letter(self):
        if self.index <= 0 or self.index >= 27:
            return
        return chr(ord('A')-1+self.index)
    letter.admin_order_field = 'index'
    
    def save(self, *args, **kwargs):
        if self.id:
            self.population = self.genotypes.all().count()
        if not self.index:
            self.index = self.genome.species.all().count() + 1
        super(Species, self).save(*args, **kwargs)

class GenomeManager(models.Manager):
    
    def get_by_natural_key(self, name):
        return self.get(name=name)

class Genome(BaseModel):
    """
    All possible parameters of a problem domain.
    """
    
    objects = GenomeManager()
    
    name = models.CharField(
        max_length=100,
        blank=False,
        null=False,
        unique=True)
    
    evaluator = models.CharField(
        max_length=1000,
        choices=get_evaluators(),
        verbose_name=_('backend'),
        blank=True,
        null=True,
        help_text=_('The backend to use when evaluating genotype fitness.'))
    
#    admin_extender = models.CharField(#TODO:remove?
#        max_length=1000,
#        choices=get_extenders(),
#        blank=True,
#        help_text='Method to call to selectively extend the admin interface.',
#        null=True)
    
    maximum_population = models.PositiveIntegerField(
        default=10,
        verbose_name=_('maximum unevaluated population'),
        help_text=_('''The maximum number of new genotype records to create
            each epoche. If set to zero, no limit will be enforced.''')
    )
    
    maximum_evaluated_population = models.PositiveIntegerField(
        default=1000,
        help_text=_('''The maximum number of evaluted genotype records
            to store indefinitely.
            If set to zero, no limit will be enforced.
            If delete_inferiors is checked, all after this top amount, ordered
            by fitness, will be deleted.''')
    )
    
    mutation_rate = models.FloatField(
        default=0.1,
        blank=False,
        null=False,
        help_text=_('''The probability of each gene being randomly altered
            during creation of a mutated genotype.'''))
    
    evaluation_timeout = models.PositiveIntegerField(
        default=300,
        blank=False,
        null=False,
        help_text=_('''The number of seconds to the genotype will allow
            training before forcibly terminating the process.<br/>
            A value of zero means no timeout will be enforced.<br/>
            Note, how this value is enforced, if at all, is up to the
            genotype backend.<br/>
            e.g. If a genotype runs multiple evaluation methods internally,
            this may be used on each individual method, not the overall
            evaluation.
        '''))
    
    epoche = models.PositiveIntegerField(
        default=0,
        blank=False,
        null=False,
        help_text=_('''The number of epoches thus far evaluated. This number
            will also be used to seed an epoche-specific random number
            generator for each genotype.'''))
    
    epoches_since_improvement = models.PositiveIntegerField(
        default=0,
        blank=False,
        null=False,
        editable=True,
        help_text=_('''The number of epoches since an increase in the maximum
            fitness was observed.'''))
    
    epoche_stall = models.PositiveIntegerField(
        default=10,
        blank=False,
        null=False,
        editable=True,
        help_text=_('''The number of epoches to process without seeing
            a maximum fitness increase before stopping.'''))
    
    def stalled(self):
        return self.epoches_since_improvement >= self.epoche_stall
    stalled.boolean = True
    
    def improving(self):
        return not self.stalled()
    improving.boolean = True
    
    max_species = models.PositiveIntegerField(
        default=10,
        blank=False,
        null=False,
        help_text=_('''The number of genotype clusters to track.'''))
#    
#    track_species = models.BooleanField(
#        default=False,
#        help_text=_('''If checked, all genotypes will be grouped into one of N clusters.'''))
    
    delete_inferiors = models.BooleanField(
        default=False,
        help_text=_('''If checked, all but the top fittest genotypes
            will be deleted. Requires maximum_population to be set to
            a non-zero value.''')
    )
    
    min_fitness = models.FloatField(
        blank=True,
        null=True,
        editable=False,
        help_text=_('The smallest observed fitness.'))
    
    max_fitness = models.FloatField(
        blank=True,
        null=True,
        editable=False,
        help_text=_('The largest observed fitness.'))
    
    production_genotype_auto = models.BooleanField(
        default=False,
        help_text=_('''If checked, the `production_genotype` will automatically
            be set to the genotype with the best fitness.'''))
    
    production_genotype = models.ForeignKey(
        'Genotype',
        related_name='production_genomes',
        on_delete=models.SET_NULL,
        blank=True,
        null=True)
    
    evaluating_part = models.PositiveIntegerField(
        default=0,
    )
    
    ratio_evaluated = models.FloatField(
        blank=True,
        null=True,
    )
    
    class Meta:
        app_label = APP_LABEL
        verbose_name = _('genome')
        verbose_name_plural = _('genomes')
    
    def __unicode__(self):
        return self.name
    
    def natural_key(self):
        return (self.name,)
    natural_key.dependencies = []
    
    def save(self, *args, **kwargs):
        
        if self.id:
            
            old = type(self).objects.get(id=self.id)
            
            q = self.genotypes.filter(fitness__isnull=False).exclude(fitness=float('nan'))\
                .aggregate(Max('fitness'), Min('fitness'))
            self.max_fitness = q['fitness__max']
            self.min_fitness = q['fitness__min']
        
            #TODO:how to handle species that need to be deleted?
            missing_species = max(0, self.max_species - self.species.all().count())
            for _ in xrange(missing_species):
                print 'creating',_
                species = Species(genome=self)
                species.save()
                
            if self.max_fitness > old.max_fitness:
                self.epoches_since_improvement = 0
        
        super(Genome, self).save(*args, **kwargs)
    
    def organize_species(self, all=False, **kwargs):
        """
        Assigns a species to each genotype.
        """
        
        class Cluster:
            def __init__(self, species):
                self.species = species
                self._centroid = species.centroid or {}
                self.population = []#list(species.genotypes.all())
                self.centroid_fresh = True
                
            def add(self, genotype):
                self.centroid_fresh = False
                self.population.append(genotype)
                
            def empty(self):
                return not bool(len(self.population))
            
            def refresh_centroid(self):
                todo
                
            @property
            def centroid(self):
                if not self.centroid_fresh:
                    
                    # Determine all the unique values for all gene names.
                    values = defaultdict(list)
                    for genotype in self.population:
                        for k,v in genotype.as_dict().iteritems():
                            values[k].append(v)
                            
                    value_counts = {}
                    for k,v in values.iteritems():
                        types = list(set(type(_) for _ in v))
                        assert len(types) == 1, 'Gene %s uses more than two value types: %s' % (k, ', '.join(map(str, types)))
                        if types[0] in (int, float, bool):
                            value_counts[k] = sum(v)/float(len(v))
                        elif types[0] in (str, unicode, basestring):
                            # Get nominal value used the most.
                            _counts = defaultdict(int)
                            for _ in v:
                                _counts[_] += 1
                            most_v, most_count = sorted(_counts.iteritems(), key=lambda o:(o[1],o[0]), reverse=True)[0]
                            value_counts[k] = most_v
                        else:
                            raise NotImplementedError, 'Unknown type: %s' % (types[0],)
                    
                    for k,v in value_counts.iteritems():
                        print k,v
                    self._centroid = value_counts
                return self._centroid
            
            def measure(self, genotype):
                centroid = self.centroid
                score = []
                other_data = genotype.as_dict()
                for k,v in centroid.iteritems():
                    other_v = other_data.get(k)
                    print k,v,other_v
                    if isinstance(v, (int, float, bool)):
                        if other_v is None:
                            other_v = 0
                        score.append(abs(v-other_v))
                    elif isinstance(v, basestring):
                        score.append(v == other_v)
                    else:
                        raise NotImplementedError
                if not score:
                    return 1e9999999999999
                return sum(score)/float(len(score))
                    
            def save(self):
                print 'Saving species %s...' % (self.species.letter(),)
                self.species.centroid = self._centroid
                for gt in self.population:
                    gt.species = self.species
                    gt.save()
                self.species.save()
        
        species = list(self.species.all())
        assert len(species) >= 2, 'There must be two or more species.'
        
        clusters = [Cluster(s) for s in species]
        empty_clusters = [_ for _ in clusters if _.empty()]
        
        q = self.genotypes.all()
        if not all:
            q = q.filter(species__isnull=True)
        pending = list(q)
        total = len(pending)
        i = 0
        while pending:
            i += 1
            genotype = pending.pop(0)
            print 'genotype %s (%i of %i)' % (genotype, i, total)
            if all or not genotype.species:
                if empty_clusters:
                    cluster = empty_clusters.pop(0)
                    print '%s -> %s' % (genotype, cluster.species.letter())
                    cluster.add(genotype)
                else:
                    scores = []
                    for cluster in clusters:
                        scores.append((cluster.measure(genotype), cluster))
                    measure, cluster = sorted(scores)[0]
                    cluster.add(genotype)
                    
        for cluster in clusters:
            cluster.save()
        print 'clusters saved'
    
    def total_possible_genotypes(self):
        """
        Calculates the maximum number of unique genotypes given the product
        of the unique values of all genes.
        """
        import operator
        values = [gene.get_max_value_count() for gene in self.genes.all()]
        return utils.sci_notation(reduce(operator.mul, values, 1))
    
    def is_allowable_gene(self, priors, next):
        """
        Given a dict of {gene:value} representing a hypothetical genotype,
        determines if the next proposed gene is applicable.
        """
        assert isinstance(next, Gene)
        if not next.dependee_gene:
            return True
        elif priors.get(next.dependee_gene) == next.dependee_value:
            return True
        return False
    
    def create_random_genotype(self):
        """
        Generates a genotype with a random assortment of genes.
        """
        d = {}
        for gene in self.genes.all().order_by('-dependee_gene__id'):
            if not self.is_allowable_gene(priors=d, next=gene):
                continue
            d[gene] = gene.get_random_value()
        new_genotype = Genotype(genome=self)
        new_genotype.save(check_fingerprint=False)
        GenotypeGene.objects.bulk_create([
            GenotypeGene(genotype=new_genotype, gene=gene, _value=str(value))
            for gene, value in d.iteritems()
        ])
        try:
            new_genotype.save() # Might raise fingerprint error
        except ValidationError, e:
            print>>sys.stderr, '!'*80
            print>>sys.stderr, 'Validation Error: %s' % (e,)
            sys.stderr.flush()
            connection._rollback()
        except IntegrityError, e:
            print>>sys.stderr, '!'*80
            print>>sys.stderr, 'Integrity Error: %s' % (e,)
            sys.stderr.flush()
            connection._rollback()
        return new_genotype
    
    def create_hybrid_genotype(self, genotypeA, genotypeB):
        from collections import defaultdict
        new_genotype = Genotype(
            genome=self,
            generation=max(genotypeA.generation, genotypeB.generation)+1,
        )
        new_genotype.save(check_fingerprint=False)
        
        all_values = defaultdict(list) # {gene:[values]}
        for gene in genotypeA.genes.all():
            all_values[gene.gene].append(gene.value)
        for gene in genotypeB.genes.all():
            all_values[gene.gene].append(gene.value)
            
        # Note, crossover may result in many invalid or duplicate genotypes,
        # so we can't check the fingerprint until the gene selection
        # is complete.
        priors = {}
        # Order independent genes first so we don't automatically ignore
        # dependent genes just because we haven't added their dependee yet.
        genes = sorted(all_values.iterkeys(), key=lambda gene: gene.dependee_gene)
#        print
#        print 'new_genotype:',new_genotype
#        print 'genes:',genes
#        print 'all_values:',all_values
        sys.stdout.flush()
        for gene in genes:
            if not self.is_allowable_gene(priors=priors, next=gene):
                continue
            new_value = random.choice(all_values[gene])
#            print 'new gene:',gene,new_value
            new_gene = GenotypeGene.objects.create(
                genotype=new_genotype,
                gene=gene,
                _value=new_value)
            priors[gene] = new_value
#        print 'priors:',priors
        self.add_missing_genes(new_genotype)
        new_genotype.delete_illegal_genes()
        new_genotype.fingerprint = None
        new_genotype.fingerprint_fresh = False
        new_genotype.save(check_fingerprint=False)
        try:
            new_genotype.save() # Might raise fingerprint error
        except ValidationError, e:
            print>>sys.stderr, '!'*80
            print>>sys.stderr, 'Validation Error: %s' % (e,)
            sys.stderr.flush()
            connection._rollback()
        except IntegrityError, e:
            print>>sys.stderr, '!'*80
            print>>sys.stderr, 'Integrity Error: %s' % (e,)
            sys.stderr.flush()
            connection._rollback()
        return new_genotype
        
    def create_mutant_genotype(self, genotype):
        new_genotype = Genotype(genome=self, generation=genotype.generation+1)
        new_genotype.save(check_fingerprint=False)
        priors = {}
        
        ggenes = genotype.genes.order_by('-gene__dependee_gene__id')
        ggene_count = ggenes.count()
        ggene_weights = dict((ggene, ggene.gene.mutation_weight or 0) for ggene in ggenes.iterator())
        ggene_weights_sum = sum(ggene.gene.mutation_weight or 0 for ggene in ggenes.iterator())
        
        # Randomly select K elements and add weight then do weighted selection.
        k = max(1, int(round(self.mutation_rate * ggene_count, 0)))
        mutated_genes = set(utils.weighted_samples(choices=ggene_weights, k=k))
        for ggene in ggenes.iterator():
#            if not self.is_allowable_gene(priors=priors, next=gene.gene):
#                continue
            new_gene = ggene
            new_gene.id = None
            new_gene.genotype = new_genotype
#            if random.random() <= self.mutation_rate:
            if ggene in mutated_genes:
                print 'Mutating gene:',ggene
                new_gene._value = ggene.get_random_value()
            new_gene.save()
#            priors[new_gene.gene] = new_gene._value
            
        self.add_missing_genes(new_genotype)
        new_genotype.fingerprint = None
        new_genotype.fingerprint_fresh = False
        new_genotype.save(check_fingerprint=False)
        try:
            new_genotype.save() # Might raise fingerprint error
        except ValidationError, e:
            print>>sys.stderr, '!'*80
            print>>sys.stderr, 'Validation Error: %s' % (e,)
            sys.stderr.flush()
            connection._rollback()
        except IntegrityError, e:
            print>>sys.stderr, '!'*80
            print>>sys.stderr, 'Integrity Error: %s' % (e,)
            sys.stderr.flush()
            connection._rollback()
        return new_genotype
    
    @property
    def pending_genotypes(self):
        """
        Returns genotypes that have not yet had their fitness evaluated.
        """
        return self.genotypes.filter(fresh=False, evaluating=False)
    
    @property
    def valid_genotypes(self):
        return Genotype.objects.valid().filter(genome=self)
    
    def delete_corrupt(self, save=True):
        """
        Deletes genotypes without a fingerprint, which should only happen
        because it collided with a duplicate genotype.
        """
        
        # Delete all genotypes that couldn't render a fingerprint, implying
        # it's a duplicate.
        for gt in self.genotypes.filter(fingerprint__isnull=True).iterator():
            print 'Deleting corrupt genotype %s...' % (gt,)
            gt.delete()
        
        # Delete all genotype genes that are illegal.
        genotype_ids = set()
        q = GenotypeGeneIllegal.objects.all().values_list('genotype', flat=True).distinct()
        total = q.count()
        if total:
            print 'Deleting illegal genes from %i genotypes.' % (total,)
            for gt in q.iterator():
                gt.delete_illegal_genes(save=save)
                genotype_ids.add(gt.id)
        Genotype.mark_stale(genotype_ids, save=save)
    
    def populate(self):
        """
        Creates random genotypes until the maximum limit for un-evaluated
        genotypes is reached.
        """
        max_retries = 10
        last_pending = None
        populate_count = 0
        max_populate_retries = 1000
        print 'Populating genotypes...'
        while 1:
            
            # Delete failed and/or corrupted genotypes.
            self.delete_corrupt()
            
            #TODO:only look at fitness__isnull=True if maximum_population=0 or delete_inferiors=False?
            pending = self.pending_genotypes.count()
            if pending >= self.maximum_population:
                print 'Maximum unevaluated population has been reached.'
                break
            
            # If there are literally no more unique combinations to find,
            # then don't bother.
            if self.genotypes.count() >= self.total_possible_genotypes():
                print 'Maximum theoretical population has been reached.'
                break
            
            # Note, because we don't have a good way of avoiding hash collisons
            # due to the creation of duplicate genotypes, it's possible we may
            # not generate unique genotypes within a reasonable amount of time
            # so we keep track of our failures and stop after too many
            # attempts.
            if last_pending is not None and last_pending == pending:
                populate_count += 1
                if populate_count > max_populate_retries:
                    break
            else:
                populate_count = 0
            print ('='*80)+'\nAttempt %i of %i to create %i, currently %i' % (
                populate_count,
                max_populate_retries,
                self.maximum_population,
                pending)
            sys.stdout.flush()
            
            valid_genotypes = self.valid_genotypes
            random_valid_genotypes = valid_genotypes.order_by('?')
            last_pending = pending
            creation_type = random.randint(1,10)
            for retry in xrange(max_retries):
                genotype = None
#                try:
                if valid_genotypes.count() <= 1 or creation_type == 1:
                    genotype = self.create_random_genotype()
                elif valid_genotypes.count() >= 2 and creation_type < 5:
                    a = utils.weighted_choice(
                        random_valid_genotypes,
                        get_total=lambda: valid_genotypes.aggregate(Sum('fitness'))['fitness__sum'],
                        get_weight=lambda o:o.fitness)
                    b = utils.weighted_choice(
                        random_valid_genotypes,
                        get_total=lambda: valid_genotypes.aggregate(Sum('fitness'))['fitness__sum'],
                        get_weight=lambda o:o.fitness)
                    #genotype = self.create_hybrid_genotype(random_valid_genotypes[0], random_valid_genotypes[1])
                    genotype = self.create_hybrid_genotype(a, b)
                else:
                    genotype = self.create_mutant_genotype(random_valid_genotypes[0])
                break
    
    @property
    def evaluator_function(self):
        return _evaluators.get(self.evaluator).evaluate_genotype
    
    @property
    def reset_function(self):
        return _evaluators.get(self.evaluator).reset_genotype
    
    @property
    def calculate_fitness_function(self):
        return _evaluators.get(self.evaluator).calculate_genotype_fitness
    
    @property
    def admin_extender_function(self):
        return _modeladmin_extenders.get(self.admin_extender)
    
    def add_missing_genes(self, genotype=None, save=True):
        """
        Find all genotype gene values that should exist but don't,
        and creates them.
        """
        genotype_ids = set()
        q = GenotypeGeneMissing.objects.all()
        if genotype:
            q = q.filter(genotype=genotype)
        total = q.count()
        if total:
            print 'Adding %i missing gene values.' % (total,)
            i = 0
            for missing in q.iterator():
                i += 1
                print '\rAdding gene value %i of %i...' % (i, total),
                sys.stdout.flush()
                genotype_ids.add(missing.genotype_id)
                missing.create()
            # Check a second time in case we added a dependee gene
            # which should now catch any missing dependent genes.
            for missing in q.iterator():
                genotype_ids.add(missing.genotype_id)
                missing.create()
        Genotype.mark_stale(genotype_ids, save=save)
        
#        while 1:
#            added = False
#            
#            # Add missing independent genes.
#            independent_genes = self.genes.filter(dependee_gene__isnull=True)
#            incomplete_genotype_ids = set()
#            #TODO:inefficient, just use simple left outer join instead?
#            for igene in independent_genes:
#                incomplete_genotype_ids.update(Genotype.objects.exclude(genes__gene=igene).values_list('id', flat=True))
#            q = Genotype.objects.filter(id__in=incomplete_genotype_ids)
#            if genotype:
#                q = q.filter(id=genotype.id)
##            print 'add_missing_genes.q:',q
##            sys.exit()#TODO:remove
#            for gt in q.iterator():
#                print 'add_missing_genes.current_genes:',gt.genes.all()
#                missing_genes = independent_genes.exclude(id__in=gt.genes.values_list('gene__id', flat=True))
#                print 'add_missing_genes.gt:',missing_genes
#                just_added = False
#                for gene in missing_genes.iterator():
#                    GenotypeGene.objects.get_or_create(genotype=gt, gene=gene, _value=gene.default)#TODO:enforce valid default if values set?
#                    added = True
#                    just_added = True
#                if just_added:
#                    # If we just added genes, we may have changed the
#                    # fingerprint to one that may collide with another
#                    # genotype.
#                    try:
#                        gt.fingerprint_fresh = False
#                        gt.save()
#                    except DatabaseError, e:
#                        print>>sys.stderr, '!'*80
#                        print>>sys.stderr, 'Add missing genes error: %s' % (e,)
#                        sys.stderr.flush()
#                        connection._rollback() # Needed by Postgres.
#            
#            # Add missing dependent genes.
#            dependent_genes = self.genes.filter(dependee_gene__isnull=False)
#            q = self.genotypes.exclude(genes__gene__id__in=dependent_genes.values_list('id', flat=True))
#            if genotype:
#                q = q.filter(id=genotype.id)
#            #print 'q:',q
#            for gt in q.iterator():
#                missing_genes = [
#                    dependent_gene for dependent_gene in dependent_genes.exclude(id__in=gt.genes.values_list('gene__id', flat=True))
#                    if gt.genes.filter(gene=dependent_gene.dependee_gene, _value=dependent_gene.dependee_value).count()]
#                #print 'gt:',missing_genes
#                just_added = False
#                for gene in missing_genes:
#                    GenotypeGene.objects.get_or_create(genotype=gt, gene=gene, _value=gene.default)
#                    added = True
#                    just_added = True
#                if just_added:
#                    # If we just added genes, we may have changed the
#                    # fingerprint to one that may collide with another
#                    # genotype.
#                    try:
#                        gt.fingerprint_fresh = False
#                        gt.save()
#                    except DatabaseError, e:
#                        print>>sys.stderr, '!'*80
#                        print>>sys.stderr, 'Add missing genes error: %s' % (e,)
#                        sys.stderr.flush()
#                        connection._rollback() # Needed by Postgres.
#                
#            if not added:
#                break
                
    def evolve(self,
        genotype_id=None,
        populate=True,
        evaluate=True,
        force_reset=False,
        continuous=False,
        processes=0):
        """
        Runs a one or more cycles of genotype deletion, generation and evaluation.
        """
        from multiprocessing import Process, Queue, Lock
        
        assert processes >= 0
        
#        assert not processes or utils.is_power_of_two(processes), \
#            'Processes must be a power of 2.'
        
        tmp_debug = settings.DEBUG
        settings.DEBUG = False
        try:
            while 1:
                
                #TODO:Delete the worst if population at max.
                
                #TODO:Mark old evaluated genotypes as stale after N days to force re-evaluation.
                
                # Delete genotypes that are incomplete or duplicates.
                print 'Deleting corrupt genotypes...'
                self.delete_corrupt()
                
                # Add missing genes to genotypes.
                print 'Adding missing genes...'
                self.add_missing_genes()
            
                # Creates the initial genotypes.
                if populate:
                    print 'Populating...'
                    self.populate()
                
                # Evaluate un-evaluated genotypes.
                #max_fitness = self.max_fitness
                if evaluate:
                    if genotype_id:
                        self.evaluate(genotype_id=genotype_id, force_reset=force_reset)
                        return
                    else:
                        
                        process_stack = []
                        lock = Lock()
                        if processes:
                            # Start processes.
                            for _ in xrange(processes):
                                django.db.connection.close()
                                p = Process(target=self.evaluate, kwargs=dict(lock=lock))
                                #p.daemon = True#breaks evaluate() launching processes of its own
                                p.start()
                                process_stack.append(p)
                            # Wait for processes to end.
                            while any(p for p in process_stack if p.is_alive()):
                                time.sleep(0.1)
                        else:
                            self.evaluate(genotype_id=genotype_id, force_reset=force_reset)
                        
                        Genome.objects.update()
                        genome = Genome.objects.get(id=self.id)
                        genome.epoche += 1
                        genome.epoches_since_improvement += 1
                        #type(self).objects.filter(id=self.id).update(epoche=self.epoche)
                        genome.save()
                else:
                    return
                
                Genome.objects.update()
                genome = Genome.objects.get(id=self.id)
                if not continuous or genome.stalled():
                    break
                
                # Clear the query cache to help reduce memory usage.
                reset_queries()
        
        finally:
            settings.DEBUG = tmp_debug
            #django.db.transaction.commit()
            
    def evaluate(self, genotype_id=None, force_reset=False, lock=None, fout=None):
        """
        Calculates the fitness of all currently unevaluated genotypes
        using the genome's linked fitness metric.
        """
        fout = fout or sys.stdout
        
        while 1:
            
            # Build query to retrieve next genotype to evaluate.
            if force_reset:
                print>>fout, 'Retrieving all genotypes...'
                q = self.genotypes.all()
            else:
                print>>fout, 'Retrieving pending genotypes...'
                q = self.pending_genotypes
            if genotype_id:
                q = q.filter(id=genotype_id)
            
            # Retrieve the next genotype.
            try:
                print 'lock:',lock
                if lock:
                    lock.acquire()
                if q.exists():
                    Genotype.objects.update()
                    gt = q[0]
                    gt.reset()
                    gt.evaluating = True
                    gt.evaluating_pid = os.getpid()
                    gt.fitness_evaluation_datetime_start = timezone.now()
                    gt.save()
                else:
                    # Nothing left to evaluate.
                    return
            finally:
                if lock:
                    lock.release()
            
            # Run the backend evaluator.
            try:
                self.evaluator_function(gt)
                gt.fitness_evaluation_datetime = timezone.now()
                gt.evaluating = False
                gt.evaluating_pid = None
                gt.fresh = True
                gt.valid = True
                gt.error = None
                gt.save()
                reset_queries()
            except Exception, e:
                print>>sys.stderr, 'Error evaluating genotype %i.' % (gt.id,)
                fout = StringIO()
                traceback.print_exc(file=fout)
                error = fout.getvalue()
                print>>sys.stderr, error
                sys.stderr.flush()
                django.db.connection.close()
                Genotype.objects.filter(id=gt.id).update(
                    fresh=True,
                    valid=False,
                    error=error,
                )

class GeneManager(models.Manager):
    
    def get_by_natural_key(self, name, *args, **kwargs):
        genome = Genome.objects.get_by_natural_key(*args, **kwargs)
        return self.get(name=name, genome=genome)

class Gene(BaseModel):
    """
    Describes a specific configurable settings in a problem domain.
    """
    
    objects = GeneManager()
    
    genome = models.ForeignKey(Genome, related_name='genes')
    
    name = models.CharField(
        max_length=1000,
        blank=False,
        null=False)
    
    description = models.TextField(
        default='',
        blank=True)
    
    dependee_gene = models.ForeignKey(
        'self',
        related_name='dependent_genes',
        blank=True,
        null=True,
        help_text='''The gene this gene is dependent upon. This gene will only
            activate when the dependee gene has a certain value.''')
    
    dependee_value = models.CharField(
        max_length=1000,
        blank=True,
        null=True,
        help_text='''The value of the dependee gene that activates
            this gene.''')
    
    type = models.CharField(
        choices=c.GENE_TYPE_CHOICES,
        max_length=100,
        blank=False,
        null=False)
    
    values = models.TextField(
        blank=True,
        null=True,
        help_text=re.sub('[\s\n]+', ' ', '''A comma-delimited list of values the gene will be
            restricted to. Prefix with "source:package.module.attribute" to dynamically load values
            from a module.'''))
    
    default = models.CharField(max_length=1000, blank=True, null=True)
    
    min_value = models.CharField(
        max_length=100,
        blank=True, null=True,
        help_text=_('The minimum value this gene will be allowed to store.'))
    
    max_value = models.CharField(
        max_length=100,
        blank=True, null=True,
        help_text=_('The maximum value this gene will be allowed to store.'))
    
    min_value_observed = models.CharField(
        max_length=100,
        blank=True,
        null=True,
        editable=False,
        help_text=_('The minimum value observed for this gene.'))
    
    max_value_observed = models.CharField(
        max_length=100,
        blank=True,
        null=True,
        editable=False,
        help_text=_('The maximum value observed for this gene.'))
    
    coverage_ratio = models.FloatField(
        blank=True,
        null=True,
        editable=False,
        db_index=True,
        verbose_name=_(' CR'),
        help_text=_('Coverage ratio. The ratio of values being used by genotypes.'))
    
    exploration_priority = models.PositiveIntegerField(
        default=1,
        blank=False,
        null=False,
        db_index=True,
        verbose_name=_(' EP'),
        help_text=_('''Exploration priority.
            The importance by which all unique values will be
            tested. The higher, the more likely all values will be
            explored.'''))
    
    mutation_weight = models.FloatField(
        blank=True,
        null=True,
        db_index=True,
        editable=False,
        verbose_name=_(' MW'),
        help_text=_('''Mutation weight.
            The cached calculation
            (1-coverage_ratio)*exploration_priority.
            Designed to explore high-priority genes first.'''))
    
    max_increment = models.CharField(
        help_text='''When mutating an integer or float value, this is the
            maximum amount by which they'll be randomly adjusted.''',
        max_length=100,
        blank=True,
        null=True)
    
    class Meta:
        app_label = APP_LABEL
        verbose_name = _('gene')
        verbose_name_plural = _('genes')
        unique_together = (
            ('genome', 'name'),
        )
        ordering = (
            '-dependee_gene__id',
            'name',
        )
        
    def __unicode__(self):
        #return '<Gene:%s %s>' % (self.id, self.name)
        return '%i:%s' % (self.genome.id if self.genome else None, self.name)
    
    def natural_key(self):
        return (self.name,) + self.genome.natural_key()
    natural_key.dependencies = ['genome']
    
    def update_mutation_weight(self, auto_update=True):
        cr = self.coverage_ratio
        ep = self.exploration_priority
        self.mutation_weight = None
        if cr is not None and ep is not None:
            self.mutation_weight = (1 - cr)*ep
        if auto_update:
            type(self).objects.filter(id=self.id).update(mutation_weight=self.mutation_weight)
    
    def update_coverage(self, auto_update=True):
        possible_value_count = self.get_max_value_count()
        actual_values = set(self.gene_values.all().values_list('_value', flat=True).distinct())
        ratio = len(actual_values)/float(possible_value_count)
        self.coverage_ratio = ratio
        if auto_update:
            type(self).objects.filter(id=self.id).update(coverage_ratio=ratio)
        self.update_mutation_weight(auto_update=auto_update)
    
    def save(self, *args, **kwargs):
        if self.id:
            self.update_coverage(auto_update=False)
        super(Gene, self).save(*args, **kwargs)
    
    def get_max_value_count(self):
        """
        Returns the number of values this gene can assume.
        """
        if self.is_continuous():
            return 1e999999999999999
        elif self.values:
            return len(self.get_values_list())
        elif self.type == c.GENE_TYPE_BOOL:
            return 2
        elif self.type == c.GENE_TYPE_INT and self.min_value and self.max_value:
            return abs(int(self.min_value) - int(self.max_value))
        # Otherwise, we're implicitly infinite.
        return 1e999999999999999
    
    def is_discrete(self):
        if self.type == c.GENE_TYPE_FLOAT and self.values:
            return True
        discrete_types = (c.GENE_TYPE_INT, c.GENE_TYPE_BOOL, c.GENE_TYPE_STR)
        return self.type in discrete_types
    
    def is_continuous(self):
        return not self.is_discrete()
    
    def is_checkable(self):
        if self.type == c.GENE_TYPE_BOOL:
            return True
        elif self.type == c.GENE_TYPE_INT \
        and (self.values or (self.min_value and self.max_value)):
            return True
        elif self.type == c.GENE_TYPE_FLOAT and self.values:
            return True
        elif self.type == c.GENE_TYPE_STR and self.values:
            return True
        return False
    
    def get_random_value(self):
        values_list = self.get_values_list()
        if values_list:
            return random.choice(values_list)
        elif self.type == c.GENE_TYPE_INT:
            assert (self.min_value and self.max_value) or (self.default and self.max_increment)
            print 'min/max:',self.name,self.min_value,self.max_value
            if self.min_value and self.max_value:
                return random.randint(int(self.min_value), int(self.max_value))
            else:
                return random.randint(-int(self.max_increment), int(self.max_increment)) + int(self.default)
        elif self.type == c.GENE_TYPE_FLOAT:
            assert (self.min_value and self.max_value) or (self.default and self.max_increment)
            print 'min/max:',self.name,self.min_value,self.max_value
            if self.min_value and self.max_value:
                return random.uniform(float(self.min_value), float(self.max_value))
            else:
                return random.uniform(float(self.max_increment), float(self.max_increment)) + float(self.default)
        elif self.type == c.GENE_TYPE_BOOL:
            return random.choice([True,False])
        elif self.type == c.GENE_TYPE_STR:
            raise NotImplementedError, \
                'Cannot generate a random value for a string with no values.'
    
    def get_default(self):
        if self.default is None:
            return self.get_random_value()
        return str_to_type[self.type](self.default)
    
    def get_values_list(self):
        """
        Returns a list of allowable values for this gene.
        If gene value starts with "source:package.module.attribute", will dynamically lookup this list.
        """
        values = (self.values or '').strip()
        if not values:
            return
        if values.startswith('source:'):
            parts = values[7:].split('.')
            module_path = '.'.join(parts[:-1])
            var_name = parts[-1]
            module = importlib.import_module(module_path)
            lst = getattr(module, var_name)
        else:
            lst = values.replace('\n', ',').split(',')
        assert isinstance(lst, (tuple, list))
        return [str_to_type[self.type](_) for _ in lst]

class GenotypeManager(models.Manager):

    def get_by_natural_key(self, fingerprint, *args, **kwargs):
        genome = Genome.objects.get_by_natural_key(*args, **kwargs)
        return self.get(fingerprint=fingerprint, genome=genome)

    def stale(self):
        return self.filter(
            valid=True,
            fitness__isnull=True,
        )
    
    def valid(self):
        """
        A valid genotype is one that has been fully evaluated
        (e.g. fresh=True) and has received a fitness rating
        (e.g. fitness is not null).
        """
        return self.filter(
            valid=True,
            fresh=True,
            fitness__isnull=False,
        ).exclude(fitness=float('nan'))

class Genotype(models.Model):
    """
    A specific configuration for solving a problem.
    """
    
    objects = GenotypeManager()
    
    genome = models.ForeignKey(
        Genome,
        related_name='genotypes')
    
    species = models.ForeignKey(
        'Species',
        on_delete=models.SET_NULL,
        blank=True,
        null=True,
        editable=False,
        related_name='genotypes')
    
    fingerprint = models.CharField(
        max_length=700,
        db_column='fingerprint',
        blank=True,
        null=True,
        db_index=True,
        editable=False,
        help_text=_('''A unique hash calculated from the gene names and values
            used to detect duplicate genotypes.'''))
    
    fingerprint_fresh = models.BooleanField(
        default=False,
        editable=False,
        help_text=_('''If checked, indicates the fingerprint needs
            to be regenerated.'''))
    
    created = models.DateTimeField(auto_now_add=True, editable=False)
    
    generation = models.PositiveIntegerField(
        default=1,
        blank=False,
        null=False,
        editable=False,
        help_text=_('''A count used to denote inheritance.
            A mutated genotype will inherit the parent\'s count plus one.
            A hybrid genotype will inherit the largest parent\'s count plus
            one.'''))
    
    fitness = models.FloatField(blank=True, null=True, editable=False)
    
    fitness_evaluation_datetime_start = models.DateTimeField(
        blank=True,
        null=True,
        editable=False,
        verbose_name=_('evaluation start time'))
    
    fitness_evaluation_datetime = models.DateTimeField(
        blank=True,
        null=True,
        editable=False,
        verbose_name=_('evaluation stop time'))
    
    mean_evaluation_seconds = models.PositiveIntegerField(blank=True, null=True, editable=False)
    
    total_evaluation_seconds = models.PositiveIntegerField(blank=True, null=True, editable=False)
    
    mean_absolute_error = models.FloatField(
        blank=True,
        null=True,
        db_index=True,
        editable=False,
        help_text=_('''The mean-absolute-error measure recorded during
            fitness evaluation.'''))
    
    gene_count = models.PositiveIntegerField(
        verbose_name='genes',
        blank=True,
        null=True,
        editable=False)
    
#    preserve = models.BooleanField(
#        default=False,
#        help_text=_('''If checked, this genotype will not be deleted
#            even if it becomes unfit.'''))
    
    #evaluation_epoche = 
    
    evaluating = models.BooleanField(
        default=False,
        db_index=True,
        help_text=_('''If checked, indicates this genotype is currently having
            its fitness evaluated.'''))
    
    evaluating_pid = models.IntegerField(
        blank=True,
        null=True,
        editable=False,
        help_text=_('The PID of the process evaluating this genotype.'))
    
    fresh = models.BooleanField(
        default=False,
        editable=False,
        db_index=True,
        help_text=_('If true, indicates this predictor has been evaluated.'))
    
    fresh_datetime = models.DateTimeField(
        blank=True,
        null=True,
        help_text=_('The timestamp of when this record was made fresh.'))
    
    valid = models.BooleanField(
        default=True,
        #editable=False,
        db_index=True,
        help_text=_('''If true, indicates this genotype was evaluted without
            any fatal errors. Note, other errors may have occurred as reported
            by the success ratio.'''))
    
    total_parts = models.PositiveIntegerField(
        blank=True,
        null=True,
        editable=False,
        help_text=_('Total number of sub-evaluations to run.'))
    
    success_parts = models.PositiveIntegerField(
        blank=True,
        null=True,
        editable=False,
        help_text=_('Total number of sub-evaluations successfully run.'))
    
    ontime_parts = models.PositiveIntegerField(
        blank=True,
        null=True,
        editable=False,
        help_text=_('Total number of sub-evaluations that ran ontime.'))
    
    success_ratio = models.FloatField(
        blank=True,
        null=True,
        db_index=True,
        editable=False)
    
    ontime_ratio = models.FloatField(
        blank=True,
        null=True,
        db_index=True,
        editable=False)
    
    error = models.TextField(
        blank=True,
        null=True,
        help_text=_('Any error message received during evaluation.'))
    
    class Meta:
        app_label = APP_LABEL
        verbose_name = _('genotype')
        verbose_name_plural = _('genotypes')
        unique_together = (
            ('genome', 'fingerprint'),
        )
        ordering = (
            '-fitness',
        )
        index_together = (
            (
                'valid',
                'fresh',
                'fitness',
            ),
            ('genome', 'fresh'),
        )
        pass
    
    def __unicode__(self):
        #return '<%s:%i>' % (type(self).__name__, self.id)#(self.fingerprint or u'') or unicode(self.id)
        return unicode(self.id)
    
    def natural_key(self):
        return (self.fingerprint,) + self.genome.natural_key()
    natural_key.dependencies = ['genome']

    def get_fingerprint(self):
        return obj_to_hash(tuple(sorted(
            (gene.gene.name, gene.value)
            for gene in self.genes.all())
        ))
    
    @property
    def status(self):
        if self.evaluating:
            if self.evaluating_pid:
                return 'evaluating by %s' % (self.evaluating_pid,)
            else:
                return 'evaluating'
#        elif self.fitness_evaluation_datetime_start and not self.fitness_evaluation_datetime:
#            return 'evaluating'
#        elif not self.fresh and self.success_ratio is not None:
#            return 'evaluating'
        return ''
    
    @classmethod
    def freshen_fingerprints(cls):
        """
        Recalculates the fingerprint for all genotypes with a stale
        fingerprint.
        """
        q = cls.objects.filter(fingerprint_fresh=False)
        total = q.count()
        if total:
            print 'Freshening %i fingerprints.' % (total,)
            i = 0
            for gt in q.iterator():
                i += 1
                #TODO:handle fingerprint conflicts?
                print '\rFreshening fingerprint %i of %i...' % (i, total),
                sys.stdout.flush()
                gt.save(check_fingerprint=True)
    
    @classmethod
    def mark_stale(cls, genotype_ids, save=True):
        """
        Updates the given genotypes as needing the fitness and fingerprint
        re-evaluated.
        """
        if not genotype_ids:
            return
        q = cls.objects.filter(id__in=genotype_ids)
        for genotype in q.iterator():
            #TODO:wrap in error handling for fingerprint conflict?
            genotype.fresh = False
            genotype.fingerprint_fresh = False
            if save:
                genotype.save(check_fingerprint=True)
    
    def delete_illegal_genes(self, save=True):
        """
        Deletes genes that aren't allowed to exist according to genome rules.
        """
        q = self.illegal_gene_values.all()
        total = q.count()
        if total:
            print 'Deleting %i illegal gene values.' % (total,)
            for illegal in q.iterator():
                illegal.gene_value.delete()
    #        for gene in list(self.genes.all()):
    #            if not gene.is_legal():
    #                print 'Deleting illegal genotype gene %s...' % (gene,)
    #                gene.delete()
            self.fingerprint_fresh = False
            if save:
                self.save(check_fingerprint=False)
    
    def clean(self, check_fingerprint=True, *args, **kwargs):
        """
        Called to validate fields before saving.
        Override this to implement your own model validation
        for both inside and outside of admin. 
        """
        try:
            if self.id and check_fingerprint:
                fingerprint = self.get_fingerprint()
                q = self.genome.genotypes.filter(fingerprint=fingerprint).exclude(id=self.id)
                if q.count():
                    url = get_admin_change_url(q[0])
                    raise ValidationError(mark_safe(('Fingerprint conflicts with '
                        '<a href="%s" target="_blank">genotype %i</a>, indicating '
                        'one of these genotypes is a duplicate of the other. '
                        'Either delete one of these genotypes or change their '
                        'gene values so that they differ.') % (url, q[0].id,)))
            
            super(Genotype, self).clean(*args, **kwargs)
        except Exception, e:
#            print '!'*80
#            print e
            raise

    def full_clean(self, check_fingerprint=True, *args, **kwargs):
        return self.clean(check_fingerprint=check_fingerprint, *args, **kwargs)
    
    def update_status(self, success_parts, ontime_parts, total_parts):
        
        self.total_parts = total_parts
        self.success_parts = success_parts
        self.ontime_parts = ontime_parts
        
        self.success_ratio = success_parts/float(total_parts) if total_parts else None
        self.ontime_ratio = ontime_parts/float(total_parts) if total_parts else None
        
        type(self).objects.filter(id=self.id).update(
            total_parts=self.total_parts,
            success_parts=self.success_parts,
            ontime_parts=self.ontime_parts,
            success_ratio=self.success_ratio,
            ontime_ratio=self.ontime_ratio,
        )
    
    def save(self, check_fingerprint=True, using=None, *args, **kwargs):
        
        if self.id:
            
            self.gene_count = self.genes.all().count()
            
            if check_fingerprint and not self.fingerprint_fresh:
                self.fingerprint = self.get_fingerprint()
                self.fingerprint_fresh = True
                
        self.full_clean(check_fingerprint=check_fingerprint)
        
        self.total_evaluation_seconds = 0
        if self.fitness_evaluation_datetime and self.fitness_evaluation_datetime_start:
            self.total_evaluation_seconds = (self.fitness_evaluation_datetime - self.fitness_evaluation_datetime_start).seconds
        
        super(Genotype, self).save(using=using, *args, **kwargs)
        #print 'genotype.save().fresh:',self.fresh
        self.genome.save()
        
    def getattr(self, name):
        return self.genes.filter(gene__name=name)[0].value
        
    def as_dict(self):
        return dict((gene.gene.name, gene.value) for gene in self.genes.all())
    
    def refresh_fitness(self):
        """
        Refreshes fitness calculation.
        Only necessary to do if the fitness calculation code has been changed
        and you want to show updated fitness values without re-evaluating
        the genotypes.
        """
        self.genome.calculate_fitness_function(self)
        type(self).objects.filter(id=self.id).update(fitness=self.fitness)
    
    def reset(self):
        """
        Modifies this genotype according to the genome's reset backend.
        """
        self.genome.reset_function(self)
        type(self).objects.filter(id=self.id).update(
            valid=True,
            fresh=False,
            error=None,
            total_parts=None,
            success_parts=None,
            ontime_parts=None,
            success_ratio=None,
            ontime_ratio=None,
            total_evaluation_seconds=None,
            mean_evaluation_seconds=None,
            mean_absolute_error=None,
            fitness_evaluation_datetime_start=None,
            fitness_evaluation_datetime=None,
            evaluating=False,
            evaluating_pid=None,
        )

class GenotypeGeneIllegal(BaseModel):
    """
    Wraps the efficient SQL view that detects all invalid gene values.
    """
    
    gene_value = models.ForeignKey(
        'GenotypeGene',
        db_column='illegal_genotypegene_id',
        primary_key=True,
        blank=False,
        null=False,
        on_delete=models.DO_NOTHING,
        editable=False)
    
    illegal_gene_name = models.CharField(
        max_length=1000,
        editable=False)
        
    genotype = models.ForeignKey(
        'Genotype',
        db_column='illegal_genotype_id',
        related_name='illegal_gene_values',
        blank=False,
        null=False,
        editable=False,
        on_delete=models.DO_NOTHING)
        
    dependee_name = models.CharField(
        max_length=1000,
        editable=False)
    
    dependee_value = models.CharField(
        max_length=1000,
        editable=False)
    
    illegal_value = models.CharField(
        editable=False,
        max_length=1000)
    
    class Meta:
        managed = False
        db_table = 'django_analyze_genotypegeneillegal'

class GenotypeGeneMissing(BaseModel):
    """
    Wraps the efficient SQL view that detects all missing gene values.
    """
    
    gene = models.ForeignKey(
        'Gene',
        db_column='gene_id',
        primary_key=True,
        blank=False,
        null=False,
        editable=False,
        on_delete=models.DO_NOTHING)
        
    genotype = models.ForeignKey(
        'Genotype',
        db_column='genotype_id',
        related_name='missing_gene_values',
        blank=False,
        null=False,
        editable=False,
        on_delete=models.DO_NOTHING)
        
    gene_name = models.CharField(
        max_length=1000,
        editable=False)
    
    dependee_gene = models.ForeignKey(
        'Gene',
        db_column='dependee_gene_id',
        related_name='missing_dependents',
        primary_key=True,
        blank=False,
        null=False,
        editable=False,
        on_delete=models.DO_NOTHING)
    
    default = models.CharField(
        max_length=1000,
        editable=False)
    
    class Meta:
        managed = False
        db_table = 'django_analyze_genotypegenemissing'
        
    def create(self):
        GenotypeGene.objects.create(
            genotype=self.genotype,
            gene=self.gene,
            _value=self.default,
        )
#        Genotype.objects\
#            .filter(id=self.genotype_id)\
#            .update(fingerprint_fresh=False)

class GenotypeGeneManager(models.Manager):
    
    def get_by_natural_key(self, fingerprint, genome_name, gene_name, genome_name2):
        genotype = Genotype.objects.get_by_natural_key(fingerprint, genome_name)
        gene = Gene.objects.get_by_natural_key(gene_name, genome_name2)
        return self.get(genotype=genotype, gene=gene)
    
class GenotypeGene(BaseModel):
    
    genotype = models.ForeignKey(
        Genotype,
        related_name='genes')
    
    gene = models.ForeignKey(
        Gene,
        related_name='gene_values')

    _value = models.CharField(
        max_length=1000,
        db_column='value',
        verbose_name='value',
        blank=False,
        null=False)
    
    class Meta:
        app_label = APP_LABEL
        ordering = (
            'gene__name',
        )
        unique_together = (
            ('genotype', 'gene'),
        )
        verbose_name = _('genotype gene')
        verbose_name_plural = _('genotype genes')
    
    def natural_key(self):
        return self.genotype.natural_key() + self.gene.natural_key()
    natural_key.dependencies = ['genotype', 'gene']
    
    def __unicode__(self):
        return '<GenotypeGene:%s %s=%s>' % (self.id, self.gene.name, self._value)
    
    @property
    def value(self):
        if self.gene.type == c.GENE_TYPE_INT:
            return int(self._value)
        elif self.gene.type == c.GENE_TYPE_FLOAT:
            return float(self._value)
        elif self.gene.type == c.GENE_TYPE_BOOL:
            if self._value in ('True', '1'):
                return True
            return False
        elif self.gene.type == c.GENE_TYPE_STR:
            return self._value
        return self._value
    
    @value.setter
    def value(self, v):
        if not isinstance(v, c.GENE_TYPES):
            raise NotImplementedError, 'Unsupported type: %s' % type(v)
        self._value = str(v)
    
    def is_legal(self):
        """
        Returns true if this gene value is allowed to exist in this genotype
        based on gene dependency rules.
        Returns false otherwise, implying it should be deleted.
        """
        if not self.gene.dependee_gene:
            # We're not dependent on any other gene, so we're implicitly
            # allowed to exist.
            return True
        q = self.genotype.genes.filter(gene=self.gene.dependee_gene)
        if not q.count():
            # We're dependent on a gene that doesn't exist in this genotype,
            # so we shouldn't exist either.
            return False
        elif q[0]._value != self.gene.dependee_value:
            # We're dependent on an existing gene, but its value differs from
            # the value we require so we shouldn't exist.
            return False
        return True
    is_legal.boolean = True
    
    def get_random_value(self):
        if self.gene.type == c.GENE_TYPE_BOOL:
            return not self.value
        elif self.gene.type == c.GENE_TYPE_INT:
            if self.gene.values:
                return random.choice(self.gene.get_values_list())
            else:
                inc = int(float(self.gene.max_increment))
                max_value = float(self.gene.max_value) if self.gene.max_value else None
                min_value = float(self.gene.min_value) if self.gene.min_value else None
                new_value = random.randint(-inc, inc) + self.value
                if max_value is not None:
                    new_value = min(new_value, max_value)
                if min_value is not None:
                    new_value = max(new_value, min_value)
                return new_value
        elif self.gene.type == c.GENE_TYPE_FLOAT:
            if self.gene.values:
                return random.choice(self.gene.get_values_list())
            else:
                inc = float(self.gene.max_increment)
                max_value = float(self.gene.max_value) if self.gene.max_value else None
                min_value = float(self.gene.min_value) if self.gene.min_value else None
                new_value = random.uniform(-inc, inc) + self.value
                if max_value is not None:
                    new_value = min(new_value, max_value)
                if min_value is not None:
                    new_value = max(new_value, min_value)
                return new_value
        elif self.gene.type == c.GENE_TYPE_STR:
            if self.gene.values:
                return random.choice(self.gene.get_values_list())
            else:
                #TODO:any other way to randomize a string without discrete values?
                return self.value
        else:
            raise NotImplementedError
        
    def clean(self, *args, **kwargs):
        """
        Called to validate fields before saving.
        Override this to implement your own model validation
        for both inside and outside of admin. 
        """
        
        try:
            #print 'self._value:',self._value
            value = str_to_type[self.gene.type](self._value)
            if self.gene.values:
                values = self.gene.get_values_list()
                if value not in values:
                    raise ValidationError({'_value': 'Value must be one of %s, not %s' % (', '.join(map(str, values)), repr(value))})
        except ValueError:
            raise ValidationError({'_value': 'Value must be of type %s.' % self.gene.type})
        
        super(GenotypeGene, self).clean(*args, **kwargs)
        
    def full_clean(self, *args, **kwargs):
        return self.clean(*args, **kwargs)
        
    def save(self, using=None, *args, **kwargs):
        self.full_clean()
        super(GenotypeGene, self).save(using=using, *args, **kwargs)
        self.gene.update_coverage(auto_update=True)
#        self.genotype.fingerprint_fresh = False
#        self.genotype.fresh = False
#        self.genotype.save(check_fingerprint=False)
        
    @staticmethod
    def post_save(sender, instance, *args, **kwargs):
        self = instance
        Genotype.objects\
            .filter(id=self.genotype_id)\
            .update(fresh=False, fingerprint_fresh=False)
#        self.genotype.fresh = False
#        self.genotype.fingerprint_fresh = False
#        self.genotype.save(check_fingerprint=False)
#        print '?'*80
#        print 'self.genotype.fresh:',self.genotype.fresh
        
    @staticmethod
    def post_delete(sender, instance, *args, **kwargs):
        self = instance
        try:
            # If the genotype gene was deleted but the genotype still exists,
            # then mark the genotype as stale requiring re-evaluation.
#            genotype = Genotype.objects.get(id=self.genotype.id)
#            genotype.fresh = False
#            genotype.fingerprint_fresh = False
#            genotype.save(check_fingerprint=False)
            Genotype.objects\
                .filter(id=self.genotype_id)\
                .update(fresh=False, fingerprint_fresh=False)
        except Genotype.DoesNotExist:
            # If the genotype gene was deleted because the genotype was
            # deleted, then do nothing.
            pass

#TODO:fix signals not being sent after inline parent saved
from django.db.models import signals
signals.post_save.connect(GenotypeGene.post_save, sender=GenotypeGene)
signals.post_delete.connect(GenotypeGene.post_delete, sender=GenotypeGene)
