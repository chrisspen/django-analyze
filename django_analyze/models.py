
from base64 import b64encode, b64decode
from collections import defaultdict
from datetime import timedelta
from functools import partial
from threading import Thread
from StringIO import StringIO
import gc
import inspect
import importlib
import os
import random
import re
import sys
import tempfile
import threading
import time
import traceback
import urllib

from multiprocessing import Process, Queue, Lock, cpu_count, Pool

import psutil

from picklefield.fields import PickledObjectField

import django
from django import forms
from django.conf import settings
from django.core.exceptions import ValidationError
from django.db import models, DatabaseError, connection, reset_queries, transaction
from django.db.transaction import commit_on_success
from django.db.models import Avg, Sum, Count, Max, Min, Q, F
from django.db.utils import IntegrityError
from django.utils import timezone
from django.utils.safestring import mark_safe
from django.utils.translation import ugettext, ugettext_lazy as _
from django.db.models import signals
from django.dispatch.dispatcher import Signal
from django.core.signals import request_finished
from django.dispatch import receiver

from joblib import Parallel, delayed

try:
    from django_materialized_views.models import MaterializedView
except ImportError:
    class MaterializedView(object):
        pass

try:
    from chroniker.models import Job
except ImportError:
    class Job(object):
        @classmethod
        def update_progress(cls, total_parts_complete, total_parts):
            pass

from admin_steroids.utils import StringWithTitle
APP_LABEL = StringWithTitle('django_analyze', 'Analyze')

import constants as c
import utils

from admin_steroids.utils import get_admin_change_url

def get_fingerprint(d):
    return obj_to_hash(tuple(sorted(
        (k, v)
        for k, v in d.items()
    )))

class FingerprintConflictError(ValidationError):
    pass

def lookup_genome_genotypes(id, only_production=False):
    """
    Converts a genome_id to a list of exposed genotypes.
    """
    try:
        if isinstance(id, basestring) and len(id.split(':')) == 2:
            genome_id, genotype_id = id.split(':')
            return Genotype.objects.only('id').filter(
                genome__id=int(genome_id), id=int(genotype_id))
        else:
            if only_production:
                return Genotype.objects.only('id').filter(
                    Q(genome__id=int(id)) & Q(production_genomes__id__isnull=False)
                )
            else:
                return Genotype.objects.only('id').filter(
                    Q(genome__id=int(id)) & Q(Q(production_genomes__id__isnull=False)|Q(export=True))
                )
    except ValueError:
        pass
    except TypeError:
        pass
    except Genotype.DoesNotExist:
        pass
    except Genome.DoesNotExist:
        pass

str_to_type = {
    c.GENE_TYPE_INT:int,
    c.GENE_TYPE_FLOAT:float,
    c.GENE_TYPE_BOOL:(lambda v: True if v in (True, 'True', 1, '1') else False),
    c.GENE_TYPE_STR:(lambda v: str(v)),
    c.GENE_TYPE_GENOME:lookup_genome_genotypes,
}

_global_fingerprint_check = [True]

def enable_fingerprint_check():
    _global_fingerprint_check[0] = True

def disable_fingerprint_check():
    _global_fingerprint_check[0] = False

def do_fingerprint_check():
    return _global_fingerprint_check[0]

_global_validation_check = [True]

def enable_validation_check():
    _global_validation_check[0] = True

def disable_validation_check():
    _global_validation_check[0] = False

def do_validation_check():
    return _global_validation_check[0]

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
        #print args, kwargs
        kwargs = kwargs.copy()
        if 'exclude' in kwargs:
            del kwargs['exclude']
        if 'validate_unique' in kwargs:
            del kwargs['validate_unique']
        super(BaseModel, self).clean(*args, **kwargs)
        
    def full_clean(self, *args, **kwargs):
        return self.clean(*args, **kwargs)
    
    def save(self, *args, **kwargs):
        self.full_clean()
        super(BaseModel, self).save(*args, **kwargs)

class LabelManager(models.Manager):
    
    def get_by_natural_key(self, name):
        return self.get_or_create(name=name)[0]

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

class PredictorManager(models.Manager):
    
    def usable(self):
        return self.filter(fresh=True, valid=True)

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
        editable=False,
        help_text='The number of CPU seconds the algorithm spent training.')
    
    training_ontime = models.BooleanField(
        default=True,
        blank=False,
        null=False,
        editable=False,
        help_text='''If false, indicates this predicator failed to train within
            the allotted time, and so it is not suitable for use.''',
    )
    
    epoche_trained = models.PositiveIntegerField(
        default=0,
        blank=False,
        null=False,
        editable=False,
        db_index=True,
        help_text=_('The epoche when this predictor was trained.'))
    
    testing_r2 = models.FloatField(
        blank=True,
        null=True,
        db_index=True,
        editable=False,
        help_text='''The coefficient of determination recorded during testing.
            A value close to 1 indicates the model was able to perfectly
            predict values.''')
    
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
        db_index=True,
        help_text='The accuracy of the classifier predicting its training input.')
    
    testing_accuracy = models.FloatField(
        blank=True,
        null=True,
        editable=False,
        db_index=True,
        help_text='The accuracy of the classifier predicting its testing input.')
    
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
        editable=False,
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
    
    training_threshold_accuracy = models.FloatField(
        blank=True,
        null=True,
        editable=False)
    
    training_threshold_coverage = models.FloatField(
        blank=True,
        null=True,
        editable=False,
        help_text=('''The ratio of training samples that fell within
            the certainty threshold. 1.0 means everything covered.
            0.0 means nothing covered. Higher is better.'''))
    
    testing_threshold_accuracy = models.FloatField(
        blank=True,
        null=True,
        editable=False)
    
    testing_threshold_coverage = models.FloatField(
        blank=True,
        null=True,
        editable=False,
        help_text=('''The ratio of testing samples that fell within
            the certainty threshold. 1.0 means everything covered.
            0.0 means nothing covered. Higher is better.'''))
    
    extra_evaluation_seconds = models.FloatField(
        blank=True,
        null=True,
        help_text=_('''Additional seconds that should be added to the
            evaluation time, due to caching of shared resources.'''))
    
    #_predictor = models.TextField(
    _predictor = PickledObjectField(
        blank=True,
        null=True,
        compress=True,
        editable=False,
        db_column='classifier',
        help_text=_('The serialized model for use in production.'))
    
    @property
    def predictor(self):
        return self._predictor
#        if data:
#            _, fn = tempfile.mkstemp()
#            fout = open(fn, 'wb')
#            fout.write(b64decode(data))
#            fout.close()
#            obj = joblib.load(fn)
#            os.remove(fn)
#            return obj
#        else:
#            return
        
    @predictor.setter
    def predictor(self, v):
        self._predictor = v
        #TODO:buggy? doesn't return a simple single-file that we can store!?
#        if v:
#            _, fn = tempfile.mkstemp()
#            joblib.dump(v, fn, compress=9)
#            fin = open(fn, 'rb')
#            self._predictor = b64encode(fin.read())
#            fin.close()
#            os.remove(fn)
#        else:
#            self._predictor = None
    
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
        editable=False,
        help_text='The future value predicted by the algorithm.')
    
    expected_value = models.FloatField(
        blank=True,
        null=True,
        db_index=True,
        editable=False,
        help_text='When back-testing, this stores the true value, what we hope predicted_value will be.')
    
    reference_value = models.FloatField(
        blank=True,
        null=True,
        db_index=True,
        editable=False,
        help_text='The current value before the change event that caused the predicted value.')
    
    reference_difference = models.FloatField(
        blank=True,
        null=True,
        db_index=True,
        editable=False,
        help_text='The predicted future value minus a reference value.')
    
    def percent_change(self):
        n = self.predicted_value
        o = self.reference_value
        if n is None or o is None:
            return
        return (n - o)/float(o)
    
    predicted_prob = models.FloatField(
        blank=True,
        null=True,
        db_index=True,
        editable=False,
        help_text='The future value probability by the algorithm.')
    
    predicted_score = models.FloatField(
        blank=True,
        null=True,
        db_index=True,
        editable=False,
#        verbose_name=_('score'),
        help_text='How well this prediction compares to others.')
    
    evaluating = models.BooleanField(
        default=False,
        db_index=True,
        help_text=_('''If checked, indicates this predictor is currently being trained or tested.'''))
    
    fresh = models.NullBooleanField(
        editable=False,
        db_index=True,
        help_text='If true, indicates this predictor is ready to classify.')

    test = models.NullBooleanField(
        editable=False,
        default=True,
        db_index=True,
        help_text=_('''If true, indicates this predictor is for testing.
            If false, indicates this predictor is for production use.'''))
    
    valid = models.NullBooleanField(
        editable=False,
        db_index=True,
        help_text='If true, indicates this was trained without error.')

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
        raise NotImplementedError
        
    def predict(self):
        raise NotImplementedError

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

def get_evaluator_name(cls):
    if hasattr(cls, '_meta'):
        # Use Django's logic module name if one given.
        return cls._meta.app_label + '.' + cls.__name__
    return cls.__module__ + '.' + cls.__name__

def register_evaluator(cls):
    assert hasattr(cls, 'evaluate_genotype'), \
        'Class %s does not implement `evaluate_genotype`.' % (cls.__name__,)
    assert hasattr(cls, 'reset_genotype'), \
        'Class %s does not implement `reset_genotype`.' % (cls.__name__,)
    assert hasattr(cls, 'calculate_genotype_fitness'), \
        'Class %s does not implement `calculate_genotype_fitness`.' % (cls.__name__,)
    #name = get_method_name(cls)
    name = get_evaluator_name(cls)
    _evaluators[name] = cls
    return name

def register_modeladmin_extender(func):
    name = get_method_name(func)
    _modeladmin_extenders[name] = func

def get_evaluators():
    for name in sorted(_evaluators.iterkeys()):
        yield (name, name)

#def get_extenders():
#    for name in sorted(_modeladmin_extenders.iterkeys()):
#        yield (name, name)

class SpeciesManager(models.Manager):

    def get_by_natural_key(self, index, *args, **kwargs):
        genome = Genome.objects.get_by_natural_key(*args, **kwargs)
        return self.get_or_create(index=index, genome=genome)[0]
    
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
    
    natural_key_fields = ('index', 'genome')
    
    def natural_key(self):
        return (self.index,) + self.genome.natural_key()
    natural_key.dependencies = ['django_analyze.Genome']
    
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

def did_genotype_change(gt1, gt2):
    """
    Returns true if the given genotype changed in a manner requiring a Genome
    to re-evaluate itself.
    """
    if gt1 and gt2:
        if gt1 != gt2:
            return True
        elif not gt1.fresh and gt2.fresh:
            return True
    return False

# Sent when a genome changes its production genotype of the production genotype
# switches from stale to fresh.
production_genotype_changed = Signal(providing_args=["genome"])
#production_genotype_changed.send(sender=self, genome=genome)

@receiver(production_genotype_changed)
def handle_production_genotype_change(sender, genome, **kwargs):
    """
    Called when the production genotype in the given genome changes
    in such a way that all dependent production genotypes should be marked
    as stale so they too can be re-evaluated.
    """
    #print 'handling changed genome:',genome
    
    # Lookup all production genotpyes in genomes that depend on the given
    # changed genome.
    q = GenotypeGene.objects.filter(
        gene__type=c.GENE_TYPE_GENOME,
        _value=str(genome.id)
    ).values_list('genotype__genome', flat=True)
    #print 'dependent genomes:',q
    
    # Mark those genotypes as stale so they'll be re-tested with whatever
    # new information is contained in the dependent genome.
    # Note, this will not cause a cascade because a positive change only
    # occurs when a genotype transitions from stale to fresh.
    Genotype.objects.filter(genome__id__in=q)\
        .update(fresh=False, production_fresh=False)

class GenomeBackendMixin(object):
    """
    Defines standard methods that must or may be implemented
    by a genome backend.
    """
    
    @classmethod
    def reset_genotype(cls, genotype):
        """
        Called immediately before the genotype is evaluated
        in each epoche.
        
        Implementation optional.
        """
        
    @classmethod
    def pre_delete_genotype(cls, genotype, fout=None):
        """
        Called before the genotype is deleted.
        This allows us time to incrementally delete any linked data
        records, which might otherwise consume all system memory
        if we tried to delete them in a single large transaction.
        
        Implementation optional.
        """

    @classmethod
    def mark_stale_genotype(cls, genotype_qs):
        """
        Bulk updates any data associated with the given genotype queryset
        as stale.
        
        Implementation optional.
        """
    
    @classmethod
    def is_production_ready_genotype(cls, genotype):
        """
        Returns true if the given genotype can be used for production
        prediction tasks.
        
        Implementation REQUIRED.
        """
        raise NotImplementedError
    
    @classmethod
    def evaluate_genotype(cls, genotype, test=True, force_reset=False, progress=None):
        """
        Calculates the given genotype's fitness.
        
        Implementation REQUIRED.
        """
        raise NotImplementedError

class GenomeManager(models.Manager):
    
    def get_by_natural_key(self, name):
        #print 'GenomeManager:',name
        return self.get_or_create(name=name)[0]

def ret_to_ready(v):
    if isinstance(v, (tuple, list)):
        return v[0]
    return bool(v)

class Genome(BaseModel):
    """
    All possible parameters of a problem domain.
    """
    
    objects = GenomeManager()
    
    natural_key_fields = ('name',)
    
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
    
    evolving = models.BooleanField(
        default=False,
        db_index=True,
        help_text=_('If true, indicates this genome is currently being evaluated.'))
    
    evolution_start_datetime = models.DateTimeField(
        blank=True,
        null=True,
        editable=False,
        help_text=_('The date and time when the most recent evolution epoche began.'))
    
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
        default=1,
        blank=False,
        null=False,
        help_text=_('''The number of epoches thus far evaluated. This number
            will also be used to seed an epoche-specific random number
            generator for each genotype.'''))
    
    _epoche = models.ForeignKey(
        'Epoche',
        on_delete=models.SET_NULL,
        blank=True,
        null=True,
        related_name='current_genome')
    
    @property
    def current_epoche(self):
        return Epoche.objects.get_or_create(genome=self, index=self.epoche)[0]
    
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
    
    
    def total_genotype_count(self):
        return self.genotypes.all().count()
    total_genotype_count.short_description = 'total genotypes'
    
    def pending_genotype_count(self):
        return self.pending_genotypes.count()
    pending_genotype_count.short_description = 'pending genotypes'
        
    def evaluating_genotype_count(self):
        return self.evaluating_genotypes.count()
    evaluating_genotype_count.short_description = 'evaluating genotypes'
    
    def complete_genotype_count(self):
        return self.complete_genotypes.count()
    complete_genotype_count.short_description = 'complete genotypes'
    
    def invalid_genotype_count(self):
        return self.invalid_genotypes.count()
    invalid_genotype_count.short_description = 'invalid genotypes'
    
    def stalled(self):
        #TODO:change to look at epoche model?
        #e.g. if last N epoches have same oldest_epoche_of_creation, then
        #that implies progress has stalled
        #e.g. if oldest_epoche_of_creation has changed, but mean_fitness
        # is the same, that implies no convergence and that the evaluation
        # method is not comprehensive
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
    
    elite_ratio = models.FloatField(
        default=0.1,
        blank=False,
        null=False,
        help_text=_('''The ratio of the best genotypes to keep between epoches.'''))
    
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
        verbose_name=_('Automatically select production genotype'),
        help_text=_('''If checked, the `production_genotype` will automatically
            be set to the genotype with the best fitness.'''))
    
    production_genotype = models.ForeignKey(
        'Genotype',
        related_name='production_genomes',
        on_delete=models.SET_NULL,
        blank=True,
        null=True)
    
    production_at_best = models.BooleanField(
        default=False,
        db_index=True,
        editable=False,
        help_text=_('''If true, indicates the production genotype is the most
            fit of all known genotypes.'''))
    
    production_evaluation_timeout = models.PositiveIntegerField(
        default=0,
        blank=True,
        null=True,
        help_text=_('''If non-zero, the number of seconds the production
            genotype will be evaluated for production use.'''))
    
    evaluating_part = models.PositiveIntegerField(
        default=0,
        #help_text=_('?'),
    )
    
    ratio_evaluated = models.FloatField(
        blank=True,
        null=True,
        #help_text=_('?'),
    )
    
    error_report = models.TextField(
        blank=True,
        null=True,
        editable=False,
        help_text=_('''A summary of unhandled exceptions encountered while
            evaluating genotypes.'''))
    
    version = models.PositiveIntegerField(
        default=1,
        editable=False,
        help_text=_('''A unique reference number for the current allowable
            genes. This is automatically incremented whenever genes
            are modified.'''),
    )
    
    max_memory_usage_ratio = models.FloatField(
        default=0.5,
        blank=False,
        null=False,
        help_text=_('''The maximum ratio of available memory used to estimate
            how many processes to run. 1.0=all, 0.5=half, etc.
            Note, this requires at least one genotype having been successfully
            evaluated and had its memory usage recorded.'''))
    
    class Meta:
        app_label = APP_LABEL
        verbose_name = _('genome')
        verbose_name_plural = _('genomes')
    
    def __unicode__(self):
        return self.name
    
    def natural_key(self):
        return (self.name,)
    natural_key.dependencies = []
    
    @property
    def mean_max_memory_usage(self):
        """
        The average maximum amount of memory used by all genotypes.
        """
        max_memory_usage__avg = self.genotypes.filter(
            max_memory_usage__isnull=False
        ).aggregate(Avg('max_memory_usage'))['max_memory_usage__avg']
        return max_memory_usage__avg
    
    def get_evolve_processes(self):
        """
        Returns the maximum number of processes most appropriate to use
        given memory usage constraints and the number of cores available.
        """
        total_processes = cpu_count()
        mean_max_memory_usage = self.mean_max_memory_usage # bytes
        if mean_max_memory_usage is None:
            return 1 #total_processes
        max_memory_usage_ratio = self.max_memory_usage_ratio
#        print'mean_max_memory_usage:',mean_max_memory_usage/1024./1024./1024.
        free_memory = psutil.virtual_memory().available # bytes
        free_memory *= max_memory_usage_ratio
#        print'free_memory:',free_memory/1024./1024./1024.
        max_processes_by_memory = int(free_memory/float(mean_max_memory_usage))
        return min(total_processes, max_processes_by_memory)
    
    def generate_error_report(self, force=False, save=True):
        if not force and self.error_report:
            return
        error_pattern = re.compile('[a-zA-Z0-9]+: [^\n]+')
        error_counts = defaultdict(int)
        for gt in self.genotypes.filter(error__isnull=False).exclude(error='').only('id').iterator():
            error = (gt.error or '').strip()
            matches = error_pattern.findall(error)
            if matches:
                error_counts[matches[-1]] += 1
        if error_counts:
            report = ['<table><tr><th>Count</th><th>Error</th></tr>']
            for error, count in sorted(error_counts.iteritems(), key=lambda o:o[1], reverse=True):
                url_params = urllib.urlencode(dict(genome=self.id, error__icontains=error))
                kwargs = dict(
                    count=count,
                    error=error,
                    url='/admin/django_analyze/genotype/?%s' % url_params)
                report.append('<tr><td>{count}</td><td><a href="{url}" target="_blank">{error}</a></td></tr>'.format(**kwargs))
            report.append('</table>')
            self.error_report = ''.join(report)
        else:
            self.error_report = 'No errors found.'
        if save:
            self.save()
    
    def create_blank_genotype(self, generation=1):
        """
        Creates a new genotype without any genes.
        """
        new_genotype = Genotype(genome=self, generation=generation)
        new_genotype.save(check_fingerprint=False)
        return new_genotype
    
    @property
    def complete_genotypes(self):
        return self.genotypes.filter(fitness__isnull=False, fresh=True, valid=True)
    
    @property
    def invalid_genotypes(self):
        return self.genotypes.filter(valid=False)
    
    @property
    def evaluating_genotypes(self):
        return self.genotypes.filter(evaluating=True)
    
    @property
    def pending_genotypes(self):
        """
        Returns genotypes that have not yet had their fitness evaluated.
        
        fresh__exact=0&evaluating__exact=0
        """
        return self.genotypes.filter(fresh=False, evaluating=False)
    
    @property
    def best_genotype(self):
        """
        Returns the completely evaluated valid genotype with the highest
        fitness.
        """
        q = self.complete_genotypes.order_by('-fitness')
        if q.count():
            return q[0]
    
    def save(self, *args, **kwargs):
        
        old = None
        old_production_genotype = None
        
        if self.id:
            
            old = type(self).objects.get(id=self.id)
            old_production_genotype = old.production_genotype
            
            q = self.genotypes.filter(fitness__isnull=False).exclude(fitness=float('nan'))\
                .aggregate(Max('fitness'), Min('fitness'))
            self.max_fitness = q['fitness__max']
            self.min_fitness = q['fitness__min']
            
            if self._epoche is None or self._epoche.index != self.epoche:
                self._epoche = self.current_epoche
            self._epoche.save()
            
            best_genotype = self.best_genotype
            self.production_at_best = self.production_genotype == best_genotype
        
            #TODO:how to handle species that need to be deleted?
            missing_species = max(0, self.max_species - self.species.all().count())
            for _ in xrange(missing_species):
                #print 'creating',_
                species = Species(genome=self)
                species.save()
                
            if self.max_fitness > old.max_fitness:
                self.epoches_since_improvement = 0
        
        super(Genome, self).save(*args, **kwargs)
        
        if did_genotype_change(old_production_genotype, self.production_genotype):
            production_genotype_changed.send(sender=self, genome=self)
    
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
        return reduce(operator.mul, values, 1)
    
    def total_possible_genotypes_sci(self):
        """
        Calculates the maximum number of unique genotypes given the product
        of the unique values of all genes.
        """
        total = self.total_possible_genotypes()
        if total < 1000:
            return total
        return utils.sci_notation(total)
    total_possible_genotypes_sci.short_description = 'total possible genotypes'
    
    def is_allowable_gene(self, priors, next_gene):
        """
        Given a dict of prior {gene:value} representing a hypothetical genotype,
        determines if the next proposed gene is applicable.
        """
        assert isinstance(next_gene, Gene)
        return True
        #TODO:fix to use GeneDependency
#        if not next_gene.dependee_gene:
#            return True
#        elif priors.get(next_gene.dependee_gene) == next_gene.dependee_value:
#            return True
#        return False
    
    def create_genotype_from_dict(self, d):
        """
        Generates a genotype from a dictionary of gene values.
        """
        new_genotype = self.create_blank_genotype()
        new_ggenes = []
        for gene_name, gene_value in d.items():
            if isinstance(gene_value, models.Model):
                _value = str(gene_value.id)
            else:
                _value = str(gene_value)
            gene = Gene.objects.get(genome=self, name=gene_name)
            new_ggenes.append(GenotypeGene(
                genotype=new_genotype,
                gene=gene,
                _value=_value,
            ))
        GenotypeGene.objects.bulk_create(new_ggenes)
        return new_genotype
    
    def create_random_genotype(self):
        """
        Generates a genotype with a random assortment of genes.
        """
        print 'creating random genotype'
        new_genotype = self.create_blank_genotype()
        d = {}
        new_ggenes = []
        for gene in self.genes.all():#.order_by('-dependee_gene__id'):
            if not self.is_allowable_gene(priors=d, next_gene=gene):
                continue
            d[gene] = _value = gene.get_random_value()
            if isinstance(_value, models.Model):
                _value = str(_value.id)
            else:
                _value = str(_value)
            new_ggenes.append(GenotypeGene(
                genotype=new_genotype,
                gene=gene,
                _value=_value,
            ))
        GenotypeGene.objects.bulk_create(new_ggenes)
        return new_genotype
    
    def create_hybrid_genotype(self, genotypeA, genotypeB):
        """
        Returns a new genotype that is a mixture of the two parents.
        """
        print 'creating hybrid genotype'
        new_genotype = self.create_blank_genotype(
            generation=max(genotypeA.generation, genotypeB.generation)+1,
        )
        
        # Lookup all genes and gene values for both parents
        # so we can quickly access them below.
        all_values = defaultdict(list) # {gene:[values]}
        for gene in genotypeA.genes.all():
            all_values[gene.gene].append(gene._value)
        for gene in genotypeB.genes.all():
            all_values[gene.gene].append(gene._value)
            
        # Note, crossover may result in many invalid or duplicate genotypes,
        # so we can't check the fingerprint until the gene selection
        # is complete.
        priors = {}
        # Order independent genes first so we don't automatically ignore
        # dependent genes just because we haven't added their dependee yet.
        genes = all_values.iterkeys()
        #sorted(all_values.iterkeys(), key=lambda gene: gene.dependee_gene)
        ggenes = []
        for gene in genes:
            if not self.is_allowable_gene(priors=priors, next_gene=gene):
                continue
            new_value = random.choice(all_values[gene])
            ggenes.append(GenotypeGene(
                genotype=new_genotype,
                gene=gene,
                _value=new_value))
            priors[gene] = new_value
        
        GenotypeGene.objects.bulk_create(ggenes)
        
        return new_genotype
        
    def create_mutant_genotype(self, genotype):
        """
        Returns a new genotype that is a slight random modification of the
        given parent.
        """
        print 'creating mutant genotype'
        new_genotype = self.create_blank_genotype(generation=genotype.generation+1)
        
        priors = {}
        
        ggenes = genotype.genes.all()#order_by('-gene__dependee_gene__id')
        ggene_count = ggenes.count()
        ggene_weights = dict((ggene, max(ggene.gene.mutation_weight or 0, 0)) for ggene in ggenes.iterator())
        ggene_weights_sum = sum(max(ggene.gene.mutation_weight or 0, 0) for ggene in ggenes.iterator())
        
        # Randomly select K elements and add weight then do weighted selection.
        k = max(1, int(round(self.mutation_rate * ggene_count, 0)))
        mutated_genes = set(_.gene.name for _ in utils.weighted_samples(choices=ggene_weights, k=k))
        print '%i mutated genes' % len(mutated_genes), ', '.join(sorted(_ for _ in mutated_genes))
        new_ggenes = []
        for ggene in ggenes.iterator():
#            if not self.is_allowable_gene(priors=priors, next_gene=gene.gene):
#                continue
            new_gene = ggene
            new_gene.id = None
            new_gene.genotype = new_genotype
#            if random.random() <= self.mutation_rate:
            if ggene.gene.name in mutated_genes:
                old_value = new_gene._value
                new_gene._value = ggene.gene.get_random_value()
#                if ggene.gene.type == c.GENE_TYPE_GENOME:
#                    genome_id = Genome.objects.get(genotypes__id=int(new_gene._value))
                #print 'Mutating gene %s from %s to %s.' % (ggene.gene.name, old_value, new_gene._value)
            new_ggenes.append(new_gene)
        
        GenotypeGene.objects.bulk_create(new_ggenes)
        
        return new_genotype
    
    @property
    def valid_genotypes(self):
        return Genotype.objects.valid().filter(genome=self)
    
    def get_dependee_genomes(self, genotype=None):
        """
        Returns a list of genomes this genome depends on for current
        production use.
        """
        genotype = genotype or self.production_genotype
        if not genotype:
            return []
        a = set()
        genes = genotype.genes
        q = genes.filter(gene__type=c.GENE_TYPE_GENOME)
        for ggene in q.iterator():
            genome_id = (ggene._value or '').split(':')
            if not genome_id:
                continue
            genome_id = int(genome_id[0])
            try:
                genome = Genome.objects.get(id=genome_id)
                a.add((ggene.gene.name, genome))
            except Genome.DoesNotExist:
                continue
        return a
    
    def is_production_ready(self, genotype=None, as_bool=False):
        genotype = genotype or self.production_genotype
        if not genotype:
            return False
        genomes = self.get_dependee_genomes(genotype=genotype)
        print('Genome %i has %i dependee genomes.' % (self.id, len(genomes)))
        for gene_name, dependee_genome in genomes:
            dependee_genotype = genotype.getattr(gene_name)
            assert isinstance(dependee_genotype, Genotype)
            sub_ret = dependee_genome.is_production_ready(genotype=dependee_genotype)
            is_ready = ret_to_ready(sub_ret)
            if not is_ready:
                if as_bool:
                    return is_ready
                return sub_ret
        ret = self.is_production_ready_function(genotype=genotype)
        if as_bool:
            return ret_to_ready(ret)
        return ret
    is_production_ready.boolean = True
    is_production_ready.short_description = 'production ready'
    
    def delete_corrupt(self, genotype_ids=[], save=True):
        """
        Deletes genotypes without a fingerprint, which should only happen
        because it collided with a duplicate genotype.
        """
        
        # Delete all genotypes that couldn't render a fingerprint, implying
        # it's a duplicate.
        q = self.genotypes.filter(fingerprint__isnull=True)
        if genotype_ids:
            q = q.filter(id__in=genotype_ids)
        for gt in q.iterator():
            print 'Deleting corrupt genotype %s...' % (gt,)
            gt.delete()
        
        # Delete all genotype genes that are illegal.
        processed_genotype_ids = set()
        q = GenotypeGeneIllegal.objects.filter(genotype__genome=self)
        if genotype_ids:
            q = q.filter(genotype__id__in=genotype_ids)
        q = q.values_list('genotype', flat=True).distinct()
        total = q.count()
        if total:
            print 'Deleting illegal genes from %i genotypes.' % (total,)
            for gt in q.iterator():
                #print 'gt:',gt
                assert isinstance(gt, (int, Genotype))
                if isinstance(gt, int):
                    gt = Genotype.objects.get(id=gt)
                gt.delete_illegal_genes(save=save)
                processed_genotype_ids.add(gt.id)
        # Note, by this point, some genotypes may have had enough genes removed
        # to make them identical to another genotype, causing a conflict.
        Genotype.mark_stale(processed_genotype_ids, save=save)
    
    def freshen_fingerprints(self, genotype_ids=[]):
        """
        Genome
        
        Attempts to calculate or re-calculate the fingerprint for all genotypes
        that either don't have one or have one marked as stale.
        Any genotype that is found to be in conflict will be deleted.
        """
        q = Genotype.objects.get_stale_fingerprints()
        q = q.filter(genome=self)
        if genotype_ids:
            q = q.filter(id__in=genotype_ids)
        total = q.count()
        i = 0
        for genotype in q.iterator():
            try:
                genotype.save(check_fingerprint=True)
            except FingerprintConflictError, e:
                # If we're can't recalculate a unique fingerprint, then
                # we're a duplicate, so delete.
                print 'Deleting genotype %i because it conflicts.' % (genotype.id,)
                genotype.delete()
    
    def populate(self, population=0, populate_method=None):
        """
        Creates random genotypes until the maximum limit for un-evaluated
        genotypes is reached.
        """
        max_retries = 10
        last_pending = None
        populate_count = 0
        max_populate_retries = 10
        theoretical_maximum_population = self.total_possible_genotypes()
        maximum_population = min(population or self.maximum_population, theoretical_maximum_population)
        print 'Populating genotypes...'
        transaction.enter_transaction_management()
        transaction.managed(True)
        try:
            while 1:
                
                # Delete failed and/or corrupted genotypes.
                self.delete_corrupt()
                
                #TODO:only look at fitness__isnull=True if maximum_population=0 or delete_inferiors=False?
                #pending = self.pending_genotypes.count()
                pending = self.genotypes.all().count()
#                if pending >= maximum_population:
#                    print 'Maximum unevaluated population has been reached.'
#                    break
                
                # If there are literally no more unique combinations to find,
                # then don't bother.
                if self.genotypes.count() >= maximum_population:
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
                    
                print
                print ('='*80)+('\nAttempt %i of %i to create new genotype %i of %i (%.02f%%)' % (
                    populate_count+1,
                    max_populate_retries,
                    pending,
                    self.maximum_population,
                    pending/float(self.maximum_population)*100,
                ))
                sys.stdout.flush()
                
                if populate_method == c.POPULATE_ALL:
                    # Populate all possible genotypes by simply iterating over
                    # all possible gene value combinations.
                    # Note, as this bypasses selection and mutation, you should
                    # only do this for trivially small genomes during the
                    # initial population stage.
                    print 'Populating all genotypes.'
                    prior_fingerprints = set(self.genotypes.all().values_list('fingerprint', flat=True))
                    element_sets = []
                    element_names = []
                    for gene in self.genes.all():
                        if not gene.is_discrete():
                            continue
                        element_names.append(gene.name)
                        element_sets.append(set(gene.get_values_list()))
                    #print'element_sets:',element_sets
                    comb_iter = utils.iter_elements(element_sets, rand=True)
                    i = 0
                    j = self.genotypes.count()
                    for comb in comb_iter:
                        i += 1
                        try:
                            attrs = dict(zip(element_names, comb))
                            fp = get_fingerprint(attrs)
                            if fp in prior_fingerprints:
                                continue
                            prior_fingerprints.add(fp)
                            #print fp, attrs
                            print 'Creating genotype %i of %i...' % (i, maximum_population-j)
                            new_genotype = self.create_genotype_from_dict(attrs)
                            new_genotype.fingerprint = None
                            new_genotype.fingerprint_fresh = False
                            # Might raise fingerprint conflict error?
                            new_genotype.save(check_fingerprint=True)
                            transaction.commit()
                            # Check for stopping criteria.
                            if self.genotypes.count() >= maximum_population:
                                print 'Maximum theoretical population has been reached.'
                                #sys.exit()#TODO:remove
                                break
                        except FingerprintConflictError, e:
                            # We catch these to explicitly ignore.
                            print>>sys.stderr, '!'*80
                            print>>sys.stderr, 'FingerprintConflictError: %s' % (e,)
                            sys.stderr.flush()
                            #connection._rollback()
                            transaction.rollback()
                        except ValidationError, e:
                            # We catch these to explicitly ignore.
                            print>>sys.stderr, '!'*80
                            print>>sys.stderr, 'Validation Error: %s' % (e,)
                            sys.stderr.flush()
                            #connection._rollback()
                            transaction.rollback()
                        except IntegrityError, e:
                            # We catch these to explicitly ignore.
                            print>>sys.stderr, '!'*80
                            print>>sys.stderr, 'Integrity Error: %s' % (e,)
                            sys.stderr.flush()
                            #connection._rollback()
                            transaction.rollback()
                        except Exception, e:
                            # These we catch to ensure the transaction is cleanly,
                            # rolled back, but otherwise we want it to continue.
                            transaction.rollback()
                            raise
                        
                else:
                    
                    # Note, we can't just look at fresh genotypes, because all may
                    # have been marked as stale prior to population.
                    #valid_genotypes = self.valid_genotypes
                    valid_genotypes = Genotype.objects.filter(
                        valid=True, fitness__isnull=False).filter(genome=self)
                    random_valid_genotypes = valid_genotypes.order_by('?')
                    random_valid_genotype_keys = list(random_valid_genotypes)
                    random_valid_genotype_weights = utils.normalize_list(
                        [_.fitness for _ in random_valid_genotype_keys])
                    random_valid_genotype_choices = zip(
                        random_valid_genotype_keys,
                        random_valid_genotype_weights)
                    
                    last_pending = pending
                    creation_type = random.randint(1,11)
                    print 'creation_type:',creation_type
                    print 'valid_genotypes:',valid_genotypes.count()
                    for retry in xrange(max_retries):
                        try:
                            print 'Sub-attempt %i of %i.' % (retry+1, max_retries)
                            
                            # Create a new genotype in one of three ways.
                            if valid_genotypes.count() <= 1 or creation_type == 1:
                                new_genotype = self.create_random_genotype()
                            elif valid_genotypes.count() >= 2 and creation_type < 5:
                                a = utils.weighted_choice(
                                    random_valid_genotype_choices,
#                                    random_valid_genotypes,
#                                    get_total=lambda: valid_genotypes.aggregate(Sum('fitness'))['fitness__sum'],
#                                    get_weight=lambda o:o.fitness
                                )
                                b = utils.weighted_choice(
                                    random_valid_genotype_choices,
#                                    random_valid_genotypes,
#                                    get_total=lambda: valid_genotypes.aggregate(Sum('fitness'))['fitness__sum'],
#                                    get_weight=lambda o:o.fitness
                                )
                                new_genotype = self.create_hybrid_genotype(a, b)
                            else:
                                new_genotype = self.create_mutant_genotype(random_valid_genotypes[0])
                            
                            self.add_missing_genes(new_genotype)
                            new_genotype.fingerprint = None
                            new_genotype.fingerprint_fresh = False
                            
                            # Might raise fingerprint conflict error.
                            new_genotype.save(check_fingerprint=True)
                            
                            transaction.commit()
                            break
                        except ValidationError, e:
                            # We catch these to explicitly ignore.
                            print>>sys.stderr, '!'*80
                            print>>sys.stderr, 'Validation Error: %s' % (e,)
                            sys.stderr.flush()
                            #connection._rollback()
                            transaction.rollback()
                        except IntegrityError, e:
                            # We catch these to explicitly ignore.
                            print>>sys.stderr, '!'*80
                            print>>sys.stderr, 'Integrity Error: %s' % (e,)
                            sys.stderr.flush()
                            #connection._rollback()
                            transaction.rollback()
                        except Exception, e:
                            # These we catch to ensure the transaction is cleanly,
                            # rolled back, but otherwise we want it to continue.
                            transaction.rollback()
                            raise
                    
        finally:
            transaction.commit()
            transaction.leave_transaction_management()
            connection.close()
    
    @property
    def evaluator_cls(self):
        name = self.evaluator
        name = name.replace('.models.', '.')
        cls = _evaluators.get(name)
#        if cls is None:
#            raise Exception, 'Could not find class for evaluator %s' % (self.evaluator,)
        return cls
    
    @property
    def evaluator_function(self):
        cls = self.evaluator_cls
        if cls:
            return cls.evaluate_genotype
    
    @property
    def mark_stale_function(self):
        cls = self.evaluator_cls
        if cls:
            return cls.mark_stale_genotype
    
    @property
    def reset_function(self):
        cls = self.evaluator_cls
        if cls:
            return cls.reset_genotype
    
    @property
    def pre_delete_function(self):
        cls = self.evaluator_cls
        if cls:
            return cls.pre_delete_genotype
    
    @property
    def is_production_ready_function(self):
        obj = self.evaluator_cls
        if not obj:
            return (lambda *args, **kwargs: False)
        return obj.is_production_ready_genotype
    
    @property
    def calculate_fitness_function(self):
        return self.evaluator_cls.calculate_genotype_fitness
    
    def add_missing_genes(self, genotype=None, save=True, genotype_ids=[]):
        """
        Find all genotype gene values that should exist but don't,
        and creates them.
        """
        _genotype_ids = set()
        while 1:
            q = GenotypeGeneMissing.objects.filter(genotype__genome=self)
            if genotype:
                q = q.filter(genotype=genotype)
            if genotype_ids:
                q = q.filter(genotype__id__in=genotype_ids)
            total = q.count()
            if not total:
                break
            print 'Adding %i missing gene values.' % (total,)
            i = 0
            for missing in q:#.iterator():
                i += 1
                #if i == 1 or not i % 10 or i == total:
                print 'Adding gene value %i of %i: %s...' \
                    % (i, total, missing.gene_name)
                #sys.stdout.flush()
                _genotype_ids.add(missing.genotype_id)
                #print 'adding missing:',missing.gene_name
                missing.create()
            print 'Done!'
            sys.stdout.flush()
        if _genotype_ids:
            Genotype.mark_stale(_genotype_ids, save=save)
    
    @commit_on_success
    def delete_worst_genotypes(self):
        """
        Deletes all but the best genotypes.
        The amount kept is controlled by the elite_ratio parameter.
        """
        #TODO:delete invalid genotypes as well?
        #Note, invalid genotypes are ones that encountered ANY errors
        #during evaluation but still had enough data to acquire a fitness.
        q = self.genotypes.filter(fitness__isnull=False).order_by('-fitness')
        keep_n = int(round(self.maximum_population * self.elite_ratio))
        keep_ids = [_.id for _ in q[:keep_n]]
        if self.production_genotype:
            keep_ids.append(self.production_genotype.id)
        q = self.genotypes\
            .exclude(id__in=keep_ids)\
            .exclude(immortal=True)\
            .exclude(genotype_genes_referrers__id__isnull=False)\
            .only('id')
        total = q.count()
        print '%i inferior genotypes to delete.' % total
        i = 0
        for gt in q:
            i += 1
            print '\rDeleting genotype %i (%i of %i %.02f%%)...' \
                % (gt.id, i, total, i/float(total)*100),
            sys.stdout.flush()
            self.pre_delete_function(gt)
            gt.delete()
        print
    
    def set_production_genotype(self, save=True):
        """
        Assigns the most fit genotype as the recommended solution.
        """
        if not self.production_genotype_auto:
            # We can't select the best unless we've been told we're allowed to.
            return
        if self.pending_genotypes.all().count():
            # We can't select the best until everything's evaluated.
            return
        # Order highest fitness first.
        q = self.complete_genotypes.all().order_by('-fitness')
        if not q.count():
            # Don't set anything if there are no valid genotypes to select.
            return
        self.production_genotype = q[0]
        #TODO:notify dependent models to mark themselves as stale?
        if save:
            self.save()
    
    def cleanup(self, genotype_ids=[], create=True, update=True, delete=True):
        """
        If the genome is edited or altered, this may cause
        genotypes to become broken in some way.
        This attempts to fix some common problems caused by adding, removing,
        or editing gene values to existing genotypes.
        """
        genotype_ids = genotype_ids or []
        genotype_ids = [_ for _ in genotype_ids if _]
        
        if create:
            print 'Adding missing genes...'
            self.add_missing_genes(genotype_ids=genotype_ids)
        
        if update:
            print 'Freshening fingerprints...'
            self.freshen_fingerprints(genotype_ids=genotype_ids)
        
        if delete:
            print 'Deleting corrupt genotypes...'
            self.delete_corrupt(genotype_ids=genotype_ids)
    
    def evolve(self,
        genotypes=None,
        populate=True,
        populate_method=None,
        population=0,
        evaluate=True,
        epoches=0,
        force_reset=False,
        continuous=False,
        cleanup=True,
        clear=True,
        progress=True,
        processes=None):
        """
        Runs a one or more cycles of genotype deletion, generation and evaluation.
        """
        
#        assert not processes or utils.is_power_of_two(processes), \
#            'Processes must be a power of 2.'
        
#        print 'mean_max_memory_usage:',self.mean_max_memory_usage
#        print 'processes:',processes
        processes = processes or self.get_evolve_processes()
#        print 'processes:',processes
        if not processes:
            raise Exception, 'Not enough memory to run evaluate even one genotype. Free up more memory and re-run.'
#        return
        
        tmp_debug = settings.DEBUG
        settings.DEBUG = False
        pid0 = os.getpid()
        max_epoches = epoches
#        print 'max epoches:',max_epoches
        passed_epoches = 0
        pool = None
        genotype_ids = genotypes
        try:
            self.evolving = True
            self.evolution_start_datetime = timezone.now()
            Genome.objects.filter(id=self.id).update(
                evolving=self.evolving,
                evolution_start_datetime=self.evolution_start_datetime,
            )
            while 1:
                if pid0 != os.getpid():
                    return
                
                # Reset stuck genotypes.
                self.genotypes.filter(evaluating=True)\
                    .update(evaluating=False, evaluating_pid=None)
                
                if cleanup:
                    self.cleanup(genotype_ids=genotype_ids)
                    
                # Only cycle epoches if we've completely evaluated all
                # genotypes in the previous epoche.
                # Otherwise, continue evaluating the current epoche.
                pending_genotypes = self.pending_genotypes
                if cleanup and not pending_genotypes.count():
                    
                    # Delete the worst if population at max.
                    self.delete_worst_genotypes()
                    
                    # Mark remaining genotypes as stale so they can be
                    # re-evaluated and compared with new genotypes.
                    #TODO:make this optional?
                    self.genotypes.all().update(fresh=False)
                
                # Creates the initial genotypes for the new epoche.
                # Note, must come before add_missing_genes in case hybridization
                # results in gene loss.
                if populate:
                    print 'Populating...'
                    self.populate(
                        population=population,
                        populate_method=populate_method)
                    
                    # Ensure all new genotypes have a valid fingerprint.
                    if cleanup:
                        self.cleanup(genotype_ids=genotype_ids)
                else:
                    print 'Skipping population.'
                
                # Evaluate un-evaluated genotypes.
                if evaluate:
                    
                    # Start processes.
                    #progress = utils.MultiProgress(clear=clear)
                    lock = Lock()
                    
                    keep_updating_progress = True
                    
                    def update_progress():
                        connection.close()
                        pid0 = os.getpid()
                        while keep_updating_progress:
                            if os.getpid() != pid0:
                                return
                            Genotype.objects.update()
                            q = Genotype.objects.filter(genome__id=self.id)
                            if genotype_ids:
                                q = q.filter(id__in=genotype_ids)
                            overall_total_count = q.count()
                            overall_current_count = q.filter(fresh=True).count()
                            
                            Job.update_progress(
                                total_parts_complete=overall_current_count,
                                total_parts=overall_total_count,
                            )
#                            progress.pid = pid0
#                            progress.cpu = utils.get_cpu_usage(pid=os.getpid())
#                            progress.seconds_until_timeout = 1e999999999999999
#                            progress.current_count = overall_current_count
#                            progress.total_count = overall_total_count
                            #alive_count = len(pool._pool._pool) if pool and pool._pool else None # for multiprocessing.Pool
                            #alive_count = len(pool._pool._pool) if pool else 0 #TODO:for joblib.Parallel?
#                            progress.write('EVOLVE: Evaluating %i genotypes.' % alive_count)
#                            progress.flush(force=True)
                            #print 'pool:',pool,pool and pool._pool
                            print (('-'*80)+'\n'+'Evaluating %i of %i with %i processes.\n'+('-'*80)) \
                                % (overall_current_count, overall_total_count, processes)
                            
                            time.sleep(10)
                    
                    # Start monitoring process status.
                    progress_thread = None
                    if progress:
                        progress_thread = threading.Thread(target=update_progress)
                        progress_thread.daemon = True
                        progress_thread.start()
                    
                    # Collect genotypes to evaluate.
                    if force_reset:
                        genotypes = self.genotypes.all()
                    else:
                        genotypes = self.pending_genotypes
                    if genotype_ids:
                        genotypes = genotypes.filter(id__in=genotype_ids)
                    
                    # Build task list.
                    tasks = [
                        delayed(genome_evolve_pool_helper)(**dict(
                            genome_id=self.id,
                            genotype_id=_.id,
                            force_reset=force_reset,
                        ))
                        for _ in genotypes
                    ]
                    
                    # Required, otherwise we get a django.db.utils.OperationalError:
                    # "server closed the connection unexpectedly"
                    connection.close()
                    
                    # Launch task processes.
#                    print 'tasks:',len(tasks)
                    #NOTE:still can't be killed with KeyboardInterrupt?!
                    #pool = Pool(processes=processes)
#                    pool_iter = pool.imap_unordered(
#                        genome_evolve_pool_helper,
#                        tasks)
                    #results = pool.map_async(genome_evolve_pool_helper, tasks).get(9999999)
                    #pool.close()
                    #pool.join()
#                    print 'processes:',processes
#                    print 'genotypes:',genotypes.count()
                    processes = min(processes, genotypes.count())
                    pool = Parallel(n_jobs=processes, verbose=10)
                    ret = pool(_ for _ in tasks)
                    
                    connection.close()
                    
                    keep_updating_progress = False
                    progress_thread.join()
                    
                    # Make final update.
#                    Genotype.objects.update()
#                    q = Genotype.objects.filter(genome__id=self.id)
#                    if genotype_id:
#                        q = q.filter(id=int(genotype_id))
#                    overall_total_count = q.count()
#                    overall_current_count = q.filter(fresh=True).count()
#                    Job.update_progress(
#                        total_parts_complete=overall_current_count,
#                        total_parts=overall_total_count,
#                    )
                    
                    # Reload the current genome in case we've received updates.
                    Genome.objects.update()
                    genome = self = Genome.objects.get(id=self.id)
                    genome.generate_error_report(force=True, save=False)
                    genome.genotypes.filter(evaluating=True).update(evaluating=False)
                    if not genotype_ids:
                        q = Epoche.objects.filter(index=genome.epoche, genome=genome)
                        if q.exists():
                            q[0].save(force_recalc=True)
                        genome.epoche += 1
                        passed_epoches += 1
                        genome.epoches_since_improvement += 1
                        #type(self).objects.filter(id=self.id).update(epoche=self.epoche)
                        genome.set_production_genotype(save=False)
                    genome.evolving = True
                    genome.save()
                else:
                    print 'Stopping because evaluation not selected.'
                    return
                
                Genome.objects.update()
                genome = Genome.objects.get(id=self.id)
                if max_epoches:
                    if passed_epoches >= max_epoches:
                        print 'Stopping because we have completed %s epoches out of %s max epoches.' % (passed_epoches, max_epoches)
                        break
                elif not continuous or genome.stalled():
                    print 'Stopping because genome has stalled.'
                    break
                
                # Clear the query cache to help reduce memory usage.
                reset_queries()
        
        except KeyboardInterrupt, e:
            if pool:
                pool.terminate()
                pool.close()
                pool.join()
                
        finally:
            if pid0 == os.getpid():
                self.evolving = False
                Genome.objects.filter(id=self.id).update(evolving=self.evolving)
                settings.DEBUG = tmp_debug
                #django.db.transaction.commit()
                connection.close()
        print 'Done.'
            
    def evaluate(self, genotype_id, force_reset=False):
        """
        Calculates the fitness of a genotype.
        """
        
        _stdout = sys.stdout
        sys.stdout = utils.PrefixStream(
            stream=sys.stdout,
            prefix='%i: ' % genotype_id)
        
        # Start tracking process memory usage.
        mt = utils.MemoryThread()
        mt.start()
        
        # Run the backend evaluator.
        try:
            gt = Genotype.objects.get(id=genotype_id)
            gt.reset()
            gt.evaluating = True
            gt.evaluating_pid = os.getpid()
            gt.fitness_evaluation_datetime_start = timezone.now()
            gt.error = None # This may be overriden
            gt.save()
            fitness_evaluation_datetime_start = timezone.now()
            self.evaluator_function(gt, force_reset=force_reset)#, progress=progress)
            print 'Done evaluating genotype %s.' % (gt.id,)
            mt.stop = True
            mt.join()
            gt = Genotype.objects.get(id=gt.id)
            print 'final fitness: %s' % (gt.fitness,)
            gt.memory_usage_samples = mt.memory_history
            print 'memory_usage_samples:',gt.memory_usage_samples
            gt.mean_memory_usage = None
            if mt.memory_history:
                gt.mean_memory_usage = int(sum(mt.memory_history)/float(len(mt.memory_history)))
            print 'mean_memory_usage:',gt.mean_memory_usage
            gt.max_memory_usage = None
            if gt.memory_usage_samples:
                gt.max_memory_usage = max(gt.memory_usage_samples)
            gt.total_evaluation_seconds = None
            gt.fitness_evaluation_datetime_start = fitness_evaluation_datetime_start
            gt.fitness_evaluation_datetime = timezone.now()
            gt.fresh = True # evaluated, even if failed
            gt.evaluating = False
            gt.evaluating_pid = None
            gt.valid = not gt.error
            gt.epoche_of_evaluation = self.epoche
            gt.epoche = gt.genome.current_epoche
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
                evaluating=False,
                evaluating_pid=None,
            )
        finally:
            sys.stdout = _stdout
            
    def production_evaluate(self, all=False, genotype=0):
        """
        Prepares the production genotype for production use.
        Does nothing if any dependent genomes are not ready.
        """
        #progress = utils.MultiProgress(clear=False)
        
        # Determine production genotype.
        production_genotype = self.production_genotype
        if genotype:
            production_genotype = Genotype.objects.get(genome=self, id=genotype)
        if not production_genotype:
            print('No production genotype set.')
            return False
        
        # Determine of genotypes are evaluated and ready
        # for production use.
        prod_ready_ret = self.is_production_ready(genotype=production_genotype)
        prod_ready = ret_to_ready(prod_ready_ret)
        production_genotype.production_fresh = prod_ready
        production_genotype.save()
        dep_genomes = list(self.get_dependee_genomes()) or None
        print('\tproduction ready:',prod_ready)
        print('\tdependent genomes:',dep_genomes)
        
        if prod_ready:
            # Don't evaluate if there's nothing to do.
            print('Genome %i:%i is ready.' % (self.id, production_genotype.id))
            return True
        elif dep_genomes:
            print('%i dependee genomes.' % len(dep_genomes))
            for gene_name, dep_genome in dep_genomes:
                dependee_genotype = production_genotype.getattr(gene_name)
                print('Checking dependee genome %s with genotype %i...' \
                    % (dep_genome, dependee_genotype.id))
                is_dep_ready_ret = dep_genome.is_production_ready(
                    genotype=dependee_genotype)
                is_dep_ready = ret_to_ready(is_dep_ready_ret)
                if is_dep_ready:
                    print('Dependee genome %i with genotype %i is ready!' \
                        % (dep_genome.id, dependee_genotype.id))
                else:
                    if all:
                        # Refresh dependee if we're refreshing everything.
                        dep_genome.production_evaluate(
                            all=all,
                            genotype=dependee_genotype)
                        # Recheck.
                        is_dep_ready_ret = dep_genome.is_production_ready(
                            genotype=dependee_genotype)
                        is_dep_ready = ret_to_ready(is_dep_ready_ret)
                        
                    if not is_dep_ready:
                        # Don't evaluate if we're dependent on another genome
                        # that's not production ready.
                        print>>sys.stderr, \
                            'Genome %i depends on %i:%i which is not ready: %s' \
                                % (
                                    self.id,
                                    dep_genome.id,
                                    dependee_genotype.id,
                                    is_dep_ready_ret[-1])
                        return False
                    
        print('Evaluating production genotype...')
        t = Thread(
            target=self.evaluator_function,
            kwargs=dict(
                genotype=self.production_genotype,
                test=False,
                #progress=progress,
            ))
        t.daemon = True
        t.start()
        while t.is_alive():
#            progress.pid = os.getpid()
#            progress.cpu = utils.get_cpu_usage(pid=os.getpid())
#            progress.flush()
            time.sleep(1)
            
        prod_ready_ret = self.is_production_ready(genotype=production_genotype)
        prod_ready = ret_to_ready(prod_ready_ret)
        production_genotype.production_fresh = prod_ready
        production_genotype.save()
        if prod_ready:
            print('Genome %i:%i is ready.' % (self.id, production_genotype.id))
        else:
            print('Genome %i:%i is not ready: %s' \
                % (self.id, production_genotype.id, prod_ready_ret[-1]))
        return prod_ready

def genome_evolve_pool_helper(**kwargs):
    # Django is not multiprocessing safe, so we must manually close
    # its database connection, otherwise each process will corrupt
    # the other's queries.
    print '!'*80
    print'kwargs:',kwargs
    connection.close()
    
    try:
        genome_id = kwargs['genome_id']
        del kwargs['genome_id']
        genome = Genome.objects.only('id').get(id=genome_id)
        return genome.evaluate(**kwargs)

    except Exception, e:
        print '!'*80
        print 'Error:',e
        traceback.print_exc(file=sys.stderr)
        return e

class EpocheManager(models.Manager):
    
    def get_by_natural_key(self, index, *args, **kwargs):
        genome = Genome.objects.get_by_natural_key(*args, **kwargs)
        return self.get_or_create(index=index, genome=genome)[0]

class Epoche(BaseModel):
    
    objects = EpocheManager()
    
    genome = models.ForeignKey(Genome, related_name='epoches')
    
    index = models.PositiveIntegerField(
        default=1,
        editable=False,
        db_index=True)
    
    min_fitness = models.FloatField(
        blank=True,
        null=True,
        editable=False,
        db_index=True,
        help_text=_('The smallest observed fitness.'))
    
    mean_fitness = models.FloatField(
        blank=True,
        null=True,
        editable=False,
        db_index=True,
        help_text=_('The mean observed fitness.'))
    
    max_fitness = models.FloatField(
        blank=True,
        null=True,
        editable=False,
        db_index=True,
        help_text=_('The largest observed fitness.'))
    
    oldest_epoche_of_creation = models.PositiveIntegerField(
        blank=True,
        null=True,
        editable=False,
        help_text=_('The oldest EOC of any non-immortal genotype.'))
    
    def __unicode__(self):
        return unicode(self.index)
    
    class Meta:
        app_label = APP_LABEL
        unique_together = (
            ('genome', 'index'),
        )
        ordering = ('genome', '-index')
    
    natural_key_fields = ('index', 'genome')
    
    def natural_key(self):
        return (self.index,) + self.genome.natural_key()
    natural_key.dependencies = ['django_analyze.Genome']
    
    def save(self, force_recalc=False, *args, **kwargs):
        if self.id and (force_recalc or self.genome.epoche == self.index):
            
            q0 = self.genotypes.filter(
                fitness__isnull=False, epoche_of_evaluation=self.index
            ).exclude(
                fitness=float('nan')
            )
            
            q = q0.aggregate(
                Max('fitness'),
                Min('fitness'),
                Avg('fitness'),
                Max('epoche_of_creation'),
            )
            self.max_fitness = q['fitness__max']
            self.mean_fitness = q['fitness__avg']
            self.min_fitness = q['fitness__min']
            
            q = q0.filter(
                # Note, we have to exclude immortals, otherwise they'd always
                # show up as the oldest, and we're only interested in tracking
                # the naturally-occurring oldest, not the ones we explicitly
                # made be the oldest.
                immortal=False
            ).aggregate(Max('epoche_of_creation'))
            self.oldest_epoche_of_creation = q['epoche_of_creation__max']
            
        super(Epoche, self).save(*args, **kwargs)

class GeneStatistics(BaseModel):
    """
    Tracks gene fitness statistics within a specific epoche.
    """
    
    id = models.CharField(
        max_length=1000,
        primary_key=True,
        editable=False,
        blank=False,
        null=False)
    
    genome = models.ForeignKey(
        'Genome',
        editable=False,
        blank=False,
        null=False,
        on_delete=models.DO_NOTHING,
        related_name='gene_statistics',
        db_column='genome_id')
    
    gene = models.ForeignKey(
        'Gene',
        editable=False,
        blank=False,
        null=False,
        on_delete=models.DO_NOTHING,
        db_column='gene_id')
    
    value = models.CharField(
        max_length=1000,
        editable=False,
        blank=False,
        null=False)
    
    min_fitness = models.FloatField(
        blank=True,
        null=True,
        editable=False,
        help_text=_('The smallest observed fitness.'))
    
    mean_fitness = models.FloatField(
        blank=True,
        null=True,
        editable=False,
        help_text=_('The mean observed fitness.'))
    
    max_fitness = models.FloatField(
        blank=True,
        null=True,
        editable=False,
        help_text=_('The largest observed fitness.'))
    
    genotype_count = models.PositiveIntegerField(
        blank=False,
        null=False,
        editable=False,
        verbose_name=_('genotypes'),
        help_text=_('The number of genotypes using this specific gene value.'),
    )
    
    class Meta:
        managed = False
        app_label = APP_LABEL
        verbose_name = _('gene statistics')
        verbose_name_plural = _('gene statistics')
        ordering = ('genome', 'gene', '-mean_fitness')
        
#class EpocheGene(BaseModel):
#    """
#    Tracks gene fitness statistics within a specific epoche.
#    """
#    
#    epoche = models.ForeignKey('Epoche', editable=False)
#    
#    gene = models.ForeignKey('Gene', editable=False)
#    
#    value = models.CharField(
#        max_length=1000,
#        editable=False,
#        db_index=True,
#        blank=False,
#        null=False)
#    
#    min_fitness = models.FloatField(
#        blank=True,
#        null=True,
#        editable=False,
#        db_index=True,
#        help_text=_('The smallest observed fitness.'))
#    
#    mean_fitness = models.FloatField(
#        blank=True,
#        null=True,
#        editable=False,
#        db_index=True,
#        help_text=_('The mean observed fitness.'))
#    
#    max_fitness = models.FloatField(
#        blank=True,
#        null=True,
#        editable=False,
#        db_index=True,
#        help_text=_('The largest observed fitness.'))
#    
#    class Meta:
#        app_label = APP_LABEL
#        unique_together = (
#            ('epoche', 'gene', 'value'),
#        )
#        ordering = ('epoche', 'gene', 'value')
#    
#    @classmethod
#    def populate(cls, epoche):
#        exclude_gene_ids = cls.objects.filter(epoche=epoche).values_list('gene__id', flat=True)
#        genome = epoche.genome
#        fit_genotypes = genome.genotypes.filter(fitness__isnull=False)
#    
#    def save(self, force_recalc=False, *args, **kwargs):
#            
#        super(EpocheGene, self).save(*args, **kwargs)

class GeneDependencyManager(models.Manager):

    def get_by_natural_key(self, gene_name, genome_name, dependee_gene_name, dependee_gene_genome_name, dependee_value):
        gene = Gene.objects.get_by_natural_key(gene_name, genome_name)
        dependee_gene = Gene.objects.get_by_natural_key(dependee_gene_name, dependee_gene_genome_name)
        return self.get_or_create(
            gene=gene,
            dependee_gene=dependee_gene,
            dependee_value=dependee_value)[0]
    
class GeneDependency(BaseModel):
    """
    Defines when a gene can be used if the dependee gene exists
    in the genotype with the dependee value.
    
    When one gene has multiple dependencies, all are accumulated
    in a disjunction (ORed together), so only one dependency
    has to be true to allow the gene to be used.
    """
    
    objects = GeneDependencyManager()
    
    gene = models.ForeignKey(
        'Gene',
        related_name='dependencies',
        blank=False,
        null=False,
        help_text=_('The dependent gene.'))
    
    dependee_gene = models.ForeignKey(
        'Gene',
        related_name='dependents',
        blank=False,
        null=False,
        help_text='''The gene this gene is dependent upon. This gene will only
            activate when the dependee gene has a certain value.''')
    
    dependee_value = models.CharField(
        max_length=1000,
        blank=False,
        null=False,
        help_text='''The value of the dependee gene that activates
            this gene.''')
    
    positive = models.BooleanField(
        default=True,
        help_text=_('If false, implies the dependent gene must NOT have this value.'))

    class Meta:
        app_label = APP_LABEL
        verbose_name = _('gene dependency')
        verbose_name_plural = _('gene dependencies')
        unique_together = (
            ('gene', 'dependee_gene', 'dependee_value'),
        )
    
    natural_key_fields = ('gene', 'dependee_gene', 'dependee_value')
    
    def natural_key(self):
        return self.gene.natural_key() + self.dependee_gene.natural_key() + (self.dependee_value,)
    natural_key.dependencies = ['django_analyze.Gene']

class GeneManager(models.Manager):
    
    def get_by_natural_key(self, name, *args, **kwargs):
        #print 'GeneManager:',name,args,kwargs
        genome = Genome.objects.get_by_natural_key(*args, **kwargs)
        return self.get_or_create(name=name, genome=genome)[0]

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
    
#    dependee_gene = models.ForeignKey(
#        'self',
#        related_name='dependent_genes',
#        blank=True,
#        null=True,
#        on_delete=models.SET_NULL,
#        help_text='''The gene this gene is dependent upon. This gene will only
#            activate when the dependee gene has a certain value.''')
#    
#    dependee_value = models.CharField(
#        max_length=1000,
#        blank=True,
#        null=True,
#        help_text='''The value of the dependee gene that activates
#            this gene.''')
    
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
    
    default = models.CharField(
        max_length=1000,
        blank=False,
        null=True)
    
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
#            '-dependee_gene__id',
            'name',
        )
        
    def __unicode__(self):
        #return '<Gene:%s %s>' % (self.id, self.name)
        return '%i:%s' % (self.genome.id if self.genome else None, self.name)
    
    natural_key_fields = ('name', 'genome')
    
    def natural_key(self):
        return (self.name,) + self.genome.natural_key()
    natural_key.dependencies = ['django_analyze.Genome']
    
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
        ratio = None
        if possible_value_count:
            ratio = len(actual_values)/float(possible_value_count)
        self.coverage_ratio = ratio
        if auto_update:
            type(self).objects.filter(id=self.id).update(coverage_ratio=ratio)
        self.update_mutation_weight(auto_update=auto_update)

    def clean(self, *args, **kwargs):
        """
        Called to validate fields before saving.
        Override this to implement your own model validation
        for both inside and outside of admin. 
        """
        try:
            
            values_list = self.get_values_list()
            default_value = self.get_default(allow_random=False)
            if values_list is not None and default_value is not None:
                if default_value not in values_list:
                    raise ValidationError({
                        'default': ['Default must one of the allowed values.'],
                    })
            
            super(Gene, self).clean(*args, **kwargs)
        except Exception, e:
#            print '!'*80
#            print e
            raise

    def full_clean(self, *args, **kwargs):
        return self.clean(*args, **kwargs)

    def save(self, *args, **kwargs):
        old = None
        if self.id:
            old = type(self).objects.get(id=self.id)
            self.update_coverage(auto_update=False)
            
        self.full_clean()
            
        super(Gene, self).save(*args, **kwargs)
        
        schema_change = False
        if old:
            old_values_list = old.get_values_list(as_text=True)
            values_list = self.get_values_list(as_text=True)
            if old_values_list != values_list:
                # Mark stale any genotype that was using a gene value
                # that was removed.
                # Needs to happen before the gene value reset, otherwise this
                # won't find anything.
                q = self.genome.genotypes.exclude(
                    genes__gene=self,
                    genes___value__in=values_list
                )
                print 'Marking %i genotypes as having a stale fingerprint.' % (q.count(),)
                q.update(fingerprint_fresh=False)
                # Set genotype gene value to the default if they're using
                # a now-illegal value.
                q = GenotypeGene.objects.filter(
                    genotype__genome=self.genome, gene=self
                ).exclude(_value__in=values_list)
                cnt = q.count()
                print 'Found %i genotype genes using illegal values.' % (cnt,)
                if cnt:
                    print 'Setting %i genotype genes to the default value.' % (cnt,)
                    q.update(_value=self.default)
                    
            fields_to_check = ['type', 'values', 'default', 'min_value', 'max_value']
            for field in fields_to_check:
                if getattr(self, field) != getattr(old, field):
                    schema_change = True
                    break
        else:
            schema_change = True
        
        # If we changed a schema-effecting field, increment the genome's
        # schema version.
        if schema_change:
            Genome.objects.filter(id=self.genome.id).update(version=F('version')+1)
    
#    @staticmethod
#    def post_save(sender, instance, *args, **kwargs):
#        self = instance
#        Genome.objects.filter(id=self.genome.id).update(version=F('version')+1)
        
    @staticmethod
    def post_delete(sender, instance, *args, **kwargs):
        self = instance
        # If we deleted a gene, the genome schema version always increments.
        Genome.objects.filter(id=self.genome.id).update(version=F('version')+1)
        
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
        """
        Returns true if the gene has a finite number of values
        within non-infinite bounds.
        """
        if self.type == c.GENE_TYPE_FLOAT and self.values:
            return True
        discrete_types = (c.GENE_TYPE_INT, c.GENE_TYPE_BOOL, c.GENE_TYPE_STR, c.GENE_TYPE_GENOME)
        return self.type in discrete_types
    
    def is_continuous(self):
        """
        Returns true if not discrete.
        """
        return not self.is_discrete()
    
    def get_random_value(self):
        values_list = self.get_values_list()
        if values_list:
            if self.type == c.GENE_TYPE_GENOME:
                values_list = [
                    '%i:%i' % (_.genome.id, _.id)
                    for _ in values_list]
            return random.choice(values_list)
        elif self.type == c.GENE_TYPE_INT:
            assert (self.min_value and self.max_value) or (self.default and self.max_increment)
            #print 'min/max:',self.name,self.min_value,self.max_value
            if self.min_value and self.max_value:
                return random.randint(int(self.min_value), int(self.max_value))
            else:
                return random.randint(-int(self.max_increment), int(self.max_increment)) + int(self.default)
        elif self.type == c.GENE_TYPE_FLOAT:
            assert (self.min_value and self.max_value) or (self.default and self.max_increment)
            #print 'min/max:',self.name,self.min_value,self.max_value
            if self.min_value and self.max_value:
                return random.uniform(float(self.min_value), float(self.max_value))
            else:
                return random.uniform(float(self.max_increment), float(self.max_increment)) + float(self.default)
        elif self.type == c.GENE_TYPE_BOOL:
            return random.choice([True,False])
        elif self.type == c.GENE_TYPE_STR:
            raise NotImplementedError, \
                'Cannot generate a random value for a string type with no values.'
        elif self.type == c.GENE_TYPE_GENOME:
            raise NotImplementedError, \
                'Cannot generate a random value for a genome type with no values.'
        else:
            raise NotImplementedError, 'Unknown type: %s' % (self.type,)
    
    def get_default(self, allow_random=True):
        if self.default is None:
            if allow_random:
                return self.get_random_value()
            return
            
        if self.type == c.GENE_TYPE_GENOME:
            default = str_to_type[self.type](self.default, only_production=True)
            if default:
                return default[0]
            else:
                return
        else:
            default = str_to_type[self.type](self.default)
            
        return default
    
    def get_values_list(self, as_text=False):
        """
        Returns a list of allowable values for this gene.
        If gene value starts with "source:package.module.attribute",
        will dynamically lookup this list.
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
        if as_text:
            return lst
            
        lst = [str_to_type[self.type](_) for _ in lst]
        
        # If a gene storing a link to a genotype, flatten a list
        # of genotype lists.
        if self.type == c.GENE_TYPE_GENOME:
            lst = [item for sublist in lst for item in sublist]
            
        return lst

class GenotypeManager(models.Manager):

    def get_by_natural_key(self, fingerprint, *args, **kwargs):
        #print 'GenotypeManager:',fingerprint,args,kwargs
        genome = Genome.objects.get_by_natural_key(*args, **kwargs)
        return self.get_or_create(fingerprint=fingerprint, genome=genome)[0]

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
        
    def get_stale_fingerprints(self):
        return self.filter(
            Q(fingerprint_fresh=False)|Q(fingerprint__isnull=True))

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
    
    description = models.CharField(
        max_length=500,
        blank=True,
        null=True)
    
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
    
    mean_evaluation_seconds = models.FloatField(
        blank=True,
        null=True,
        editable=False,
        help_text=_('''When an evaluation is composed of multiple parts,
            this stores the average time to evaluate each part.'''))
    
    total_evaluation_seconds = models.PositiveIntegerField(
        blank=True,
        null=True,
        editable=False,
        help_text=_('''The total time it last took to evaluate
            the genotype. This is set automatically.'''))
    
    memory_usage_samples = PickledObjectField(
        blank=True,
        null=True,
        editable=False,
        help_text=_('''A list of periodic memory usage measurements during
            evaluation. Used to calculate the mean memory usage.'''))
    
    mean_memory_usage = models.BigIntegerField(
        blank=True,
        null=True,
        editable=False,
        help_text=_('''The average amount of memory in bytes consumed
            at any given point in time during the evaluation.'''))
    
    max_memory_usage = models.BigIntegerField(
        blank=True,
        null=True,
        editable=False,
        help_text=_('''The maximum amount of memory in bytes consumed
            at any given point in time during the evaluation.'''))
    
    #TODO:deprecated? genotype should only store fitness
    mean_absolute_error = models.FloatField(
        blank=True,
        null=True,
        db_index=True,
        editable=False,
        help_text=_('''The mean-absolute-error measure recorded during
            fitness evaluation.'''))
    
    #TODO:deprecated? genotype should only store fitness
    accuracy = models.FloatField(
        blank=True,
        null=True,
        db_index=True,
        editable=False,
        help_text=_('''Boolean accuracy, if the domain supports it.'''))
    
    gene_count = models.PositiveIntegerField(
        verbose_name='genes',
        blank=True,
        null=True,
        editable=False)
    
#    preserve = models.BooleanField(
#        default=False,
#        help_text=_('''If checked, this genotype will not be deleted
#            even if it becomes unfit.'''))
    
    #DEPRECATED, use epoche instead
    epoche_of_evaluation = models.PositiveIntegerField(
        verbose_name=_(' EOE'),
        blank=True,
        null=True,
        editable=False,
        db_index=True,
        help_text=_('The epoche when this genotype was last evaluated.'))
    
    epoche_of_creation = models.PositiveIntegerField(
        verbose_name=_(' EOC'),
        blank=True,
        null=True,
        editable=False,
        db_index=True,
        help_text=_('The epoche when this genotype was created.'))
    
    epoche = models.ForeignKey(
        'Epoche',
        related_name='genotypes',
        on_delete=models.SET_NULL,
        blank=True,
        null=True,
        help_text=_('The epoche when this genotype was last evaluated.'))
    
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
    
    export = models.BooleanField(
        default=False,
        #editable=False,
        db_index=True,
        help_text=_('If true, indicates this genotype may be used by other genomes.'))
        
    valid = models.BooleanField(
        default=True,
        #editable=False,
        db_index=True,
        help_text=_('''If true, indicates this genotype was evaluted without
            any fatal errors. Note, other errors may have occurred as reported
            by the success ratio.'''))
    
    immortal = models.BooleanField(
        default=False,
        editable=True,
        db_index=True,
        help_text=_('If true, this genotype will not be automatically deleted if it is found to be inferior.'))
    
    total_parts = models.PositiveIntegerField(
        blank=True,
        null=True,
        editable=False,
        help_text=_('Total number of sub-evaluations to run.'))
    
    complete_parts = models.PositiveIntegerField(
        blank=True,
        null=True,
        editable=False,
        help_text=_('Total number of sub-evaluations that have run.'))
    
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
        editable=False,
        help_text=_('''If implemented by the backend, represents the fraction
            of cases this genotype successfully addresses.'''))
    
    ontime_ratio = models.FloatField(
        blank=True,
        null=True,
        db_index=True,
        editable=False,
        help_text=_('''If implemented by the backend, represents the fraction
            of cases this genotype is able to evaluate within the timeout.'''))
    
    complete_ratio = models.FloatField(
        blank=True,
        null=True,
        db_index=True,
        editable=False,
        help_text=_('The ratio of parts evaluated for production use.'))
    
    def complete_percent(self):
        if self.complete_ratio is None:
            return '(None)'
        return '%.02f%%' % (self.complete_ratio*100,)
    complete_percent.admin_order_field = 'complete_ratio'
    
    error = models.TextField(
        blank=True,
        null=True,
        help_text=_('Any error message received during evaluation.'))
    
    ## Production status fields.
    
    production_evaluating = models.BooleanField(
        default=False,
        db_index=True,
        verbose_name='evaluating',
        help_text=_('''If checked, indicates this genotype is currently having
            its fitness evaluated for production use.'''))
    
    production_evaluating_pid = models.IntegerField(
        blank=True,
        null=True,
        editable=False,
        verbose_name='evaluating pid',
        help_text=_('The PID of the process evaluating this genotype for production use.'))
    
    production_fresh = models.BooleanField(
        default=False,
        editable=False,
        db_index=True,
        verbose_name='fresh',
        help_text=_('If true, indicates this predictor has been evaluated for production use.'))
    
    production_valid = models.BooleanField(
        default=True,
        #editable=False,
        db_index=True,
        verbose_name='valid',
        help_text=_('''If true, indicates this genotype was evaluted without
            any fatal errors. Note, other errors may have occurred as reported
            by the success ratio.'''))
    
    production_total_parts = models.PositiveIntegerField(
        blank=True,
        null=True,
        editable=False,
        verbose_name='total parts',
        help_text=_('Total number of sub-evaluations to run.'))
    
    production_complete_parts = models.PositiveIntegerField(
        blank=True,
        null=True,
        editable=False,
        verbose_name='complete parts',
        help_text=_('Total number of sub-evaluations that have run.'))
    
    production_success_parts = models.PositiveIntegerField(
        blank=True,
        null=True,
        editable=False,
        verbose_name='success parts',
        help_text=_('Total number of sub-evaluations successfully run.'))
    
    production_ontime_parts = models.PositiveIntegerField(
        blank=True,
        null=True,
        editable=False,
        verbose_name='ontime parts',
        help_text=_('Total number of sub-evaluations that ran ontime.'))
    
    production_success_ratio = models.FloatField(
        blank=True,
        null=True,
        db_index=True,
        editable=False,
        verbose_name='success ratio',
        help_text=_('''If implemented by the backend, represents the fraction
            of cases this genotype successfully addresses.'''))
    
    production_ontime_ratio = models.FloatField(
        blank=True,
        null=True,
        db_index=True,
        editable=False,
        verbose_name='ontime ratio',
        help_text=_('''If implemented by the backend, represents the fraction
            of cases this genotype is able to evaluate within the timeout.'''))
    
    production_complete_ratio = models.FloatField(
        blank=True,
        null=True,
        db_index=True,
        editable=False,
        verbose_name='complete ratio',
        help_text=_('The ratio of parts evaluated for production use.'))
    
    def production_complete_percent(self):
        if self.production_complete_ratio is None:
            return '(None)'
        return '%.02f%%' % (self.production_complete_ratio*100,)
    production_complete_percent.admin_order_field = 'production_complete_ratio'
    production_complete_percent.short_description = 'complete percent'
    
    def production_complete_percent2(self):
        return self.production_complete_percent()
    production_complete_percent2.admin_order_field = 'production_complete_ratio'
    production_complete_percent2.short_description = 'complete percent (production)'
        
    production_error = models.TextField(
        blank=True,
        null=True,
        help_text=_('Any error message received during evaluation.'))
    
    production_evaluation_start_datetime = models.DateTimeField(
        blank=True,
        null=True,
        editable=False,
        db_index=True,
        verbose_name=_('evaluation start time'))
    
    production_evaluation_end_datetime = models.DateTimeField(
        blank=True,
        null=True,
        editable=False,
        db_index=True,
        verbose_name=_('evaluation stop time'))
    
    def production_evaluation_seconds(self):
        start = self.production_evaluation_start_datetime
        end = self.production_evaluation_end_datetime
        if start is None or end is None:
            return
        return (end - start).seconds
    production_evaluation_seconds.short_description = 'evaluation time (seconds)'
    
    def production_evaluation_seconds_str(self):
        from datetime import datetime, timedelta
        total_seconds = self.production_evaluation_seconds()
        if total_seconds is None:
            return
        sec = timedelta(seconds=total_seconds)
        d = datetime(1,1,1) + sec
        days = d.day-1
        hours = d.hour
        minutes = d.minute
        seconds = d.second
        return '%02i:%02i:%02i:%02i' % (days, hours, minutes, seconds)
    production_evaluation_seconds_str.short_description = 'evaluation time (days:hours:min:sec)'
    production_evaluation_seconds_str.allow_tags = True
    
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
    
    natural_key_fields = ('fingerprint', 'genome')
    
    def natural_key(self):
        return (self.fingerprint,) + self.genome.natural_key()
    natural_key.dependencies = ['django_analyze.Genome']

    def get_fingerprint(self):
        return get_fingerprint(dict(
            (gene.gene.name, gene.value)
            for gene in self.genes.all()
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
        Genotype.
        
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
    def mark_stale(cls, genotype_ids, save=True, check_fingerprint=True):
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
            if check_fingerprint:
                genotype.fingerprint_fresh = False
            if save:
                try:
                    genotype.save(check_fingerprint=check_fingerprint)
                except FingerprintConflictError, e:
                    # If we're can't recalculate a unique fingerprint, then
                    # we're a duplicate, so delete.
                    print 'Deleting genotype %i because it conflicts.' % (genotype.id,)
                    genotype.delete()
    
    def delete_illegal_genes(self, save=True):
        """
        Deletes genes that aren't allowed to exist according to genome rules.
        """
        q = self.illegal_gene_values.all()
        total = q.count()
        if total:
            print 'Deleting %i illegal gene values from genotype %i.' % (total, self.id)
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
            if do_validation_check() and self.id and check_fingerprint:
                fingerprint = self.get_fingerprint()
                q = self.genome.genotypes.filter(fingerprint=fingerprint).exclude(id=self.id)
                if q.count():
                    url = get_admin_change_url(q[0])
                    #raise ValidationError(mark_safe(('Fingerprint for genotype'
                    raise FingerprintConflictError(mark_safe((
                        'Fingerprint for genotype %s conflicts with '
                        '<a href="%s" target="_blank">genotype %i</a>, indicating '
                        'one of these genotypes is a duplicate of the other. '
                        'Either delete one of these genotypes or change their '
                        'gene values so that they differ.') % (self.id, url, q[0].id,)))
            
            if 'exclude' in kwargs:
                del kwargs['exclude']
            if 'validate_unique' in kwargs:
                del kwargs['validate_unique']
            super(Genotype, self).clean(*args, **kwargs)
        except Exception, e:
#            print '!'*80
#            print e
            raise

    def full_clean(self, check_fingerprint=True, *args, **kwargs):
        return self.clean(check_fingerprint=check_fingerprint, *args, **kwargs)
    
    def update_status(self, success_parts, ontime_parts, total_parts, complete_parts, test=True):
        if test:
            self.total_parts = total_parts
            self.success_parts = success_parts
            self.ontime_parts = ontime_parts
            self.complete_parts = complete_parts
            self.success_ratio = success_parts/float(total_parts) if total_parts else None
            self.ontime_ratio = ontime_parts/float(total_parts) if total_parts else None
            
            self.complete_ratio = None
            if self.total_parts and self.complete_parts is not None:
                self.complete_ratio = self.complete_parts/float(self.total_parts)
                
            type(self).objects.filter(id=self.id).update(
                total_parts=self.total_parts,
                complete_parts=self.complete_parts,
                success_parts=self.success_parts,
                ontime_parts=self.ontime_parts,
                success_ratio=self.success_ratio,
                ontime_ratio=self.ontime_ratio,
                complete_ratio=self.complete_ratio,
            )
        else:
            self.production_total_parts = total_parts
            self.production_success_parts = success_parts
            self.production_ontime_parts = ontime_parts
            self.production_complete_parts = complete_parts
            self.production_success_ratio = success_parts/float(total_parts) if total_parts else None
            self.production_ontime_ratio = ontime_parts/float(total_parts) if total_parts else None
            
            self.production_complete_ratio = None
            if self.production_total_parts and self.production_complete_parts is not None:
                self.production_complete_ratio = self.production_complete_parts/float(self.production_total_parts)
            
            type(self).objects.filter(id=self.id).update(
                production_total_parts=self.production_total_parts,
                production_complete_parts=self.production_complete_parts,
                production_success_parts=self.production_success_parts,
                production_ontime_parts=self.production_ontime_parts,
                production_success_ratio=self.production_success_ratio,
                production_ontime_ratio=self.production_ontime_ratio,
                production_complete_ratio=self.production_complete_ratio,
            )
    
    def save(self, check_fingerprint=True, using=None, *args, **kwargs):
        
        old = None
        
        if self.epoche_of_creation is None:
            self.epoche_of_creation = self.genome.epoche
        
        if self.id:
            try:
                old = type(self).objects.get(id=self.id)
            except type(self).DoesNotExist:
                old = None
            
            self.gene_count = self.genes.all().count()
            
            if do_fingerprint_check() and check_fingerprint and not self.fingerprint_fresh:
                self.fingerprint = self.get_fingerprint()
                self.fingerprint_fresh = True
                
        self.full_clean(check_fingerprint=check_fingerprint)
        
        if self.total_evaluation_seconds is None and self.fitness_evaluation_datetime and self.fitness_evaluation_datetime_start:
            self.total_evaluation_seconds = (self.fitness_evaluation_datetime - self.fitness_evaluation_datetime_start).seconds
        
        if self.epoche is None or self.epoche_of_evaluation != self.epoche.index:
            self.epoche = Epoche.objects.get_or_create(genome=self.genome, index=self.epoche_of_evaluation or self.genome.epoche)[0]
        
        self.complete_ratio = None
        if self.total_parts and self.complete_parts is not None:
            self.complete_ratio = self.complete_parts/float(self.total_parts)
        
        self.production_complete_ratio = None
        if self.production_total_parts and self.production_complete_parts is not None:
            self.production_complete_ratio = self.production_complete_parts/float(self.production_total_parts)
            
        if self.production_fresh:
            if not self.production_evaluation_start_datetime:
                self.production_evaluation_start_datetime = timezone.now()
            if not self.production_evaluation_end_datetime:
                self.production_evaluation_end_datetime = timezone.now()
            
        super(Genotype, self).save(using=using, *args, **kwargs)
        
        if old and not old.fresh and self.fresh:
            production_genotype_changed.send(sender=self, genome=self.genome)
    
    #TODO:Remove? Deprecated due to inefficiency.
    @staticmethod
    def post_save(sender, instance, *args, **kwargs):
        self = instance
        self.genome.save()
    
    def getattr(self, name, default=None):
        q = self.genes.filter(gene__name=name)
        if q.exists():
            return q[0].value
        if default is not None:
            return default
        q = self.genome.genes.filter(name=name)
        if q.exists():
            raise Exception, ('Gene "%s" exists on the genome, but has not '\
                'been set for genotype %s.') % (name, self.id)
        else:
            raise Exception, \
                'No gene "%s" exists in genome %s.' % (name, self.genome.id)
    
    def setattr(self, name, value):
        q = self.genes.filter(gene__name=name)
        if not q.exists():
            raise Exception, 'Genotype does not have a gene named %s' % (name,)
        gg = q[0]
        gg._value = str(value)
        gg.save()
    
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
            accuracy=None,
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
        
#    dependee_name = models.CharField(
#        max_length=1000,
#        editable=False)
#    
#    dependee_value = models.CharField(
#        max_length=1000,
#        editable=False)
#    
#    illegal_value = models.CharField(
#        editable=False,
#        max_length=1000)
    
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
    
#    dependee_gene = models.ForeignKey(
#        'Gene',
#        db_column='dependee_gene_id',
#        related_name='missing_dependents',
#        primary_key=True,
#        blank=False,
#        null=False,
#        editable=False,
#        on_delete=models.DO_NOTHING)
    
    default = models.CharField(
        max_length=1000,
        editable=False)
    
    class Meta:
        managed = False
        db_table = 'django_analyze_genotypegenemissing'
        
    def create(self):
        gene = Gene.objects.get(id=self.gene_id)
        GenotypeGene.objects.create(
            genotype_id=self.genotype_id,
            gene_id=self.gene_id,
            #_value=gene.get_random_value(),
            # Note, we shouldn't a random value, because if a gene is added
            # to formalize an implicit feature, the default represents the
            # assumed value before this formalization. Therefore if we use
            # a random value, we may be changing the behavior of existing
            # genotypes.
            _value=self.default,
        )

class GenotypeGeneManager(models.Manager):
    
    def get_by_natural_key(self, fingerprint, genome_name, gene_name, genome_name2):
        #print 'GenotypeGeneManager:',fingerprint, genome_name, gene_name, genome_name2
        genotype = Genotype.objects.get_by_natural_key(fingerprint, genome_name)
        gene = Gene.objects.get_by_natural_key(gene_name, genome_name2)
        return self.get_or_create(genotype=genotype, gene=gene)[0]
    
class GenotypeGene(BaseModel):
    
    objects = GenotypeGeneManager()
    
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
    
    _value_genome = models.ForeignKey(
        Genome,
        on_delete=models.SET_NULL,
        related_name='genotype_gene_referrers',
        editable=False,
        blank=True,
        null=True)
    
    _value_genotype = models.ForeignKey(
        Genotype,
        on_delete=models.SET_NULL,
        related_name='genotype_genes_referrers',
        editable=False,
        blank=True,
        null=True)
    
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

    natural_key_fields = ('genotype', 'gene')

    def natural_key(self):
        return self.genotype.natural_key() + self.gene.natural_key()
    natural_key.dependencies = ['django_analyze.Genotype', 'django_analyze.Gene']
    
    def __unicode__(self):
        return '<GenotypeGene:%s %s=%s>' \
            % (self.id, self.gene.name, self._value)
    
    @property
    def value(self):
        """
        Converts the internally-stored string value into the true
        logically value based on the gene's type.
        """
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
        elif self.gene.type == c.GENE_TYPE_GENOME:
            #print('value0:',self._value)
            parts = self._value.split(':')
            #print('value1:',self._value)
            if len(parts) == 2:
                return Genotype.objects.get(
                    genome__id=int(parts[0]),
                    id=int(parts[1]))
            else:
                return Genome.objects.get(id=int(parts[0])).production_genotype
        return self._value
    
    @value.setter
    def value(self, v):
        self._value_genome = None
        self._value_genotype = None
        valid_types = set(c.GENE_TYPES)
        valid_types.update([Genome, Genotype])
        if not isinstance(v, tuple(valid_types)):
            raise NotImplementedError, 'Unsupported type: %s' % type(v)
        elif self.gene.type == c.GENE_TYPE_GENOME:
            if isinstance(v, Genome):
                # Implicitly uses the production genotype.
                self._value = str(v.id)
                self._value_genome = v
                self._value_genotype = None
            elif isinstance(v, Genotype):
                # Explicitly uses a genotype.
                self._value = '%i:%i' % (v.genome.id, v.id)
                self._value_genome = v.genome
                self._value_genotype = v
            else:
                raise NotImplementedError
        else:
            self._value = str(v)
            self._value_genome = None
    
#    def is_legal(self):
#        """
#        Returns true if this gene value is allowed to exist in this genotype
#        based on gene dependency rules.
#        Returns false otherwise, implying it should be deleted.
#        """
#        if not self.gene.dependee_gene:
#            # We're not dependent on any other gene, so we're implicitly
#            # allowed to exist.
#            return True
#        q = self.genotype.genes.filter(gene=self.gene.dependee_gene)
#        if not q.count():
#            # We're dependent on a gene that doesn't exist in this genotype,
#            # so we shouldn't exist either.
#            return False
#        elif q[0]._value != self.gene.dependee_value:
#            # We're dependent on an existing gene, but its value differs from
#            # the value we require so we shouldn't exist.
#            return False
#        return True
#    is_legal.boolean = True
        
    def clean(self, *args, **kwargs):
        """
        Called to validate fields before saving.
        Override this to implement your own model validation
        for both inside and outside of admin. 
        """
        #print 'do_validation_check():',do_validation_check()
        if do_validation_check():
            try:
                #print 'self._value:',self._value
                value = str_to_type[self.gene.type](self._value)
                
                if self.gene.type == c.GENE_TYPE_GENOME and value is not None:
                    if len(str(self._value).split(':')) != 2:
                        raise ValidationError({
                            '_value': [
                                ('Value of type genome must be formatted '
                                'like genome_id:genotype_id, not %s.') \
                                    % (self._value)
                            ],
                        })
                    value = value[0]
                
                #print 'value:',value
                if self.gene.values:
                    values = self.gene.get_values_list()
                    if value not in values:
                        raise ValidationError({
                            '_value': [
                                'Value must be one of %s, not %s.' \
                                    % (
                                        ', '.join(map(str, values)),
                                        repr(value),
                                    )
                            ],
                        })
            except ValueError:
                raise ValidationError({
                    '_value': ['Value must be of type %s.' % self.gene.type],
                })
        
        super(GenotypeGene, self).clean(*args, **kwargs)
        
    def full_clean(self, *args, **kwargs):
        return self.clean(*args, **kwargs)
        
    def save(self, using=None, *args, **kwargs):
        self.full_clean()
        super(GenotypeGene, self).save(using=using, *args, **kwargs)
        
    @staticmethod
    def post_save(sender, instance, *args, **kwargs):
        self = instance
        self.gene.update_coverage(auto_update=True)
        Genotype.objects\
            .filter(id=self.genotype_id)\
            .update(fresh=False, fingerprint_fresh=False)
        
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

#TODO:fix signals not being sent after inline parent saved?
def connect_signals():
    #signals.post_save.connect(Genotype.post_save, sender=Genotype)
    signals.post_save.connect(GenotypeGene.post_save, sender=GenotypeGene)
    signals.post_delete.connect(GenotypeGene.post_delete, sender=GenotypeGene)
    signals.post_delete.connect(Gene.post_delete, sender=Gene)

def disconnect_signals():
    #signals.post_save.disconnect(Genotype.post_save, sender=Genotype)
    signals.post_save.disconnect(GenotypeGene.post_save, sender=GenotypeGene)
    signals.post_delete.disconnect(GenotypeGene.post_delete, sender=GenotypeGene)
    signals.post_delete.disconnect(Gene.post_delete, sender=Gene)

connect_signals()
