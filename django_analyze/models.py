import os
import sys
import time
import gc
from StringIO import StringIO
import traceback
from datetime import timedelta
from base64 import b64encode, b64decode
import tempfile
import importlib
import random
from multiprocessing import Process

#from picklefield.fields import PickledObjectField

import django
from django.conf import settings
from django.db import models
from django.db.models import Sum, Count, Max, Min, Q
from django.db.utils import IntegrityError
from django.utils import timezone

from django_materialized_views.models import MaterializedView

import constants as c

from sklearn.externals import joblib

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

class TimedProcess(Process):
    """
    Helper to allow us to time a specific process and determine when
    it has reached a timeout.
    """
    
    daemon = True
    
    def __init__(self, max_seconds, objective=True, fout=None, *args, **kwargs):
        super(TimedProcess, self).__init__(*args, **kwargs)
        self.fout = fout or sys.stdout
        self.objective = objective
        self.t0 = time.clock()
        self.t0_objective = time.time()
        self.max_seconds = float(max_seconds)
        self.t1 = None
        self.t1_objective = None
    
    @property
    def duration_seconds(self):
        if self.objective:
            if self.t1_objective is not None:
                return self.t1_objective - self.t0_objective
            return time.time() - self.t0_objective
        else:
            if self.t1 is not None:
                return self.t1 - self.t0
            return time.clock() - self.t0
        
    @property
    def is_expired(self):
        if not self.max_seconds:
            return False
        return self.duration_seconds >= self.max_seconds
    
    @property
    def seconds_until_timeout(self):
        return max(self.max_seconds - self.duration_seconds, 0)
    
    def start_then_kill(self, verbose=True):
        """
        Starts and then kills the process if a timeout occurs.
        
        Returns true if a timeout occurred. False if otherwise.
        """
        self.start()
        timeout = False
        if verbose:
            print>>self.fout
            print>>self.fout
        while 1:
            time.sleep(1)
            if verbose:
                print>>self.fout, '\r\t%.0f seconds until timeout.' \
                    % (self.seconds_until_timeout,),
                self.fout.flush()
            if not self.is_alive():
                break
            elif self.is_expired:
                if verbose:
                    print>>self.fout
                    print>>self.fout, 'Attempting to terminate expired process %s...' % (self.pid,)
                timeout = True
                self.terminate()
        self.t1 = time.clock()
        self.t1_objective = time.time()
        return timeout

class Predictor(models.Model, MaterializedView):
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
        blank=False,
        null=False)
    
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

class Evolver(models.Model, MaterializedView):
    """
    The top level handle for a problem solved by a genetic algorithm.
    """
    
    name = models.CharField(
        max_length=150,
        blank=False,
        null=False,
        unique=True)
    
    active = models.BooleanField(
        default=True,
        help_text='If checked, this domain will be automatically evolved.')
    
    species_count = models.PositiveIntegerField(
        default=10,
        blank=False,
        null=False,
        help_text='The number of unique species populations maintained.')
    
    species_size = models.PositiveIntegerField(
        default=10,
        blank=False,
        null=False,
        help_text='The number of organisms created within each species.')
    
    epoches = models.PositiveIntegerField(
        default=0,
        blank=False,
        null=False,
        help_text='The number of epoches thus far evaluated.')
    
    class Meta:
        abstract = True
        
    def evaluate(self, gene):
        """
        Calculates the fitness of the gene in solving the problem domain.
        """
        raise NotImplementedError
    
    def crossover(self, geneA, geneB):
        """
        Randomly merges the functionality from two genes.
        Results in the creation of a new gene.
        The originals remain unmodified.
        """
        raise NotImplementedError
    
    def mutate(self, gene):
        """
        Randomly changes a unit of functionality within the gene.
        Results in the creation of a new species.
        The original remains unmodified.
        """
        raise NotImplementedError
    
    def get_operators(self):
        """
        Returns a list of operators to use for creating atomic genes.
        """
        raise NotImplementedError
    
    def get_species(self, gene):
        """
        Calculates the closest species the gene belongs to.
        This should function similar to k-means clustering, in that a gene may
        switch species if doing so is necessary to maintain a number of
        balanced populations equal to species_count.
        """
        raise NotImplementedError

import inspect

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

def register_evaluator(func):
    name = get_method_name(func)
    _evaluators[name] = func

def register_modeladmin_extender(func):
    name = get_method_name(func)
    _modeladmin_extenders[name] = func

def get_evaluators():
    for name in sorted(_evaluators.iterkeys()):
        yield (name, name)

def get_extenders():
    for name in sorted(_modeladmin_extenders.iterkeys()):
        yield (name, name)

class Genome(models.Model):
    """
    All possible parameters of a problem domain.
    """
    
    name = models.CharField(
        max_length=100,
        blank=False,
        null=False,
        unique=True)
    
    evaluator = models.CharField(
        max_length=1000,
        choices=get_evaluators(),
        blank=True,
        help_text='The method to use to evaluate genotype fitness.',
        null=True)
    
#    admin_extender = models.CharField(#TODO:remove?
#        max_length=1000,
#        choices=get_extenders(),
#        blank=True,
#        help_text='Method to call to selectively extend the admin interface.',
#        null=True)
    
    maximum_population = models.PositiveIntegerField(
        default=1000,
        help_text='''The maximum number of genotype records to create.
            If set to zero, no limit will be enforced.
            If delete_inferiors is checked, all after this top amount, ordered
            by fitness, will be deleted.'''
    )
    
    mutation_rate = models.FloatField(
        default=0.1,
        blank=False,
        null=False)
    
    evaluation_timeout = models.PositiveIntegerField(
        default=0,
        blank=False,
        null=False,
        help_text='''The number of seconds to the genotype will allow
            training before forcibly terminating the process.<br/>
            A value of zero means no timeout will be enforced.<br/>
            Note, it's up to the genotype evaluator how this timeout is
            interpreted.
            If a genotype runs multiple evaluation methods internally, this
            may be used on each individual method, not the overall evaluation.
        ''')
    
    epoches = models.PositiveIntegerField(
        default=0,
        blank=False,
        null=False,
        help_text='The number of epoches thus far evaluated.')
    
    delete_inferiors = models.BooleanField(
        default=False,
        help_text='''If checked, all but the top fittest genotypes
            will be deleted. Requires maximum_population to be set to
            a non-zero value.'''
    )
    
    min_fitness = models.FloatField(blank=True, null=True, editable=False)
    
    max_fitness = models.FloatField(blank=True, null=True, editable=False)
    
    def __unicode__(self):
        return self.name
    
    def save(self, *args, **kwargs):
        
        if self.id:
            q = self.genotypes.filter(fitness__isnull=False)\
                .aggregate(Max('fitness'), Min('fitness'))
            self.max_fitness = q['fitness__max']
            self.min_fitness = q['fitness__min']
            
        super(Genome, self).save(*args, **kwargs)
    
    def get_random_genotype(self):
        d = []
        for gene in self.genes.all():
            d.append((gene, gene.get_random_value()))
        genotype = Genotype.objects.create(genome=self)
        GenotypeGene.objects.bulk_create([
            GenotypeGene(genotype=genotype, gene=gene, _value=str(value))
            for gene, value in d
        ])
        genotype.save()
        return genotype
    
    @property
    def pending_genotypes(self):
        return self.genotypes.filter(fresh=False)
    
    def populate(self):
        """
        Creates random genotypes until the maximum limit is reached.
        """
        max_retries = 10
        last_pending = None
        populate_count = 0
        max_populate_retries = 1000
        print 'Populating genotypes...'
        while 1:
            
            # Delete failed and/or corrupted genotypes.
            self.genotypes.filter(fingerprint__isnull=True).delete()
            
            #TODO:only look at fitness__isnull=True if maximum_population=0 or delete_inferiors=False?
            pending = self.pending_genotypes.count()
            if pending >= self.maximum_population:
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
            print '\t\rAttempt %i of %i to create %i, currently %i' % (populate_count, max_populate_retries, self.maximum_population, pending),
            sys.stdout.flush()
            
            last_pending = pending
            for retry in xrange(max_retries):
                try:
                    self.get_random_genotype()
                    #TODO:use crossover and mutation?
                    break
                except IntegrityError:
                    django.db.transaction.rollback()
                    continue
                return
    
    @property
    def evaluator_function(self):
        return _evaluators.get(self.evaluator)
    @property
    def admin_extender_function(self):
        return _modeladmin_extenders.get(self.admin_extender)
    
    def evolve(self, genotype_id=None, populate=True):
        
        # Creates the initial genotypes.
        if populate:
            self.populate()
        
        #print self.evaluator_function
        q = self.pending_genotypes
        if genotype_id:
            q = q.filter(id=genotype_id)
        total = q.count()
        print '%i pending genotypes found.' % (total,)
        for gt in q.iterator():
            self.evaluator_function(gt)
            gt.fitness_evaluation_datetime = timezone.now()
            gt.fresh = True
            gt.save()
    
    def crossover(self):
        todo
        
    def mutate(self):
        todo
        
    def evaluate(self):
        todo

class Gene(models.Model):
    """
    Describes a specific configurable settings in a problem domain.
    """
    
    genome = models.ForeignKey(Genome, related_name='genes')
    
    name = models.CharField(max_length=100, blank=False, null=False)
    
    type = models.CharField(
        choices=c.GENE_TYPE_CHOICES,
        max_length=100,
        blank=False,
        null=False)
    
    values = models.TextField(
        blank=True,
        null=True,
        help_text='''Prefix with "source:" to dynamically load values
            from a module.''')
    
    default = models.CharField(max_length=1000, blank=True, null=True)
    
    min_value = models.CharField(max_length=100, blank=True, null=True)
    
    max_value = models.CharField(max_length=100, blank=True, null=True)
    
    class Meta:
        unique_together = (
            ('genome', 'name'),
        )
        
    def __unicode__(self):
        return self.name
    
    def get_random_value(self):
        values_list = self.get_values_list()
        if values_list:
            return random.choice(values_list)
        elif self.type == c.GENE_TYPE_INT:
            return random.randint(int(self.min_value), int(self.max_value))
        elif self.type == c.GENE_TYPE_FLOAT:
            return random.uniform(float(self.min_value), float(self.max_value))
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
    
    def stale(self):
        return self.filter(fitness__isnull=True)

class Genotype(models.Model):
    """
    A specific configuration for solving a problem.
    """
    
    genome = models.ForeignKey(Genome, related_name='genotypes')
    
    fingerprint = models.CharField(
        max_length=700,
        db_column='fingerprint',
        blank=True,
        null=True,
        db_index=True,
        editable=False,
        help_text='''A unique hash calculated from the gene names and values
            used to detect duplicate genotypes.''')
    
    fingerprint_fresh = models.BooleanField(default=False)
    
    created = models.DateTimeField(auto_now_add=True, editable=False)
    
    fitness = models.FloatField(blank=True, null=True, editable=False)
    
    fitness_evaluation_datetime = models.DateTimeField(blank=True, null=True, editable=False)
    
    mean_evaluation_seconds = models.PositiveIntegerField(blank=True, null=True, editable=False)
    
    mean_absolute_error = models.FloatField(
        blank=True,
        null=True,
        db_index=True,
        editable=False,
        help_text='''The mean-absolute-error measure recorded during
            fitness evaluation.''')
    
    gene_count = models.PositiveIntegerField(
        verbose_name='genes',
        blank=True,
        null=True,
        editable=False)
    
    fresh = models.BooleanField(
        default=False,
        #editable=False,
        db_index=True,
        help_text='If true, indicates this predictor is ready to classify.')
    
    class Meta:
        #abstract = True
        unique_together = (
            ('genome', 'fingerprint'),
        )
        pass
    
    def __unicode__(self):
        return (self.fingerprint or u'') or unicode(self.id)
    
    def save(self, *args, **kwargs):
        
        if self.id:
            
            self.gene_count = self.genes.all().count()
            
            if not self.fingerprint_fresh:
                self.fingerprint = obj_to_hash(tuple(sorted(
                    (gene.gene.name, gene.value)
                    for gene in self.genes.all())
                ))
                self.fingerprint_fresh = True
        
        super(Genotype, self).save(*args, **kwargs)
        self.genome.save()
        
    def getattr(self, name):
        return self.genes.filter(gene__name=name)[0].value
        
    @property
    def as_dict(self):
        return dict((gene.name, gene.value) for gene in self.genes.all())

class GenotypeGene(models.Model):
    
    genotype = models.ForeignKey(Genotype, related_name='genes')
    
    gene = models.ForeignKey(Gene, related_name='genes')

    _value = models.CharField(
        max_length=1000,
        db_column='value',
        verbose_name='value',
        blank=False,
        null=False)
    
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
    
    class Meta:
        ordering = (
            'gene__name',
        )
        unique_together = (
            ('genotype', 'gene'),
        )
        
    def save(self, *args, **kwargs):
        super(GenotypeGene, self).save(*args, **kwargs)
        self.genotype.fingerprint_fresh = False
        self.genotype.save()

#class Gene(models.Model):
#    """
#    Represents a solution to the problem domain.
#    
#    Instances of this class can exist in two general forms:
#    1. atomic: the gene depends on no other genes
#    2. compound: the gene is a collection of other genes
#    
#    The traditional mutate and crossover functions manipulate this class by:
#    1. creating new atomic genes using a dictionary dependent on the domain
#    2. creating new compound genes from atomic or compound genes
#    
#    Since genes may depend on other genes, fitness is propagated throughout
#    the gene hierarchy. This allows the propagation of genes that may be
#    useless in solving the problem directly but are useful tools when combined
#    with other genes.
#    """
#    
#    #evolver = models.ForeignKey(Evolver, related_name='genes')
#    
#    fitness = models.FloatField(blank=True, null=True)
#    
#    fitness_evaluation_datetime = models.DateTimeField(blank=True, null=True)
#    
#    fitness_pending_externally = models.BooleanField(
#        default=False,
#        help_text='''If checked, means we are waiting for the fitness from an
#            external source.''')
#    
#    species = models.PositiveIntegerField(
#        blank=True,
#        null=True,
#        help_text='''The species to which this gene belongs.''')
#    
#    operator = models.TextField(blank=False, null=False)
#    
#    arguments = models.ManyToManyField('self', blank=True)
#    
#    class Meta:
#        abstract = True
