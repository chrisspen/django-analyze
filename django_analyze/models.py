import os
import sys
import time
import gc
from StringIO import StringIO
import traceback
from datetime import timedelta
from base64 import b64encode, b64decode
import tempfile

#from picklefield.fields import PickledObjectField

from django.conf import settings
from django.db import models
from django.db.models import Sum, Count, Max, Q

from django_materialized_views.models import MaterializedView

from sklearn.externals import joblib

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
    
    training_seconds = models.PositiveIntegerField(blank=True, null=True)
    
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

class Gene(models.Model):
    """
    Represents a solution to the problem domain.
    
    Instances of this class can exist in two general forms:
    1. atomic: the gene depends on no other genes
    2. compound: the gene is a collection of other genes
    
    The traditional mutate and crossover functions manipulate this class by:
    1. creating new atomic genes using a dictionary dependent on the domain
    2. creating new compound genes from atomic or compound genes
    
    Since genes may depend on other genes, fitness is propagated throughout
    the gene hierarchy. This allows the propagation of genes that may be
    useless in solving the problem directly but are useful tools when combined
    with other genes.
    """
    
    #evolver = models.ForeignKey(Evolver, related_name='genes')
    
    fitness = models.FloatField(blank=True, null=True)
    
    fitness_evaluation_datetime = models.DateTimeField(blank=True, null=True)
    
    fitness_pending_externally = models.BooleanField(
        default=False,
        help_text='''If checked, means we are waiting for the fitness from an
            external source.''')
    
    species = models.PositiveIntegerField(
        blank=True,
        null=True,
        help_text='''The species to which this gene belongs.''')
    
    operator = models.TextField(blank=False, null=False)
    
    arguments = models.ManyToManyField('self', blank=True)
    
    class Meta:
        abstract = True
