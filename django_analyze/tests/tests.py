import os
import re
import sys
import datetime
from datetime import timedelta
import time
from multiprocessing import Lock

from django.core.management import _commands, call_command
from django.test import TestCase
from django.utils import timezone

from django_analyze import models
from django_analyze import utils
from django_analyze import constants as c

import warnings
warnings.simplefilter('error', RuntimeWarning)

class Tests(TestCase):
    
    #fixtures = []
    
    def setUp(self):
        # Install the test command; this little trick installs the command
        # so that we can refer to it by name
#        _commands['test_sleeper'] = Sleeper()
        pass
    
    def test_timedprocess(self):
        """
        Confirm a parent process's user-time includes those of all children processes.
        """
        
        def waiter():
            for _ in xrange(10):
                time.sleep(1)
                
        def runner(lock, *args, **kwargs):
            # Do some useless non-nop calculation to constantly consume CPU cycles.
            a = 0
            for _ in xrange(100000000):
                a += 1
                a *= 3
                #time.sleep(0.1) # Do not do this.
#                if not _ % 100000:
#                    lock.acquire()
#                    #print 'running'
#                    sys.stdout.flush()
#                    lock.release()
            
#            lock.acquire()
#            #print 'runner done'
#            sys.stdout.flush()
#            lock.release()
        
        def sub_launcher(lock, *args, **kwargs):
            p = utils.TimedProcess(
                max_seconds=100,
                #target=waiter,
                target=runner,
                args=(lock,),)
            p.start()
            while p.is_alive():
#                lock.acquire()
#                print 'secs1: %s %.02f' % (p.pid, p.get_duration_seconds())
#                sys.stdout.flush()
#                lock.release()
                time.sleep(1)
        
        lock = Lock()
        
        p = utils.TimedProcess(max_seconds=100, target=sub_launcher, args=(lock,))
        p.start()
        expected_seconds = 5
        for _ in xrange(expected_seconds):
#            lock.acquire()
#            print 'secs0: %s %.02f' % (p.pid, p.get_duration_seconds())
#            sys.stdout.flush()
#            lock.release()
            time.sleep(1)
        p.terminate()
        p.join()
        final_time = p.get_duration_seconds()
        #print 'final:',final_time
        self.assertAlmostEqual(final_time, expected_seconds, places=0)
    
    def test_weighted_choice(self):
        """
        Confirm weighted_choice returns elements with a distribution
        proportional to the given weights.
        """
        from collections import defaultdict
        counts = defaultdict(int)
        a_ratio = 1.0/6
        b_ratio = 2.0/6
        c_ratio = 3.0/6
        weights = dict(a=a_ratio, b=b_ratio, c=c_ratio)
        for _ in xrange(10000):
            counts[utils.weighted_choice(weights)] += 1
        total = float(sum(counts.values()))
#        for k,v in counts.iteritems():
#            print v/total, k
        a_count = counts['a']/total
        b_count = counts['b']/total
        c_count = counts['c']/total
        self.assertAlmostEqual(a_count, a_ratio, places=1)
        self.assertAlmostEqual(b_count, b_ratio, places=1)
        self.assertAlmostEqual(c_count, c_ratio, places=1)
        
        # Test corner case where all values have zero-weight.
        counts = defaultdict(int)
        weights = dict(a=0, b=0, c=0)
        for _ in xrange(10000):
            counts[utils.weighted_choice(weights)] += 1
        total = float(sum(counts.values()))
#        for k,v in counts.iteritems():
#            print v/total, k
        a_count = counts['a']/total
        b_count = counts['b']/total
        c_count = counts['c']/total
        
        a_ratio = 1.0/3
        b_ratio = 1.0/3
        c_ratio = 1.0/3
        self.assertAlmostEqual(a_count, a_ratio, places=1)
        self.assertAlmostEqual(b_count, b_ratio, places=1)
        self.assertAlmostEqual(c_count, c_ratio, places=1)
    
    def test_weighted_samples(self):
        from collections import defaultdict
        counts = defaultdict(int)
        a_ratio = 1.0/10
        b_ratio = 2.0/10
        c_ratio = 3.0/10
        d_ratio = 4.0/10
        weights = dict(a=a_ratio, b=b_ratio, c=c_ratio, d=d_ratio)
        
        counts = defaultdict(int)
        for _ in xrange(100000):
            samples = list(utils.weighted_samples(choices=weights, k=2))
#            print samples
            self.assertEqual(len(samples), 2)
            for sample in samples:
                counts[sample] += 1
                
        total = float(sum(counts.values()))
#        for k in sorted(counts.iterkeys()):
#            print counts[k]/total, k
        a_count = counts['a']/total
        b_count = counts['b']/total
        c_count = counts['c']/total
        d_count = counts['d']/total
        
        self.assertAlmostEqual(a_count, a_ratio, places=1)
        self.assertAlmostEqual(b_count, b_ratio, places=1)
        self.assertAlmostEqual(c_count, c_ratio, places=1)
        self.assertAlmostEqual(d_count, d_ratio, places=1)
    
    def test_allowed_genes(self):
        
        genome = models.Genome.objects.create(name='test')
        
        algorithm_gene = models.Gene.objects.create(
            genome=genome,
            name='algorithm',
            type=c.GENE_TYPE_STR,
            values='SGDClassifier,RandomForestClassifier,ExtraTreesClassifier',
            default='SGDClassifier')
        
        # Denote n_estimators as requiring a tree-based algorithm gene value.
        n_estimators_gene = models.Gene.objects.create(
            genome=genome,
            name='n_estimators',
            type=c.GENE_TYPE_INT,
            values='10,20,40,80',
            default='10')
        models.GeneDependency.objects.create(
            gene=n_estimators_gene,
            dependee_gene=algorithm_gene,
            dependee_value='RandomForestClassifier',
            positive=True)
        models.GeneDependency.objects.create(
            gene=n_estimators_gene,
            dependee_gene=algorithm_gene,
            dependee_value='ExtraTreesClassifier',
            positive=True)
        
        # Create a genotype is we can confirm the gene combination is allowed.
        gt1 = models.Genotype.objects.create(genome=genome)
        
        missing = models.GenotypeGeneMissing.objects.filter(genotype=gt1)
        #print [_.gene.name for _ in missing]
        self.assertEqual(missing.count(), 1) # should recommend algorithm gene
        
        gg_algorithm = models.GenotypeGene.objects.create(
            genotype=gt1,
            gene=algorithm_gene,
            _value='ExtraTreesClassifier')
        
        missing = models.GenotypeGeneMissing.objects.filter(genotype=gt1)
        #print [_.gene.name for _ in missing]
        self.assertEqual(missing.count(), 1) # should recommend n_estimators gene
        
        models.GenotypeGene.objects.create(
            genotype=gt1,
            gene=n_estimators_gene,
            _value='10')
        
        missing = models.GenotypeGeneMissing.objects.filter(genotype=gt1)
        #print [_.gene.name for _ in missing]
        self.assertEqual(missing.count(), 0)
        
        gt1_genes = gt1.genes.all()
        #print 'genes:',[_.gene.name for _ in gt1_genes.all()]
        self.assertEqual(gt1_genes.all().count(), 2)
        
        illegal = models.GenotypeGeneIllegal.objects.filter(genotype=gt1)
        self.assertEqual(illegal.count(), 0)
        
        # Change algorithm to break rule, and confirm illegal query catches it.
        gg_algorithm._value = 'SGDClassifier'
        gg_algorithm.save()
        
        illegal = models.GenotypeGeneIllegal.objects.filter(genotype=gt1)
        #print 'illegal:',[_.illegal_gene_name for _ in illegal.all()]
        self.assertEqual(illegal.count(), 1)
    
    def test_genome_gene(self):
        
        # Create client genome.
        g0 = models.Genome(name='g0')
        g0.save()
        g0_test = models.Gene.objects.create(
            genome=g0,
            name='test',
            type=c.GENE_TYPE_INT,
            values='1,2,3',
            default='1')
        
        # Create child genome genotypes.
        g01 = models.Genotype.objects.create(genome=g0)
        models.GenotypeGene.objects.create(genotype=g01, gene=g0_test, _value='1')
        g01.export = True
        g01.save()
        g02 = models.Genotype.objects.create(genome=g0)
        models.GenotypeGene.objects.create(genotype=g02, gene=g0_test, _value='2')
        
        # Set one genotype as the production genotype.
        g0.production_genotype = g02
        g0.save()
        
        # Create parent genome.
        g1 = models.Genome(name='g1')
        g1.save()
        g1_test = models.Gene.objects.create(
            genome=g1,
            name='test',
            type=c.GENE_TYPE_GENOME,
            values=str(g0.id),
            default=str(g0.id))
        
        default = g1_test.get_default()
        #print default
        self.assertEqual(default.id, 2)
        values = g1_test.get_values_list()
        self.assertEqual(set(_.id for _ in values), set([1, 2]))
        #print values
        
        # Create parent genotype pointing to a child genotype.
        g11 = models.Genotype.objects.create(genome=g1)
        g11_test = models.GenotypeGene(genotype=g11, gene=g1_test)
        g11_test.value = g02
        g11_test.save()
        g11.save()
        self.assertEqual(g11_test._value, '%i:%i' % (g0.id, g02.id))
        
        # Confirm the gene pointing to a specific child genotype returns
        # a reference to that genotype.
        child_gt = g11.getattr('test')
        self.assertEqual(child_gt, g02)
    
    def test_normalize_list(self):
        a = [-0.37, -0.08, 0.76, 7.73]
        #print(a)
        b = utils.normalize_list(a)
        #print(b)
        self.assertEqual(
            b,
            [0.0, 0.03580246913580247, 0.13950617283950617, 1.0])
    
    def test_PrefixStream(self):
        from StringIO import StringIO
        
        _fout = StringIO()
        fout = utils.PrefixStream(_fout, prefix='start: ')
        for i in range(10):
            fout.write('\rtest: %i' % i)
            fout.flush()
            #time.sleep(1)
        fout.write('\ndone')
        fout.flush()
        v = _fout.getvalue()
        #print(v)
        self.assertEqual(re.sub(r'.*?\r', '', v), 'start: test: 9\nstart: done')

        _fout = StringIO()
        _stdout = sys.stdout
        try:
            sys.stdout = utils.PrefixStream(
                stream=_fout,
                prefix='start: ')
            print('test1')
            print('test2')
            print('test3')
        finally:
            sys.stdout = _stdout
        v = _fout.getvalue()
#        print(v)
        self.assertEqual(v, 'start: test1\nstart: test2\nstart: test3\n')
    
    def test_chunklist(self):
        x = range(13)
        y = list(utils.chunklist(x, 5))
        #print(y)
        self.assertEqual(y,
            [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12]])
        