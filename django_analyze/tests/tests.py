import os
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
        