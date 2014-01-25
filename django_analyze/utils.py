import os
import sys
import time
from multiprocessing import Process
from collections import defaultdict
import random

import psutil
        
#TODO:extend the use of Process to distributed architextures?
#http://eli.thegreenplace.net/2012/01/24/distributed-computing-in-python-with-multiprocessing/
class TimedProcess(Process):
    """
    Helper to allow us to time a specific chunk of code and determine when
    it has reached a timeout.
    
    Also, this conveniently allows us to kill the whole thing if it locks up
    or takes too long, without requiring special coding in the target code.
    
    Note, real clock time is used to gauge process time, even though this can
    be misleading. It's tempting to use user time, or the actual cpu time the
    process actually consumed, but this was found to be horribly inaccurate and
    unreliable based on the function implementation. Functions that use
    c-extensions or launch their own processes may register almost
    no cpu consumption because the real calculations are being done elsewhere
    which don't usually register with the normal timing mechanisms.
    
    The downside to this approach is that if the system is under heavy load
    due to some other unrelated process, that may artificially add to its real
    clock time, making it look longer-running than it actually is. For this
    reason, it may not be advisable to make use of multi-processing using more
    processes than there are CPUs on your system, because the excess processes
    will make each other slower than they would normally act, potentially
    effecting fitness if time is used as part of the metric.
    """
    
    daemon = True
    
    #TODO:better way to track CPU time of process including all children?
    def __init__(self, max_seconds, objective=False, recursive=True, fout=None, check_freq=1, *args, **kwargs):
        super(TimedProcess, self).__init__(*args, **kwargs)
        self.fout = fout or sys.stdout
        self.objective = objective
        self.t0 = time.clock()
        self.t0_objective = time.time()
        self.max_seconds = float(max_seconds)
        self.t1 = None
        self.t1_objective = None
        # The number of seconds the process waits between checks.
        self.check_freq = check_freq
        self.recursive = recursive
        self._p = None
        self._process_times = {} # {pid:user_seconds}
        self._last_duraction_seconds = None
    
    def terminate(self, *args, **kwargs):
        if self.is_alive() and self._p:
            # Explicitly kill children since the default terminate() doesn't
            # seem to do this very reliably.
            for child in self._p.get_children():
                # Do one last time check.
                self._process_times[child.pid] = child.get_cpu_times().user
                os.system('kill -9 %i' % (child.pid,))
            # Sum final time.
            self._process_times[self._p.pid] = self._p.get_cpu_times().user
            self._last_duraction_seconds = sum(self._process_times.itervalues())
        return super(TimedProcess, self).terminate(*args, **kwargs)
    
    def get_duration_seconds(self, recursive=None, objective=None):
        """
        Retrieve the number of seconds this process has been executing for.
        
        If process was instantiated with objective=True, then the wall-clock
        value is returned.
        
        Otherwise the user-time is returned.
        If recursive=True is given, recursively finds all child-processed,
        if any, and includes their user-time in the total calculation.
        """
        objective = self.objective if objective is None else objective
        recursive = self.recursive if recursive is None else recursive
        if self.is_alive():
            if objective:
                if self.t1_objective is not None:
                    return self.t1_objective - self.t0_objective
                self._last_duraction_seconds = time.time() - self.t0_objective
            else:
                if recursive:
                    # Note, this calculation will consume much for user
                    # CPU time itself than simply checking clock().
                    # Recommend using larger check_freq to minimize this.
                    # Note, we must store historical child times because child
                    # processes may die, causing them to no longer be included in
                    # future calculations, possibly corrupting the total time.
                    self._process_times[self._p.pid] = self._p.get_cpu_times().user
                    children = self._p.get_children(recursive=True)
                    for child in children:
                        self._process_times[child.pid] = child.get_cpu_times().user
                    #TODO:optimize by storing total sum and tracking incremental changes?
                    self._last_duraction_seconds = sum(self._process_times.itervalues())
                else:
                    if self.t1 is not None:
                        return self.t1 - self.t0
                    self._last_duraction_seconds = time.clock() - self.t0
        return self._last_duraction_seconds
        
    @property
    def is_expired(self):
        if not self.max_seconds:
            return False
        return self.duration_seconds >= self.max_seconds
    
    @property
    def seconds_until_timeout(self):
        return max(self.max_seconds - self.duration_seconds, 0)
    
    def start(self, *args, **kwargs):
        super(TimedProcess, self).start(*args, **kwargs)
        self._p = psutil.Process(self.pid)
    
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

def weighted_choice(choices, get_total=None, get_weight=None):
    """
    A version of random.choice() that accepts weights attached to each
    item, increasing or decreasing the likelyhood that each will be picked.
    
    Paramters:
        
        choices := can be either:
            1. a list of the form `[(item, weight)]`
            2. a dictionary of the form `{item: weight}`
            3. a generator that yields `(item, weight)`
    
        get_total := In some cases with large numbers of items, it may be more
            efficient to track the `total` separately and pass it in at call
            time, and then pass in a custom iterator that lazily looks up the
            item's weight. Depending on your distribution, this should
            consume much less memory than loading all items immediately.
            
    """
    
    def get_iter():
        if isinstance(choices, dict):
            return choices.iteritems()
        return choices
            
    if callable(get_total):
        total = get_total()
#        print '-'*80
#        print 'total:',total
    else:
        total = sum(w for c, w in get_iter())
    r = random.uniform(0, total)
    upto = 0.
    for c in get_iter():
        if get_weight:
            w = get_weight(c)
        else:
            c, w = c
        if upto + w >= r:
            return c
        upto += w
    raise Exception
