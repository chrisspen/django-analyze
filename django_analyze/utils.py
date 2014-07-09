import errno
import multiprocessing
import os
import random
import sys
import threading
import time
import traceback
import warnings

from datetime import datetime, timedelta
from collections import defaultdict, OrderedDict

from ctypes import c_double
from multiprocessing.sharedctypes import Value
from multiprocessing import Pool

import psutil

WAIT_FOR_STALE_ERROR_FN = '/tmp/timedprocess_wait_until_finished_or_stale.txt'

def get_stdout_fn(pid):
    return '/tmp/%i.out' % pid

def get_stderr_fn(pid):
    return '/tmp/%i.err' % pid

def is_power_of_two(x):
    return (x & (x - 1)) == 0

#http://stackoverflow.com/questions/1094841/reusable-library-to-get-human-readable-version-of-file-size
def sizeof_fmt(num):
    for x in ['bytes','KB','MB','GB']:
        if num < 1024.0:
            return "%3.1f%s" % (num, x)
        num /= 1024.0
    return "%3.1f%s" % (num, 'TB')

def normalize_list(lst, new_lower=0, new_upper=1):
    min_value = min(lst)
    max_value = max(lst)
    r = float(max_value - min_value)
    return [((x - min_value) / r)*(new_upper - new_lower) + new_lower for x in lst]

def ilog(n, base):
    """
    Find the integer log of n with respect to the base.

    >>> import math
    >>> for base in range(2, 16 + 1):
    ...     for n in range(1, 1000):
    ...         assert ilog(n, base) == int(math.log(n, base) + 1e-10), '%s %s' % (n, base)
    """
    count = 0
    while n >= base:
        count += 1
        n //= base
    return count

def sci_notation(n, prec=3):
    """
    Represent n in scientific notation, with the specified precision.

    >>> sci_notation(1234 * 10**1000)
    '1.234e+1003'
    >>> sci_notation(10**1000 // 2, prec=1)
    '5.0e+999'
    """
    base = 10
    exponent = ilog(n, base)
    mantissa = n / base**exponent
    return '{0:.{1}f}e{2:+d}'.format(mantissa, prec, exponent)

def get_cpu_usage(pid, recursive=True):
    """
    Returns the total CPU usage in seconds including both user
    and system time.
    """
    total = 0
    try:
        p = psutil.Process(pid)
        times = p.cpu_times()
        total = times.user + times.system
        if recursive:
            children = list(p.children(recursive=True))
            for child in children:
                try:
                    times = child.cpu_times()
                    total += times.user + times.system
                except psutil.NoSuchProcess:
                    pass
    except psutil.NoSuchProcess:
        pass
    return total

#TODO:extend the use of Process to distributed architextures?
#http://eli.thegreenplace.net/2012/01/24/distributed-computing-in-python-with-multiprocessing/
class TimedProcess(multiprocessing.Process):
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
    def __init__(self, max_seconds, objective=False, recursive=True, fout=None, cpu_stall_penalty=None, check_freq=1, *args, **kwargs):
        super(TimedProcess, self).__init__(*args, **kwargs)
        self.fout = fout or sys.stdout
        self.objective = objective
        self.t0 = time.clock()
        self.t0_objective = time.time()
        self.max_seconds = float(max_seconds)
        self.t1 = None
        self.t1_objective = None
        self.timeout = None # Set upon process exit, True=process timed out, False=otherwise
        # The number of seconds the process waits between checks.
        self.check_freq = check_freq
        self.recursive = recursive
        self._p = None
        self._process_times = {} # {pid:user_seconds}
        self._last_duraction_seconds = None
        
        # Create shared object for transmitting seconds-until-timeout value.
        self._kwargs['sut'] = self.sut = Value('d', -1, lock=True)
        
        # Number of seconds between status checks.
        self.wait_period = 5
        
        # If no change in cpu time is detected between monitoring loops,
        # this is the number of seconds that will be deducted from the
        # remaining seconds until expiration.
        if cpu_stall_penalty is None:
            self.cpu_stall_penalty = self.wait_period
        else:
            self.cpu_stall_penalty = cpu_stall_penalty
        self.total_cpu_stall_penalty = 0
        
        self._duration_seconds = 0
    
    def terminate(self, *args, **kwargs):
        if self.is_alive() and self._p:
            # Explicitly kill children since the default terminate() doesn't
            # seem to do this very reliably.
            for child in self._p.children():
                # Do one last time check.
                self._process_times[child.pid] = child.cpu_times().user
                os.system('kill -9 %i' % (child.pid,))
            # Sum final time.
            self._process_times[self._p.pid] = self._p.cpu_times().user
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
        class Skip(Exception):pass
        objective = self.objective if objective is None else objective
        recursive = self.recursive if recursive is None else recursive
        ret = None
        try:
            if self.is_alive():
                if objective:
                    if self.t1_objective is not None:
                        ret = self.t1_objective - self.t0_objective
                        raise Skip
                    self._last_duraction_seconds = time.time() - self.t0_objective
                else:
                    if recursive:
                        # Note, this calculation will consume much for user
                        # CPU time itself than simply checking clock().
                        # Recommend using larger check_freq to minimize this.
                        # Note, we must store historical child times because child
                        # processes may die, causing them to no longer be included in
                        # future calculations, possibly corrupting the total time.
                        times = self._p.cpu_times()
                        self._process_times[self._p.pid] = times.user + times.system
                        children = self._p.children(recursive=True)
                        for child in children:
                            times = child.cpu_times()
                            self._process_times[child.pid] = times.user + times.system
                        #TODO:optimize by storing total sum and tracking incremental changes?
                        self._last_duraction_seconds = sum(self._process_times.itervalues())
                    else:
                        if self.t1 is not None:
                            ret = self.t1 - self.t0
                            raise Skip
                        self._last_duraction_seconds = time.clock() - self.t0
        except Skip:
            pass
        if ret is None:
            ret = self._last_duraction_seconds or 0
        ret += self.total_cpu_stall_penalty
        self._duration_seconds = max(self._duration_seconds, ret)
        return ret
        
    @property
    def is_expired(self):
        if not self.max_seconds:
            return False
        return self.get_duration_seconds() >= self.max_seconds
    
    @property
    def seconds_until_timeout(self):
        return max(self.max_seconds - self.get_duration_seconds(), 0)
    
    def start(self, *args, **kwargs):
        super(TimedProcess, self).start(*args, **kwargs)
        self._p = psutil.Process(self.pid)
    
    def start_then_kill(self, verbose=True, block=True, fout=None):
        """
        Starts and then kills the process if a timeout occurs.
        
        Returns true if a timeout occurred. False if otherwise.
        """
        self.start()
        if block:
            return self.wait_until_finished_or_stale(verbose=verbose, fout=fout)
        else:
            thread = threading.Thread(
                target=self.wait_until_finished_or_stale,
                kwargs=dict(verbose=verbose, fout=fout))
            thread.daemon = True
            thread.start()
            return thread
    
    def wait_until_finished_or_stale(self, verbose=True, fout=None):
        """
        Blocks until the unlying process exits or exceeds the predefined
        timeout threshold and is killed.
        """
        _stderr = sys.stderr
        try:
            # Set child with lowest possible priority.
            #self._p.nice(15) # low priority, max lowest is 20
            p = psutil.Process(self.pid)
            p.nice(20)
            #p.ionice(ioclass=psutil.IOPRIO_CLASS_IDLE, value=7)
            p.ionice(ioclass=psutil.IOPRIO_CLASS_IDLE)
            
            # Redirect thread errors to separate file for easier review.
            sys.stderr = open(WAIT_FOR_STALE_ERROR_FN, 'a')
            
            pid0 = os.getpid()
            fout = fout or self.fout
            timeout = False
            last_cpu_usage = None
            if verbose:
                print>>fout
                print>>fout
            sut = 1e9999999999999999
            while 1:
                
                #TODO:remove
#                import random
#                if random.random() > 0.5:
#                    raise Exception, 'Error: wait_until_finished_or_stale'
                sut = min(sut, self.seconds_until_timeout)
                
                if pid0 != os.getpid():
                    print '~'*80
                    print 'Ending wait_until because pid stale:%s %s' % (pid0, os.getpid())
                    return
                time.sleep(self.wait_period)
                #assert isinstance(fout, MultiProgress), 'Fout is of type %s' % (type(fout),)
                
                cpu_usage = get_cpu_usage(pid=self.pid)
                #print last_cpu_usage,cpu_usage,last_cpu_usage == cpu_usage
                if last_cpu_usage is not None and last_cpu_usage == cpu_usage:
                    #print 'adding penalty:',self.total_cpu_stall_penalty
                    self.total_cpu_stall_penalty += self.cpu_stall_penalty
                last_cpu_usage = cpu_usage
                
                # Transmit seconds-until-timeout.
                with self.sut.get_lock():
                    self.sut.value = sut#self.seconds_until_timeout if self.seconds_until_timeout is not None else -1.0
                
                assert isinstance(fout, MultiProgress), 'Fout is of type %s' % (type(fout),)
                if isinstance(fout, MultiProgress):
                    #fout.seconds_until_timeout = self.seconds_until_timeout
                    #print 'cpu_usage:',self.pid,cpu_usage
                    #print 'sut for pid %s is %s' % (self.pid, sut)
                    #fout.show(('~'*80)+'\nself.pid:',self.pid,'os.getpid():',os.getpid(),'sut:',self.seconds_until_timeout,'\n'+('~'*80))
                    fout.update_timeout(
                        self.pid,
                        sut=sut,
                        cpu=cpu_usage)
                elif verbose:
                    msg = '%.0f seconds until timeout (target=%s, local=%s, verbose=%s).' \
                        % (self.seconds_until_timeout, self.pid, pid0, verbose)
                    print msg
                    pass
    #                print>>fout, msg
    #                fout.flush()
    
                if not self.is_alive():
                    break
                #elif self.is_expired:
                elif sut <= 0:
                    if verbose:
                        msg = 'Attempting to terminate expired process %s...' % (self.pid,)
                        if isinstance(fout, MultiProgress):
                            fout.write(msg)
                        else:
                            print>>fout
                            print>>fout, msg
                    timeout = True
                    self.terminate()
                    break
            self.t1 = time.clock()
            self.t1_objective = time.time()
            self.timeout = timeout
            return timeout
        except Exception, e:
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
        finally:
            sys.stderr = _stderr

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
    
    Note, this assumes all weights are >= 0.0.
    Negative weights may throw an exception.
    
    """
    
    def get_iter():
        if isinstance(choices, dict):
            return choices.iteritems()
        return choices
            
    if callable(get_total):
        total = get_total()
    else:
        total = sum(w for c, w in get_iter())
    
    # If no non-zero weights given, then just use a uniform distribution.
    if not total:
        return random.choice(list(get_iter()))[0]
        
    r = random.uniform(0, total)
    upto = 0.
    for c in get_iter():
        if get_weight:
            w = get_weight(c)
        else:
            c, w = c
        if w < 0:
            raise Exception, 'Invalid negative weight: %s' % (w,)
        if upto + w >= r:
            return c
        upto += w
    raise Exception, 'Unable to make weighted choice: total=%s, choices=%s' % (total, choices,)

def weighted_samples(choices, k=1):
    """
    Randomly selects from weighted choices without replacement.
    
    Choices must be a dictionary of the form {item:weight}.
    """
    if isinstance(choices, list):
        choices = dict(choices)
    assert isinstance(choices, dict)
    choices = choices.copy()
    assert k >= 1, 'K must be greater or equal to 1.'
    #assert k <= len(choices)
    for _ in xrange(k):
        if not choices:
            return
        item = weighted_choice(choices)
        del choices[item]
        yield item

UPDATE_TIMEOUT = '!UPDATE_TIMEOUT!'

def pid_exists(pid):
    """
    Returns true if the process associated with the given PID is still running.
    Returns false otherwise.
    """
    try:
        pid = int(pid)
    except ValueError:
        return False
    except TypeError:
        return False
    if pid < 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError, e:
        return e.errno == errno.EPERM
    else:
        return True

def calculate_eta(start_datetime, start_count, current_count, total_count):
    """
    Returns the datetime when the given process will likely complete, assuming
    a relatively linear projection of the current progress.
    """
    assert start_count >= 0, 'Invalid start_count: %s' % (start_count,)
    assert current_count >= 0, 'Invalid current_count: %s' % (current_count,)
    assert total_count >= 0, 'Invalid total_count: %s' % (total_count,)
    assert isinstance(start_datetime, datetime)
    if not total_count:
        return
    now_datetime = datetime.now()
    ran_parts = current_count - start_count
    ran_seconds = (now_datetime - start_datetime).total_seconds()
    remaining_parts = total_count - current_count - start_count
    if not ran_parts:
        return
    remaining_seconds = ran_seconds/float(ran_parts)*remaining_parts
    eta = now_datetime + timedelta(seconds=remaining_seconds)
    return eta

class MultiProgress(object):
    """
    Helper class for organizing progress reporting among multiple processes.
    """
    
    def __init__(self, **kwargs):
        
        self.__dict__.update(dict(
            
            # If true, causes flush() to clear the screen before output.
            clear=True,
            
            # If true, appends a newline to all lines.
            newline=True,
            
            # The true file descriptor where output will be sent.
            fout=sys.stdout,
            
            # A string to show before all output.
            # Only really useful when clear=True.
            title=None,
            
            # If true, causes flush() to not display processes that report
            # 100% completion.
            hide_complete=True,
            
            # If true, causes flush() to only display output when something
            # changes.
            only_changes=True,
            
            show_timestamp=True,
            
        ), **kwargs)
        
        self.progress = OrderedDict()
        self.progress_done = set()
        #self.status = multiprocessing.Queue()#throws error if used from Pool()
        manager = multiprocessing.Manager()
        self.status = manager.Queue()
        #self.lock = multiprocessing.Lock()#throws error if used from Pool()
        self.lock = manager.Lock()
        self.outgoing = {} # {pid:Queue()}
        self.last_progress_refresh = None
        self.bar_length = 10
        self.pid_counts = {}
        self.refresh_period = 0.5
        self.pid_start_times = {} # {pid:(start_datetime,start_current_count)}
        
        # Variables used for the child process.
        self.pid = None
        self.current_count = 0
        self.total_count = 0
        self.sub_current = 0
        self.sub_total = 0
        self.seconds_until_timeout = None
        self.eta = None
        self.cpu = None
        
        self.to_children = {} # {parent_pid:[child_pid]}
        self.to_parent = {} # {child_pid:parent_pid}
        
        self.changed = True
    
    def reinit(self):
        if self.pid != os.getpid():
            self.pid = os.getpid()
            self.current_count = 0
            self.total_count = 0
            self.sub_current = 0
            self.sub_total = 0
            self.seconds_until_timeout = None
            self.eta = None
            self.cpu = None
    
    def register_pid(self, pid):
        if not pid_exists(pid):
            if pid in self.to_children:
                del self.to_children[pid]
            if pid in self.to_parent:
                del self.to_parent[pid]
            if pid in self.pid_start_times:
                del self.pid_start_times[pid]
            return
        if pid in self.pid_start_times:
            return
        self.pid_start_times[pid] = (datetime.now(), 0)
        try:
            p = psutil.Process(pid)
            parent = p.parent()
            if parent:
                self.to_children.setdefault(parent.pid, set())
                self.to_children[parent.pid].add(pid)
                self.to_parent[pid] = parent.pid
            children = p.children()
            for child in children:
                self.to_parent[child.pid] = pid
        except psutil.NoSuchProcess:
            pass
    
    def show(self, *args):
        try:
            self.lock.acquire()
            s = ' '.join(map(str, args))
            print s
        finally:
            self.lock.release()
    
    def update_timeout(self, pid, sut=None, cpu=None):
        """
        A specialized version of write() that updates the
        seconds_until_timeout value for a specific process,
        usually from outside the process.
        """
        self.status.put([
            pid,
            None,
            None,
            None,
            None,
            None,
            sut,
            cpu,
            UPDATE_TIMEOUT,
        ])
        time.sleep(0.01)
    
    def write(self, *message):
        self.pid = os.getpid()
        self.reinit()
        message = (' '.join(map(str, message))).strip()
        if not message:
            return
        self.status.put([
            self.pid,
            self.current_count,
            self.total_count,
            self.sub_current,
            self.sub_total,
            self.eta,
            self.seconds_until_timeout,
            self.cpu,
            message,
        ])
        time.sleep(0.01)
    
    def flush(self, *args, **kwargs):
        self.flush_streaming(*args, **kwargs)
        #TODO:support switch
        #self.flush_stacked(*args, **kwargs)
        
    def flush_streaming(self, force=True):
        try:
            self.lock.acquire()
            while not self.status.empty():
                pid, current, total, sub_current, sub_total, eta, sut, cpu, message = self.status.get()
                
                self.register_pid(pid)
                
                start_datetime, start_count = self.pid_start_times.get(pid, (datetime.now(), 0))
                current = current or 0
                total = total or 0
                
                eta = eta or calculate_eta(
                    start_datetime=start_datetime,
                    start_count=start_count,
                    current_count=current,
                    total_count=total,
                )
                if eta:
                    eta = eta.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    eta = '????-??-?? ??:??:??'
                
                # If CPU not given, then attempt to calculate it.
                if cpu is None:
                    cpu = get_cpu_usage(pid=pid)
                if cpu is None:
                    cpu = '???.??'
                else:
                    cpu = '%06.2f' % (cpu,)
                    
                
                # If the process completed, and we've already shown its last
                # message, then don't show it anymore.
                if not pid_exists(pid) and self.hide_complete \
                and current is not None and total is not None \
                and current >= total and pid in self.progress_done:
                    del self.progress[pid]
                    continue
                
                elif total:
                    self.pid_counts[pid] = (current, total)
                    percent = current/float(total)
                    percent = int(percent * 100)
                else:
                    percent = 0
                    
                sub_status = ''
                if sub_current and sub_total:
                    sub_status = '(subtask %s of %s) ' % (sub_current, sub_total)
                    
                ts = ''
                if self.show_timestamp:
                    ts = '%s ' % datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    
                #alive = pid_exists(pid)
                try:
                    alive = True
                    p = psutil.Process(pid)
                except psutil.NoSuchProcess:
                    alive = False
                    continue
                
                try:
                    parent = p.parent()
                    if parent is not None:
                        parent = parent.pid
                except psutil.NoSuchProcess:
                    parent = None
                    
                if sut is None or sut == float('inf'):
                    sut = '?'
                else:
                    sut = '%03i' % sut
                
                template = "{ts}{parent}->{child} {current:04d} of {total:04d} {sub_status}{percent:03.0f}% eta={eta} sut={sut} cpu={cpu}: {message}\n"
#                    print template
                data = dict(
                    ts=ts,
                    parent=parent,
                    child=pid,
                    current=current,
                    total=total,
                    sub_status=sub_status,
                    percent=percent,
                    eta=eta,
                    sut=sut,
                    cpu=cpu,
                    message=message,
                )
                self.fout.write(template.format(**data))
            
        finally:
            self.lock.release()
        
    def flush_stacked(self, force=True):
        """
        Displays all output from all processes in an ordered list.
        This should only be called by the parent process.
        """
        
        if not force:
            if self.last_progress_refresh and (datetime.now()-self.last_progress_refresh).seconds < self.refresh_period:
                return
            elif self.status.empty():
                return
        
        change = False
        items = []
        start_readtime = datetime.now()
        # Read until the queue is empty or until we timeout (in case processes
        # are writing to the queue faster than we can clear it).
        #TODO:how should we handle queue items we don't get to?
        #option1: if we clear them, we risk losing notices from procesess
        #option2: if we don't clear them, we risk a memory leak where our queue grows unrestricted and never gets cleared
        #option3: warn user that queue is growing too fast, implying that processes are logging too much data, and should be changed to be less verbose
        while not self.status.empty() and start_readtime > (datetime.now() - timedelta(seconds=1)):
            #print 'reading status queue'
            change = True
            
            # Load item from queue.
            pid, current, total, sub_current, sub_total, eta, sut, cpu, message = self.status.get()
            (old_current, old_total, old_sub_current, old_sub_total, old_eta, old_sut, old_cpu, old_message) = (self.progress.get(pid) or (0, 0, 0, 0, None, None, None, ''))
            
            self.register_pid(pid=pid)
            
            # Prevent certain values from being reset.
            if cpu is None:
                cpu = old_cpu
            if sut is None:
                sut = old_sut
            
            if pid not in self.pid_start_times:
                self.pid_start_times[pid] = (datetime.now(), current)
                
            if message == UPDATE_TIMEOUT:
                # Only update CPU and SUT.
                current = old_current
                total = old_total
                sub_current = old_sub_current
                sub_total = old_sub_total
                eta = old_eta
                message = old_message
                data = (current, total, sub_current, sub_total, eta, sut, cpu, message)
                self.progress[pid] = data
                self.changed = True
            else:
                data = (current, total, sub_current, sub_total, eta, sut, cpu, message)
                if self.progress.get(pid) != data:
                    self.changed = True
                if self.only_changes and not self.changed:
                    continue
                self.progress[pid] = data
            
            # Prepare screen.
            if self.clear:
                # Show last progress message from all processes.
                items = self.progress.items()
            else:
                # Only show the message we just received.
                #items = self.progress.items()
                items = [(pid, data)]
        
        if not self.status.empty():
            warnings.warn('Unable to clear status queue in alotted time. '\
                'This may mean processes are writing to the queue faster '\
                'than we can read from it. Ensure your processes are not '\
                'being excessively verbose.')
        
        def cmp_pids(pid1, pid2):
            """
            Returns -1 if pid1 is parent of pid2.
            Returns +1 if pid2 is parent of pid1.
            Return 0 if pids are not directly related.
            """
#            parent1 = psutil.Process(pid1)
#            ppid1 = None if parent1 is None else parent1.pid
#            parent2 = psutil.Process(pid2)
#            ppid2 = None if parent2 is None else parent2.pid
#            if pid1 == ppid2:
#                return -1
#            elif pid2 == ppid1:
#                return +1
#            elif ppid1 == ppid2 and pid1 < pid2:
#                return -1
#            elif ppid1 == ppid2 and pid1 > pid2:
#                return +1
#            return 0
            if pid1 == self.to_parent.get(pid2):
                return -1
            elif pid2 == self.to_parent.get(pid1):
                return +1
            return cmp(pid1, pid2)
        
        if force:
            change = True
            if self.clear:
                items = self.progress.items()
        
        # Display each process line.
        if change and items:
            try:
#                print 'acquiring lock'
                self.lock.acquire()
#                print 'lock acquired'
                #self.fout.write(('-'*80)+str(len(self.progress))+'\n')
                last_pid = None
                pid_stack = []
                indent = 0
                max_total = 0
                max_sub_total = 0
                max_sut = 0
                for pid, (current, total, sub_current, sub_total, eta, sut, cpu, message) in items:
                    max_total = max(max_total, total)
                    max_sub_total = max(max_sub_total, sub_total)
                    max_sut = max(max_sut, sut)
                if self.clear:
                    self.fout.write('\033[2J\033[H') #clear screen
                    if self.title:
                        self.fout.write('%s\n' % self.title)
#                print 'sorting items'
                items = sorted(items, cmp=cmp_pids)
#                print 'showing items'
                for pid, (current, total, sub_current, sub_total, eta, sut, cpu, message) in items:
                    
                    # If ETA not given, then attempt to calculate it.
                    if pid in self.pid_start_times:
                        start_datetime, start_count = self.pid_start_times[pid]
                    else:
                        start_datetime, start_count = datetime.now(), 0
                        
                    start_count = start_count or 0
                    current = current or 0
                    total = total or 0
                    
                    eta = eta or calculate_eta(
                        start_datetime=start_datetime,
                        start_count=start_count,
                        current_count=current,
                        total_count=total,
                    )
                    if eta:
                        eta = eta.strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        eta = '????-??-?? ??:??:??'
                    
                    # If CPU not given, then attempt to calculate it.
                    if cpu is None:
                        cpu = get_cpu_usage(pid=pid)
                    if cpu is None:
                        cpu = '???.??'
                    else:
                        cpu = '%06.2f' % (cpu,)
                        
                    sub_status = ''
                    
                    # If the process completed, and we've already shown its last
                    # message, then don't show it anymore.
                    if not pid_exists(pid) and self.hide_complete \
                    and current is not None and total is not None \
                    and current >= total and pid in self.progress_done:
                        del self.progress[pid]
                        continue
                    
                    elif total:
                        self.pid_counts[pid] = (current, total)
                        percent = current/float(total)
                        bar = ('=' * int(percent * self.bar_length)).ljust(self.bar_length)
                        percent = int(percent * 100)
                    else:
                        percent = 0
                        bar = ('=' * int(percent * self.bar_length)).ljust(self.bar_length)
                        #percent = '?'
                        #total = '?'
                    if sub_current and sub_total:
                        sub_status = '(subtask %s of %s) ' % (sub_current, sub_total)
                        
                    ts = ''
                    if self.show_timestamp:
                        ts = '%s ' % datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        
                    #alive = pid_exists(pid)
                    try:
                        alive = True
                        p = psutil.Process(pid)
                    except psutil.NoSuchProcess:
                        alive = False
                        continue
                    
                    try:
                        parent = p.parent()
                        if parent is not None:
                            parent = parent.pid
                    except psutil.NoSuchProcess:
                        parent = None
                        
                    if sut is None or sut == float('inf'):
                        sut = '?'*len(str(max_sut))
                    else:
                        sut = ("%0"+str(len(str(max_sut)))+"i") % sut
                    
                    template = "{start}{ts}{parent}->{child} {current:0"+str(len(str(max_total)))+"d} of {total:0"+str(len(str(max_total)))+"d} {sub_status}{percent:03.0f}% eta={eta} sut={sut} cpu={cpu}: {message}{end}"
#                    print template
                    data = dict(
                        start=('' if self.newline else '\r'),
                        ts=ts,
                        parent=parent,
                        child=pid,
                        current=current,
                        total=total,
                        sub_status=sub_status,
                        percent=percent,
                        eta=eta,
                        sut=sut,
                        cpu=cpu,
                        message=message,
                        end=('\n' if self.newline else ''),
                    )
#                    print data
                    self.fout.write(template.format(**data))
                    if current >= total:
                        self.progress_done.add(pid)
                        
                    last_pid = pid
                self.fout.flush()
                self.last_progress_refresh = datetime.now()
            
            finally:
                self.lock.release()
                
        self.changed = False

class ProcessFactory(object):
    """
    Helper task for launching and managing multiple processes.
    """
    
    def __init__(self,
        max_processes=1,
        has_pending_tasks=None,
        get_next_process=None,
        handle_process_cleanup=None,
        progress=None,
        flush_progress=True):
        
        self.max_processes = max_processes
        self.processes = []
        self.complete_parts = 0
        self.progress = progress or MultiProgress()
        self._has_pending_tasks = has_pending_tasks
        self._get_next_process = get_next_process
        self._handle_process_cleanup = handle_process_cleanup
        self.flush_progress = flush_progress
    
    def has_pending_tasks(self):
        """
        Returns the integer count of remaining tasks to run.
        """
        if callable(self._has_pending_tasks):
            return self._has_pending_tasks(self)
        raise NotImplementedError
    
    def get_next_process(self):
        """
        Returns the next process instance to run.
        """
        if callable(self._get_next_process):
            return self._get_next_process(self)
        raise NotImplementedError
    
    def handle_process_cleanup(self, process):
        """
        Called with the process that just terminated.
        """
        if callable(self._handle_process_cleanup):
            return self._handle_process_cleanup(self, process)
        #raise NotImplementedError
    
    def run(self):
        pid0 = os.getpid()
        total = self.has_pending_tasks()
        self.progress.pid = os.getpid()
        self.progress.total_count = total
        self.progress.seconds_until_timeout = None
        self.progress.eta = None
        self.progress.sub_current = 0
        self.progress.sub_total = 0
        while self.has_pending_tasks() or self.processes:
            if pid0 != os.getpid():
                return
            
            self.progress.current_count = self.complete_parts
            pid_str = ', '.join(str(_.pid) for _ in self.processes)
            self.progress.write(
                'SUMMARY: %i running processes (%s) with %i pending tasks left.' \
                    % (len(self.processes), pid_str, self.has_pending_tasks()))
            
            # Check for complete processes.
            for process in list(self.processes):
                if process.is_alive():
                    continue
                self.processes.remove(process)
                self.complete_parts += 1
                self.handle_process_cleanup(process)
            
            # Start another processes.
            if self.has_pending_tasks() and len(self.processes) < self.max_processes:
                process = self.get_next_process()
                assert isinstance(process, Process), \
                    'Process is of type %s but must be of type %s.' \
                        % (type(process), Process)
                process.start_then_kill(block=False, fout=self.progress)
                self.processes.append(process)
            
            # Output progress messages.
            if self.flush_progress:
                self.progress.flush()
            time.sleep(1)

        self.progress.current_count = total
        self.progress.total_count = total
        self.progress.write(
            'SUMMARY: %i running processes with %i pending tasks left.' \
                % (len(self.processes), self.has_pending_tasks()))
        self.progress.flush()
        #TODO:fix? sometimes results in IOError: [Errno 32] Broken pipe?
        time.sleep(1)

def iter_elements(element_sets, rand=False):
    """
    Iterates over all combination of set elements.
    
    Will yield reduce(int.__mul__, [len(_) for _ in element_sets]) elements.
    
    If rand is True, the order of the yielded lists will be random.
    """
    element_sets = list(element_sets)
    if not element_sets:
        return
    current_set = element_sets.pop(0)
    if rand:
        current_set = list(current_set)
        random.shuffle(current_set)
    for el in current_set:
        if element_sets:
            for lst in iter_elements(element_sets, rand=rand):
                lst = [el] + lst
                yield lst
        else:
            yield [el]

class MemoryThread(threading.Thread):
    """
    Records the current process's memory usage every N seconds.
    """
    
    def __init__(self, period=30, *args, **kwargs):
        super(MemoryThread, self).__init__(*args, **kwargs)
        self.pid = None
        self.memory_history = []
        self.stop = False
        self.daemon = True
        self.period = period
    
    def run(self):
        self.pid = os.getpid()
        p = psutil.Process(self.pid)
        while not self.stop:
            if os.getpid() != self.pid:
                # Kill ourselves if we've been cloned.
                return
            self.memory_history.append(p.get_memory_info().rss)
            print 'memory:',sizeof_fmt(self.memory_history[-1])
            time.sleep(self.period)

class PrefixStream(object):
    
    def __init__(self, stream, prefix, *args, **kwargs):
        self.stream = stream
        self.prefix = str(prefix)
        self.newline = True
    
    def write(self, *args):
        prefix = self.prefix
        s = ' '.join(map(str, args))
#        s = s.strip()
        
        if s.startswith('\r'):
            prefix = '\r' + prefix
            s = s[1:]
            
        if s.startswith('\n') and s.strip():
            prefix = '\n' + prefix
            s = s[1:]
            
        if self.newline:
            #print>>sys.stderr, repr(prefix)
            self.stream.write(prefix)
        self.stream.write(s)
        self.newline = s and (s[-1] == '\n' or prefix.startswith('\r'))
    
#    def writelines(self, *args):
#        self.stream.writelines(*args)
    
    def flush(self):
        self.stream.flush()
    
    def fileno(self):
        return self.stream.fileno()
    
    def close(self):
        self.stream.close()

def chunklist(lst, length):
    return (lst[0+i:length+i] for i in range(0, len(lst), length))
    