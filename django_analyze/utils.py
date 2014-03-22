import os
import random
import sys
import threading
import time

from datetime import datetime
from collections import defaultdict, OrderedDict
from multiprocessing import Process, Queue, Lock, current_process

import psutil

def is_power_of_two(x):
    return (x & (x - 1)) == 0

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
        times = p.get_cpu_times()
        total = times.user + times.system
        if recursive:
            children = list(p.get_children(recursive=True))
            for child in children:
                try:
                    times = child.get_cpu_times()
                    total += times.user + times.system
                except psutil._error.NoSuchProcess:
                    pass
    except psutil._error.NoSuchProcess:
        pass
    return total

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
                        times = self._p.get_cpu_times()
                        self._process_times[self._p.pid] = times.user + times.system
                        children = self._p.get_children(recursive=True)
                        for child in children:
                            times = child.get_cpu_times()
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
        pid0 = os.getpid()
        fout = fout or self.fout
        timeout = False
        last_cpu_usage = None
        if verbose:
            print>>fout
            print>>fout
        while 1:
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
            
            if isinstance(fout, MultiProgress):
                #fout.seconds_until_timeout = self.seconds_until_timeout
                #print 'cpu_usage:',cpu_usage
                fout.update_timeout(self.pid, sut=self.seconds_until_timeout, cpu=cpu_usage)
            elif verbose:
                msg = '%.0f seconds until timeout (target=%s, local=%s, verbose=%s).' \
                    % (self.seconds_until_timeout, self.pid, pid0, verbose)
                print msg
                pass
#                print>>fout, msg
#                fout.flush()

            if not self.is_alive():
                break
            elif self.is_expired:
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
        self.status = Queue()
        self.last_progress_refresh = None
        self.bar_length = 10
        self.pid_counts = {}
        self.refresh_period = 0.5
        
        # Variables used for the child process.
        self.pid = None
        self.current_count = 0
        self.total_count = 0
        self.sub_current = 0
        self.sub_total = 0
        self.seconds_until_timeout = None
        self.eta = None
        self.cpu = None
        
        self.changed = True
    
    def update_timeout(self, pid, sut, cpu):
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
        message = (' '.join(map(str, message))).strip()
        if not message:
            return
#        if not self.pid:
#            return
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
    
    def flush(self):
        """
        Displays all output from all processes in an ordered list.
        This should only be called by the parent process.
        """
        from chroniker.models import Job
        if self.last_progress_refresh and (datetime.now()-self.last_progress_refresh).seconds < self.refresh_period:
            return
        elif self.status.empty():
            return
        
        while not self.status.empty():
            
            # Load item from queue.
            pid, current, total, sub_current, sub_total, eta, sut, cpu, message = self.status.get()
            if message == UPDATE_TIMEOUT:
                new_sut = sut
                new_cpu = cpu
                (current, total, sub_current, sub_total, eta, sut, cpu, message) = (self.progress.get(pid) or (0, 0, 0, 0, None, None, None, ''))
                data = (current, total, sub_current, sub_total, eta, new_sut, new_cpu, message)
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
                self.fout.write('\033[2J\033[H') #clear screen
                if self.title:
                    self.fout.write('%s\n' % self.title)
                # Show last progress message from all processes.
                items = self.progress.items()
            else:
                # Only show the message we just received.
                items = [(pid, data)]
            
            # Display each process line.
            for pid, (current, total, sub_current, sub_total, eta, sut, cpu, message) in items:
                eta = eta or '?'
                sut = sut or '?'
                cpu = '?' if cpu is None else cpu
                sub_status = ''
                if self.hide_complete and current is not None and total is not None and current >= total and pid in self.progress_done:
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
                    percent = '?'
                    total = '?'
                if sub_current and sub_total:
                    sub_status = '(subtask %s of %s) ' % (sub_current, sub_total)
                ts = ''
                if self.show_timestamp:
                    ts = '%s: ' % datetime.now()
                self.fout.write(
                    (('' if self.newline else '\r')+"%s%s [%s] %s of %s %s%s%% eta=%s sut=%s cpu=%s: %s"+('\n' if self.newline else '')) \
                        % (ts, pid, bar, current, total, sub_status, percent, eta, sut, cpu, message))
                if current >= total:
                    self.progress_done.add(pid)
            self.fout.flush()
            self.last_progress_refresh = datetime.now()
        
        # Update job.
        if not self.only_changes or (self.only_changes and self.changed):
            overall_current_count = 0
            overall_total_count = 0
            for pid, (current, total) in self.pid_counts.iteritems():
                overall_current_count += current
                overall_total_count += total
            if overall_total_count:
                Job.update_progress(
                    total_parts_complete=overall_current_count,
                    total_parts=overall_total_count,
                )
                
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
        while self.has_pending_tasks() or self.processes:
            if pid0 != os.getpid():
                return
            
            self.progress.current_count = self.complete_parts
            self.progress.sub_current = 0
            self.progress.sub_total = 0
            self.progress.seconds_until_timeout = None
            self.progress.eta = None
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
        