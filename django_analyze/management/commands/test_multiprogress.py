
from django.core.management.base import BaseCommand

from optparse import make_option

from django_analyze.utils import ProcessFactory, TimedProcess

class Command(BaseCommand):
    help = 'Tests reporting progress from multiple processes.'
    
    option_list = BaseCommand.option_list + (
#        make_option('--seconds',
#            dest='seconds',
#            default=60,
#            help='The number of total seconds to count up to.'),
        )
    
    def handle(self, *args, **options):
        #seconds = int(options['seconds'])
        
        tasks = range(10)
        
        def wait_task(total, fout):
            import time, random, os
            fout.pid = os.getpid()
            fout.current_count = 0
            fout.total_count = total
            fout.sub_current = 0
            fout.sub_total = 0
            fout.seconds_until_timeout = None
            fout.eta = None
            try:
                for _ in xrange(total):
                    fout.current_count = _ + 1
                    fout.write('%i of %i' % (fout.current_count, total))
                    time.sleep(random.randint(1, 2))
            except Exception, e:
                fout.write('Error: %s' % e)
            else:
                fout.write('Done!')
        
        def has_pending_tasks(factory):
            return len(tasks)
        
        def get_next_process(factory):
            id = tasks.pop(0)
            process = TimedProcess(
                max_seconds=1000000,
                target=wait_task,
                kwargs=dict(total=10, fout=factory.progress),
            )
            return process
        
        factory = ProcessFactory(
            max_processes=8,
            has_pending_tasks=has_pending_tasks,
            get_next_process=get_next_process,
        )
        factory.run()
        