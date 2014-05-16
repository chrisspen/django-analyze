import re
import sys
import time
from datetime import datetime, timedelta
import commands
from pprint import pprint

import django
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db.models import Q
from django.core import serializers
from django.db.models.loading import get_model
from django.db import transaction, connection
from django.utils import simplejson

from optparse import make_option

from django_analyze import models
from django_analyze.yajl_simple import YajlContentBuilder, YajlParser

def flatten_list(lst):
    new = []
    for _ in lst:
        if isinstance(_, list):
            new.extend(flatten_list(_))
        else:
            new.append(_)
    return new

def find_object_pk(model, obj, data):
    """
    Finds the existing record associated with the natural key.
    """
    assert hasattr(model, 'natural_key_fields'), \
        'Model %s must define natural_key_fields.' % (model.__name__,)
    assert hasattr(model.objects, 'get_by_natural_key'), \
        'Model manager for %s must implement get_by_natural_key(*args, **kwargs).' % (model.__name__,)
    #print 'natural key:',obj.natural_key()
    keys = [data['fields'][_] for _ in model.natural_key_fields]
    keys = flatten_list(keys)
    obj = model.objects.get_by_natural_key(*keys)
    if hasattr(obj, 'pk'):
        return obj.pk
    return obj.id

class ContentBuilderOnMap(YajlContentBuilder):
    """
    Special Yajl parser to incrementally load Django's JSON fixtures,
    since the built-in Django loaddata command is too horribly inefficient
    to handle large files.
    
    Since Django fixtures represent instances as a list of maps, this means
    that whenever the parser encounters the end of a map at the top-level,
    all preceding data can be used to unserialize an instance.
    """
    
    total_objects = 0
    current_objects = 0
    t0 = None
    
    def yajl_end_map(self, *args, **kwargs):
        ret = super(ContentBuilderOnMap, self).yajl_end_map(*args, **kwargs)
        if self.map_depth == 0:
            # Process a single record in JSON format.
            self.current_objects += 1
            data = ret.obj
            
            #pprint(data, indent=4)
            
            if not self.current_objects % 100:
                current_seconds = time.time() - self.t0
                total_seconds = current_seconds * self.total_objects / float(self.current_objects)
                remaining_seconds = total_seconds - current_seconds
                eta = datetime.now() + timedelta(seconds=remaining_seconds)
                print '\rLoading object %i of %i (%.02f%% eta = %s)' % (
                    self.current_objects,
                    self.total_objects,
                    self.current_objects/float(self.total_objects)*100,
                    eta,
                ),
                sys.stdout.flush()
            
            
            model = get_model(*data['model'].split('.', 1))
            assert hasattr(model, 'natural_key_fields'), \
                'Model %s must define natural_key_fields.' % (model.__name__,)
#            print '!'*80
#            print 'model:',model
#            print 'data:',data
                
            data_str = simplejson.dumps([data])
#            print 'data_str:',data_str
            for deserialized_object in serializers.deserialize('json', data_str):
                obj = deserialized_object.object
                #print 'deserialized_object:',deserialized_object.object
                
                # Even if the deserialized_object is created from a record
                # retrieved from get_by_natural_key() it will still not have
                # a pk, causing a new record to be created if save() is called.
                # So to avoid creating duplicate records, we have to
                # additionally lookup the pk.
                pk = find_object_pk(model, obj, data)
                #print 'pk:',pk
                if pk:
                    obj.id = pk
                
                obj.save()
        return ret

class Command(BaseCommand):
    args = '<fn>'
    help = 'Imports a genome from a JSON file exported by dump_genome.'
    option_list = BaseCommand.option_list + (
#        make_option('--force', action='store_true', default=False),
#        make_option('--delete-existing', action='store_true', default=False),
    )

    def handle(self, fn, **options):
        models.disable_fingerprint_check()
        models.disable_validation_check()
        tmp_debug = settings.DEBUG
        settings.DEBUG = False
        transaction.enter_transaction_management()
        transaction.managed(True)
        success = True
        try:
            handler = ContentBuilderOnMap()
            
            # Estimate total objects in the file without loading it completely.
            # Assumes the object's end bracket is at the start of a line.
            print 'Counting total objects in %s...' % (fn,)
            handler.total_objects = int(commands.getoutput('cat "%s" | grep -c "^\}"' % (fn,)))
            handler.t0 = time.time()
            
            parser = YajlParser(handler)
            print 'Loading objects in %s...' % (fn,)
            parser.parse(open(fn))
        except Exception, e:
            success = False
            raise
        finally:
            settings.DEBUG = tmp_debug
            if success:
                print 'Committing...'
                transaction.commit()
            else:
                print 'Rolling back...'
                transaction.rollback()
            transaction.leave_transaction_management()
            connection.close()
            models.enable_fingerprint_check()
            models.enable_validation_check()
        if success:
            print 'Success!'
        else:
            print 'Failure!'
            