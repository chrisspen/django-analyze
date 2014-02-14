import re
import sys
from pprint import pprint

import django
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db.models import Q
from django.core import serializers
from django.db.models.loading import get_model
from django.utils import simplejson

from optparse import make_option

from pyyajl import YajlContentBuilder, YajlParser

from django_analyze import models

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
    #print 'natural key:',obj.natural_key()
    keys = [data['fields'][_] for _ in model.natural_key_fields]
    keys = flatten_list(keys)
    #print 'keys:',keys
    obj = model.objects.get_by_natural_key(*keys)
    #print 'obj:',obj
    #print 'obj.pk:',obj.pk, obj.id
    if hasattr(obj, 'pk'):
        return obj.pk
    return obj.id

class ContentBuilderOnMap(YajlContentBuilder):
    
    def yajl_end_map(self, *args, **kwargs):
        ret = super(ContentBuilderOnMap, self).yajl_end_map(*args, **kwargs)
        if self.map_depth == 0:
            # Process a single record in JSON format.
            data = ret.obj
            pprint(data, indent=4)
            model = get_model(*data['model'].split('.', 1))
            assert hasattr(model, 'natural_key_fields'), 'Model %s must define natural_key_fields.' % (model.__name__,)
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
                # So to avoid created duplicate records, we have to
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
        handler = ContentBuilderOnMap()
        parser = YajlParser(handler)
        parser.parse(open(fn))
        #pprint(handler.data, indent=4)
#        print type(handler.data)
#        print type(handler.data[0])
#        