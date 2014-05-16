"""
Wrappers around Yajl objects to simplify incremental parsing of YAML data.
"""

from yajl.yajl_parse import *

class Focus(object):
    def __init__(self, obj, key):
        assert not isinstance(obj, Focus)
        self.obj = obj
        self.key = key
    def __repr__(self):
        return '%s at %s' % (repr(self.obj), self.key)
    def add(self, v2, key=None):
        value = v2
        if isinstance(v2, Focus):
            value = v2.obj
        assert not isinstance(value, Focus)
        key = key or self.key
        if isinstance(self.obj, list) and not key:
            self.obj.append(value)
        else:
            self.obj[key] = value

class YajlContentBuilder(YajlContentHandler):
    def __init__(self):
        self.map_depth = 0
        self.data_stack = []
        self.next_key = None
        self.cursor = None
        self.data = None
        
    def push(self, v):
        if self.data is None:
            self.data = v
            next = Focus(v, self.next_key)
            self.data_stack.append(next)
            self.next_key = None
        elif isinstance(v, dict):
            next = Focus(v, self.next_key)
            self.data_stack[-1].add(next, key=self.next_key)
            self.data_stack.append(next)
            self.next_key = None
        elif isinstance(v, list):
            next = Focus(v, key=0)
            self.data_stack[-1].add(next, key=self.next_key)
            self.data_stack.append(next)
            self.next_key = None
        else:
            self.data_stack[-1].add(v, key=self.next_key)
            self.next_key = None
        
    def pop(self):
        return self.data_stack.pop()
        
    def yajl_null(self, ctx):
        self.push(None)
    def yajl_boolean(self, ctx, boolVal):
        self.push(boolVal)
    def yajl_integer(self, ctx, integerVal):
        self.push(int(integerVal))
    def yajl_double(self, ctx, doubleVal):
        self.push(float(doubleVal))
    def yajl_number(self, ctx, stringNum):
        '''Since this is defined both integer and double callbacks are useless.'''
        num = float(stringNum) if '.' in stringNum else int(stringNum)
        self.push(num)
    def yajl_string(self, ctx, stringVal):
        self.push(stringVal)
    def yajl_start_map(self, ctx):
        self.map_depth += 1
        self.push({})
    def yajl_map_key(self, ctx, stringVal):
        self.next_key = stringVal
    def yajl_end_map(self, ctx):
        self.map_depth -= 1
        return self.pop()
    def yajl_start_array(self, ctx):
        self.push([])
    def yajl_end_array(self, ctx):
        self.pop()
        