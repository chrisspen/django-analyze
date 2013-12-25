from collections import defaultdict
import random

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
            c, w = o
        if upto + w >= r:
            return c
        upto += w
    raise Exception

if __name__ == '__main__':
    
    counts = defaultdict(int)
    weights = dict(a=1.0, b=2.0, c=3.0)
    for _ in xrange(1000):
        counts[weighted_choice(weights)] += 1
    for k,v in counts.iteritems():
        print v, k
        