BATCH = 'batch'
MINIBATCH = 'minibatch'
INCREMENTAL = 'incremental'

SIMPLE = 'simple'
MULTICLASS = 'multiclass'
MULTILABEL = 'multilabel'

GENE_TYPE_INT = 'int'
GENE_TYPE_FLOAT = 'float'
GENE_TYPE_BOOL = 'bool'
GENE_TYPE_STR = 'str'
GENE_TYPE_GENOME = 'genome'
GENE_TYPE_CHOICES = (
    (GENE_TYPE_INT, 'int'),
    (GENE_TYPE_FLOAT, 'float'),
    (GENE_TYPE_BOOL, 'bool'),
    (GENE_TYPE_STR, 'str'),
    (GENE_TYPE_GENOME, 'genome'),
)
GENE_TYPES = (
    int,
    float,
    bool,
    basestring,
)

POPULATE_ALL = 'all'