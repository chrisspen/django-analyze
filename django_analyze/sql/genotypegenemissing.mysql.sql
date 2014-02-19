/*
Find all genes that don't have a corresponding genotype gene value but should.
*/
DROP VIEW IF EXISTS django_analyze_genotypegenemissing CASCADE;
CREATE VIEW django_analyze_genotypegenemissing
AS
SELECT  g.id AS gene_id,
        gt.id AS genotype_id,
        g.name AS gene_name,
        g.dependee_gene_id,
        g.default
FROM django_analyze_genotype AS gt
INNER JOIN django_analyze_genome AS gn ON
        gn.id = gt.genome_id
INNER JOIN django_analyze_gene AS g ON
        g.genome_id = gn.id
LEFT OUTER JOIN django_analyze_genotypegene AS gg ON
        gg.genotype_id = gt.id AND gg.gene_id = g.id
LEFT OUTER JOIN django_analyze_gene AS dg ON
        dg.id = g.dependee_gene_id
LEFT OUTER JOIN django_analyze_genotypegene AS dgg ON
        dgg.genotype_id = gt.id
    AND dgg.gene_id = dg.id

LEFT OUTER JOIN django_analyze_genedependency AS gd3 ON
        gd3.gene_id=g.id
LEFT OUTER JOIN django_analyze_gene AS g3 ON
        g3.id = gd3.dependee_gene_id
LEFT OUTER JOIN django_analyze_genotypegene AS gg3 ON
        gg3.genotype_id = gt.id
    AND gg3.gene_id = g3.id

WHERE   gg.id IS NULL
    AND (
        (g.dependee_gene_id IS NULL) OR
        (g.dependee_gene_id IS NOT NULL AND dgg.value = g.dependee_value)
    )
    AND (
        (gg3.id IS NULL) OR
        (gg3.id IS NOT NULL AND gg3.value = gd3.dependee_value AND gd3.positive = true) OR
        (gg3.id IS NOT NULL AND gg3.value != gd3.dependee_value AND gd3.positive = false)
    );