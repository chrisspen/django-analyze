/*
Find all genes that don't have a corresponding genotype gene value but should.
*/
DROP VIEW IF EXISTS django_analyze_genotypegenemissing CASCADE;
CREATE VIEW django_analyze_genotypegenemissing
AS
SELECT  m.gene_id,
        m.genotype_id,
        g.name as gene_name,
        g.default
FROM (
    SELECT  gt.id AS genotype_id,
        g.id AS gene_id,
        --g.name AS gene_name,
    /*
        gd3.id AS gene_dependency_id,
        g3.name AS dependent_gene_name,
        gd3.dependee_value AS required_dependee_value,
        gg3.value as actual_dependee_value,
        gd3.positive as dependee_gene_value_polarity,
    */
        EVERY(CASE
            WHEN gd3.id IS NULL THEN true -- no dependencies and missing, so just add
            WHEN gd3.positive = true AND gd3.dependee_value = gg3.value THEN true -- positive requirement met
            WHEN gd3.positive = false AND gd3.dependee_value != gg3.value THEN true -- negative requirement met
            ELSE false
        END) AS should_add
        
        --g.default
    FROM django_analyze_genotype AS gt
    INNER JOIN django_analyze_genome AS gn ON
        gn.id = gt.genome_id
        --and gt.id=7021
    INNER JOIN django_analyze_gene AS g ON
        g.genome_id = gn.id
    LEFT OUTER JOIN django_analyze_genotypegene AS gg ON 
        gg.genotype_id = gt.id
        AND gg.gene_id = g.id

    LEFT OUTER JOIN django_analyze_genedependency AS gd3 ON -- dependency link
        gd3.gene_id=g.id
    LEFT OUTER JOIN django_analyze_gene AS g3 ON -- dependee gene
        g3.id = gd3.dependee_gene_id
    LEFT OUTER JOIN django_analyze_genotypegene AS gg3 ON -- value of dependee gene in our genotype
        gg3.genotype_id = gt.id
        AND gg3.gene_id = g3.id

    WHERE   gg.id IS NULL -- if we have it it is not missing
    GROUP BY
        gt.id,
        g.id
) AS m
INNER JOIN django_analyze_gene AS g ON g.id = m.gene_id
WHERE m.should_add = true;
