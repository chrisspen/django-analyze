/*
Find all genotype gene values that aren't legally permissable based
on gene dependency rules.
*/
DROP VIEW IF EXISTS django_analyze_genotypegeneillegal CASCADE;
CREATE VIEW django_analyze_genotypegeneillegal
AS
SELECT  gg.id AS illegal_genotypegene_id,
        g.name AS illegal_gene_name,
        gt.id AS illegal_genotype_id,
        g2.name AS dependee_name,
        g.dependee_value,
        gg2.value AS illegal_value
FROM django_analyze_genotype AS gt
INNER JOIN django_analyze_genotypegene AS gg ON
        gg.genotype_id = gt.id
INNER JOIN django_analyze_gene AS g ON
        g.id = gg.gene_id
INNER JOIN django_analyze_gene AS g2 ON
        g2.id = g.dependee_gene_id
INNER JOIN django_analyze_genotypegene AS gg2 ON
        gg2.genotype_id = gt.id
    AND gg2.gene_id = g2.id
WHERE   g.dependee_value != gg2.value

UNION ALL

SELECT  gg.id AS illegal_genotypegene_id,
        g.name AS illegal_gene_name,
        gt.id AS illegal_genotype_id,
        g3.name AS dependee_name,
        gd3.dependee_value,
        gg3.value AS illegal_value
FROM    django_analyze_genotype AS gt
INNER JOIN django_analyze_genotypegene AS gg ON
        gg.genotype_id = gt.id
INNER JOIN django_analyze_gene AS g ON
        g.id = gg.gene_id
LEFT OUTER JOIN django_analyze_genedependency AS gd3 ON
        gd3.gene_id=g.id
LEFT OUTER JOIN django_analyze_gene AS g3 ON
        g3.id = gd3.dependee_gene_id
LEFT OUTER JOIN django_analyze_genotypegene AS gg3 ON
        gg3.genotype_id = gt.id
    AND gg3.gene_id = g3.id
WHERE   gd3.id IS NOT NULL
    AND (
        (gd3.positive = true AND gg3.value != gd3.dependee_value)
        OR
        (gd3.positive = false AND gg3.value = gd3.dependee_value)
    );
    