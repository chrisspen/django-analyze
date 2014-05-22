/*
Aggregates all gene fitness statistics for the current epoche.
*/
DROP VIEW IF EXISTS django_analyze_genestatistics;
CREATE VIEW django_analyze_genestatistics
AS
SELECT  CAST(gt.genome_id AS VARCHAR) || '-' || CAST(gg.gene_id AS VARCHAR) || '-' || CAST(gg.value AS VARCHAR) AS id,
        gt.genome_id,
        gg.gene_id,
        gg.value,
        MIN(gt.fitness) AS min_fitness,
        AVG(gt.fitness) AS mean_fitness,
        MAX(gt.fitness) AS max_fitness,
        COUNT(DISTINCT gt.id) AS genotype_count
FROM    django_analyze_genotype AS gt
INNER JOIN
        django_analyze_genotypegene AS gg ON
        gg.genotype_id = gt.id
WHERE   gt.fitness IS NOT NULL
GROUP BY
        gt.genome_id,
        gg.gene_id,
        gg.value;
