/*
Find all genotype gene values that aren't legally permissable based
on gene dependency rules.
*/
DROP VIEW IF EXISTS django_analyze_genotypegeneillegal CASCADE;
CREATE VIEW django_analyze_genotypegeneillegal
AS
SELECT  m.genotypegene_id AS illegal_genotypegene_id,
        g.name AS illegal_gene_name,
        m.genotype_id AS illegal_genotype_id
FROM (
    SELECT  m.genotype_id,
            m.gene_id,
            m.genotypegene_id,
            COUNT(CASE
                WHEN m.dependent_gene_id IS NULL
                    THEN 1
                WHEN m.dependent_gene_polarity AND m.should_add1 > 0
                    -- all values for this dependency are ORed
                    THEN 1
                WHEN not m.dependent_gene_polarity AND m.should_add1 = m.should_add2
                    -- all values for this dependency are ANDed
                    THEN 1
                ELSE NULL
            END) = COUNT(*) AS should_add
    FROM (
        SELECT  gt.id AS genotype_id,
                g.id AS gene_id,
                gg.id AS genotypegene_id,
                g3.id as dependent_gene_id,
                gd3.positive as dependent_gene_polarity,
    
                COUNT(CASE
                    WHEN gd3.id IS NULL
                        THEN 1 -- no dependencies and missing, so just add
                    WHEN gd3.positive AND gd3.dependee_value = gg3.value
                        THEN 1 -- positive requirement met
                    WHEN not gd3.positive AND gd3.dependee_value != gg3.value
                        THEN 1 -- negative requirement met
                    ELSE NULL
                END) AS should_add1,
                COUNT(*) AS should_add2
            
        FROM    django_analyze_genotype AS gt
        INNER JOIN
                django_analyze_genome AS gn ON
                gn.id = gt.genome_id
        INNER JOIN
                django_analyze_gene AS g ON
                g.genome_id = gn.id
        LEFT OUTER JOIN
                django_analyze_genotypegene AS gg ON 
                gg.genotype_id = gt.id
            AND gg.gene_id = g.id
    
        LEFT OUTER JOIN
                django_analyze_genedependency AS gd3 ON -- dependency link
                gd3.gene_id=g.id
        LEFT OUTER JOIN
                django_analyze_gene AS g3 ON -- dependee gene
                g3.id = gd3.dependee_gene_id
        LEFT OUTER JOIN
                django_analyze_genotypegene AS gg3 ON -- value of dependee gene in our genotype
                gg3.genotype_id = gt.id
            AND gg3.gene_id = g3.id
    
        WHERE   gg.id IS NOT NULL -- if we have it
        --and gt.id=2777 and gg.id=2023482
        GROUP BY
            gt.id,
            g.id,
            gg.id,
            g3.id,
            gd3.positive

    ) AS m
    GROUP BY
        m.genotype_id,
        m.gene_id,
        m.genotypegene_id
) AS m
INNER JOIN django_analyze_gene AS g ON g.id = m.gene_id
WHERE NOT m.should_add;
