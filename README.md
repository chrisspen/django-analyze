Django-Analyze - Framework for managing classifiers
===============================================================================

Overview
--------

There are tons of amazing algorithms and machine learning tools for detecting
patterns in data. However, what most of these lack is a useful framework and UI
for managing the often complicated setup of the data flow and predictions.

This package provides several tools for utilizing Django's admin interface
and ORM to help organize and manage machine learning setups.

The framework revolves around two basic objects:

1. A problem, which organizes solutions to acheive some prediction goal.
    This is mainly implemented a genetic algorithm.
2. A predictor, which organizes a specific solution to either guess a numeric
    value (i.e. regression) or a label (i.e. classification).
    
I made this separation to help myself with maintainence over the life time of
an application. Often, I'd want to monitor the accuracy of a solution, but also
evaluation other potential solutions without interrupting the solution used for
production predictions. Once a superior solution was found, then I'd want to
push it into production use with as little effort as possible. By explicitly
representing different solutions as different records in the database, I found
I could easily monitor them and slip them in and out of use as needed.

Problem
-------

The `problem` represents a domain where we're attempting to solve some
prediction task, by either guessing a number or guessing a label. In the code,
this is referred to as the `Genome`. A record in the `Genome` table represents
a distinct problem domain and stores all the parameters used to control and
manage the search for solutions.

From the `Genome` you define `Genes`, which are parameters available for use
when attempting to solve the problem.

Specific solutions to the problem are represented by the `Genotype` model,
which contains a list of genes and their associated values as key/values pairs.

To search for the best solution to a problem, you first implement a custom
evaluating function, which will take a genotype as an argument and return a
positive number, called the fitness, representing its overall suitability in
solving the problem. By default, a value of 0 is interpreted to be the worse
possible fitness and increasing value representing increasing levels of
suitability. Personally, I find it convenience and intuitive to bound fitness
between 0 and 1, but this is not strictly enforced.

You tehn set this function in your `Genome's` `evaluator` field and run
the management command:

    python manage.py anaylze_evolve <genome_id>

Depending on the other settings in the genome, this will run for a maximum
predetermined number of iterations or until improvement of the fitness has
stalled. From the genome's admin change page, you can browse the list of
generated genotypes and inspect their fitness, possibly selecting one for
production use.

For example, a trivially simple genome might consist of a single gene called
`algorithm`, which contains one of several algorithm names
(e.g. 'Bayesian', 'LinearSVC', 'RandomForest', etc.). You would write your
evaluation function to read this string and instantiate the appropriate class
associated with the name. You could then add additional genes representing
parameters common to multiple algorithms or unique to only a few.
The `Genotype` model with generate a unique hash based on which genes it
contains, and use this to avoid creating duplicate genotypes.

Predictor
---------
todo

Usage
-----
todo