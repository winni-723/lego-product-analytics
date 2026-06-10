# Metrics & Definitions Glossary

This document defines the key metrics and terms used across the LEGO Product
Analytics project. The AI assistant uses it to answer "what does X mean" style
questions.

## Set
A single LEGO product (one box / SKU), identified by a `set_num` such as
`75192-1`. Each set has a release year, a theme, and a part count.

## Part count (num_parts)
The number of individual LEGO pieces in a set. Used as the primary proxy for a
set's size and build complexity.

## Size tiers
Sets are bucketed into four size tiers by part count:
- **Small**: fewer than 100 parts
- **Medium**: 100 to 499 parts
- **Large**: 500 to 999 parts
- **Flagship**: 1,000 or more parts

## Flagship set
A set with 1,000 or more parts. Flagship sets are a proxy for premium /
adult-oriented (18+) products: they are larger, more expensive, and carry higher
margins. The "flagship %" metric tracks what share of a year's releases are
flagship sets — a signal of the catalogue's shift toward premium building.

## Complexity
Used interchangeably with average part count. Rising average parts per set over
time indicates the catalogue is shifting toward larger, more complex builds.

## Theme
A LEGO product line, such as City, Technic, or Star Wars. Themes form a tree:
a top-level theme (a "product line") can have sub-themes beneath it.

## Top-level theme (product line)
A theme with no parent. In the ML models, sets are rolled UP the theme tree so a
top-level theme like "Star Wars" includes every set from all of its sub-themes.
This is the unit of analysis for the survival model.

## Sub-theme
A child theme nested under a parent theme. Rebrickable attaches most sets to the
leaf sub-themes rather than the top-level theme.

## Lifespan
For a theme, the number of years between its first and latest set release
(`latest_year - first_year`).

## Active vs Retired theme
A theme (product line) is labelled **Active** if it released a set within the
last year of available data, otherwise **Retired**. This status is what the
theme survival model predicts.

## Popularity (owned_by)
The number of Brickset users who report owning a set. This is the project's
proxy for real-world popularity / demand, since the catalogue data itself has no
sales figures. The popularity model predicts this value.

## Wanted_by
The number of Brickset users who report wanting a set. A demand / wishlist
signal. Deliberately NOT used as a model feature, because it is only known after
a set ships (using it would be data leakage).
