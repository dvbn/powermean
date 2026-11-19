"""Fixed effects: the grouping preparation of the blocks of an estimation.

``DESIGN.md``, Sections 5.1 to 5.3 and 7; ``design/03_fixed_effects.md``, Sections 1
and 2; ``design/11_engineering.md``, Section 3.1 (the preparation layer); decision D66.
Task M1.2 adds the private grouping preparation on arrays and integer codes, and
nothing that fits: the encoding of the levels of each block in the order of their
first occurrence, with the exact original positions of the rows kept apart from the
user's labels (:mod:`powermean.fe.groups`: blocks, interactions of encoded blocks as
tuples, counts, stable sorted rows with their inverse and segment pointers); the two
bounded group
reductions copied from the pyiols backends, unchanged in their arithmetic and verified
bit for bit against their source (:mod:`powermean.fe.reductions`; ``PROVENANCE.md``,
Section 20); and the connectivity metadata of the blocks, the connected components of
their levels with the exact indicator rank of one and two blocks and the proven bounds
of three or more (:mod:`powermean.fe.connectivity`), new code checked against literal
indicator matrices in its tests.

Nothing here deletes a row or a level, builds a dense indicator design, absorbs,
profiles, decides a normalization or an omission, or fits.  The package works on arrays
and small records, imports neither pandas nor Formulaic nor any later stage, and is
internal: ``powermean`` exports none of its names.
"""
