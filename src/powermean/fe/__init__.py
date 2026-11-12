"""Fixed effects: the grouping preparation of the blocks of an estimation.

The grouping preparation works on arrays and integer codes and fits nothing: the
encoding of the levels of each block in the order of their first occurrence, with the
exact original positions of the rows kept apart from the user's labels
(:mod:`powermean.fe.groups`: blocks, interactions of encoded blocks as tuples, counts,
stable sorted rows with their inverse and segment pointers); the two bounded group
reductions (:mod:`powermean.fe.reductions`); and the connectivity metadata of the
blocks, the connected components of their levels with the exact indicator rank of one
and two blocks and the proven bounds of three or more
(:mod:`powermean.fe.connectivity`).

Nothing here deletes a row or a level, builds a dense indicator design, absorbs,
profiles, decides a normalization or an omission, or fits.  The package works on arrays
and small records, imports neither pandas nor Formulaic nor any later stage, and is
internal: ``powermean`` exports none of its names.
"""
