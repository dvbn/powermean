"""The interface layer behind the public functions: validation, preparation, dispatch.

The modules of this package are the
interface layer behind :mod:`powermean.api`: they collect the keywords of a
call, validate them and normalize them into explicit typed records
(:mod:`powermean.interface.options`); evaluate an offset or exposure expression and a
cluster interaction with the Formulaic adapter
(:mod:`powermean.interface.expressions`); prepare the sample once through the
preparation boundary (:mod:`powermean.interface.preparation`); and compose the
stages of a fit, the fresh certification of its point and the handoff to the
result (:mod:`powermean.interface.dispatch`).  The internal ordered sample composition
(:mod:`powermean.interface.sample`, with its request, invocation and
round records in :mod:`powermean.interface.sample_records` and the owned evidence of
its support invocations in :mod:`powermean.interface.sample_evidence`) coordinates the
sample diagnostics.  Nothing here is a public
name: the public names are those of ``powermean.__all__``.
"""
