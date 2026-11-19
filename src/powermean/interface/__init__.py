"""The public interface of the first milestone: validation, preparation and dispatch.

``DESIGN.md``, Sections 4.1, 4.4, 4.6 and 5.3; ``design/02_computation.md``, Sections
2, 5.5, 6, 7 and 9; ``design/11_engineering.md``, Section 5, rules 6, 7 and 13;
decisions D41, D45, D47, D49 and D51; task M0.22.  The modules of this package are the
interface layer behind :mod:`powermean.api`: they collect the approved keywords of a
call, validate them and normalize them into explicit typed records
(:mod:`powermean.interface.options`); evaluate an offset or exposure expression and a
cluster interaction with the established Formulaic adapter
(:mod:`powermean.interface.expressions`); prepare the sample once through the accepted
preparation boundary (:mod:`powermean.interface.preparation`); and compose the
accepted stages of a fit, the fresh certification of its point and the handoff to the
result (:mod:`powermean.interface.dispatch`).  The private ordered sample composition
of task M1.10 (:mod:`powermean.interface.sample`, with its request, invocation and
round records in :mod:`powermean.interface.sample_records` and the owned evidence of
its support invocations in :mod:`powermean.interface.sample_evidence`) coordinates the
accepted sample diagnostics; no public call reaches it yet.  Nothing here is a public
name: the public names are those of ``powermean.__all__``.
"""
