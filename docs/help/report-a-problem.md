# Report a reproducible problem

After [troubleshooting](troubleshooting.md), a small reproducible example lets
another reader examine the same behavior. Use the project's issue tracker if
you have access. Reports should exclude confidential data and credentials.

## Include enough to reproduce the behavior

State what you expected and what happened, using the model definition or a
matched software comparison where relevant. Include the Powermean version,
Python and dependency versions, operating system, exact formula and options,
and the complete exception identifier and message. For a local build, include
the source commit or the distribution filename and its SHA-256 digest. If the
fit returned a result with a failure status, include that status.

Use a small simulated or redistributable dataset with the same essential
structure. Keep meaningful zeros, FE block relationships and dependence units;
do not replace the problem by a different successful model. Save the random
seed if data are simulated.

For a fixed-effect fit, include the organization and explicit control
records, cancelled and retained block names, sample changes and requested
covariance. For a comparison, provide both packages' calls and conventions.
For a runtime report, state the timing boundary, thread setting and repetitions.

## Record versions without sharing credentials

Run this in the environment where you used Powermean:

```python
import platform
from importlib.metadata import version

print("Python", platform.python_version())
for package in ("powermean", "numpy", "scipy", "pandas", "formulaic", "within-py"):
    print(package, version(package))
```

From a Git checkout, `git rev-parse HEAD` gives its commit. For an installed
source distribution or wheel, give the file name and its SHA-256 digest. Identify
the build you used. A version number alone cannot distinguish local builds
from different revisions.

Remove access tokens, credentials, personal identifiers, local private paths
and confidential rows before posting. Do not attach an entire environment,
private research archive or account log. A minimal reproducible example is
more useful than a large unrelated dump.

The report should preserve the original statistical question and settings.
Changing a sample, assumption or tolerance just to obtain success can conceal
the problem. When citing an expected method result, use the
[method source map](../references.md) to identify the relevant version and
assumptions.

<!-- guide-navigation:start -->
[Previous: Troubleshooting](troubleshooting.md) · [Guide index](../README.md) · [Next: Validation](../validation.md)
<!-- guide-navigation:end -->
