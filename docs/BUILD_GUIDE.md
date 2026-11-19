# Build or read the guide

The [HTML guide](guide.html) is the complete reading document. Open it in a
current browser. Its style, navigation scripts and MathML equations are
included; reading it does not require a connection. Keep it in the unpacked
source distribution to follow links to accompanying files. Links to papers
may need a connection.

The matching [Quarto document](guide.qmd) contains the full guide text.
The individual Markdown chapters remain the editing source. After changing
a chapter, assemble and render the document from the repository root:

```bash
node docs/build-guide.mjs
quarto render docs/guide.qmd --to html
node docs/build-guide.mjs --check
```

Quarto is a documentation authoring tool, not a Powermean runtime dependency.
The earlier checked render used Quarto 1.10.18. Rendering does not
execute the Python examples. The Markdown chapters hold the captured
console output and package-generated tables. Images are embedded in the
reading copy. To refresh an example, run its displayed Python in an
installed Powermean environment before rendering, and save figure outputs
under `docs/assets/`.

The build script checks that the Markdown outline and offline guide list
the same chapters in the same five groups. It refreshes chapter navigation
and resolves links inside the combined document. `guide.css` supplies local presentation
styles. Commit the chapter changes, regenerated `guide.qmd` and rendered
`guide.html` together so the source and reading copy remain synchronized.

<!-- guide-navigation:start -->
[Previous: Papers and methods](references.md) · [Guide index](README.md)
<!-- guide-navigation:end -->
