// Assemble the Markdown chapters; no package imports, example execution or network.
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root = path.dirname(fileURLToPath(import.meta.url));
const indexChapter = ['README.md', 'read-the-guide', 'Guide index'];
const groups = [
  ['Get started', 'start', [
    ['install.md', 'install', 'Install'],
    ['methods/model.md', 'model', 'Mean and variance'],
    ['help/prepare-data.md', 'prepare-data', 'Prepare data'],
    ['examples/compare-samples.md', 'compare-samples', 'Compare samples'],
  ]],
  ['Read and report results', 'read-and-report', [
    ['methods/proportional-effects.md', 'proportional-changes', 'Proportional changes'],
    ['methods/inference.md', 'uncertainty', 'Standard errors'],
    ['examples/reuse-a-fit.md', 'reuse-a-fit', 'Explore and export a fit'],
    ['reference/results.md', 'results', 'Results and prediction'],
    ['reference/tables.md', 'tables', 'Tables and exports'],
  ]],
  ['Fixed effects', 'fit-a-model', [
    ['methods/fixed-effects.md', 'fixed-effects', 'Estimate fixed effects'],
    ['methods/cancellation.md', 'cancellation', 'Cancel one block'],
  ]],
  ['Panel designs and applications', 'panel-and-inference', [
    ['examples/nonlinear-did.md', 'nonlinear-did', 'Nonlinear panel DiD'],
    ['get-started.md', 'first-fit', 'Gravity walkthrough'],
    ['methods/jochmans.md', 'jochmans', 'Difference out two blocks'],
    ['methods/bias-correction.md', 'bias-correction', 'Coefficient bias'],
    ['methods/covariance-refinements.md', 'covariance-refinements', 'Gravity pair covariance'],
    ['from-ppmlhdfe.md', 'compare-ppmlhdfe', 'Compare with ppmlhdfe'],
    ['examples/README.md', 'examples', 'Run the examples'],
  ]],
  ['Reference and help', 'reference-and-help', [
    ['reference/api.md', 'api', 'Function index'],
    ['availability.md', 'availability', 'Supported methods'],
    ['reference/options.md', 'options', 'Fit options'],
    ['methods/numerical-methods.md', 'numerical-methods', 'Numerical methods'],
    ['reference/solver-options.md', 'solver-options', 'Solver options'],
    ['reference/experimental-fixed-effects.md', 'experimental-fe', 'Advanced fixed effects'],
    ['reference/iols_options.md', 'iols-options', 'IOLS options'],
    ['iols_installation.md', 'iols-installation', 'IOLS installation'],
    ['help/troubleshooting.md', 'troubleshooting', 'Troubleshooting'],
    ['help/report-a-problem.md', 'report-a-problem', 'Report a problem'],
    ['validation.md', 'validation', 'Validation'],
    ['references.md', 'references', 'Papers and methods'],
    ['BUILD_GUIDE.md', 'build-the-guide', 'Build the guide'],
  ]],
];
const chapters = [indexChapter, ...groups.flatMap(([, , entries]) => entries)];
const check = process.argv.includes('--check');
const navStart = '<!-- guide-navigation:start -->';
const navEnd = '<!-- guide-navigation:end -->';
const tick = String.fromCharCode(96);
const fence = tick.repeat(3);
const slug = text => text.replaceAll(tick, '').toLowerCase()
  .replace(/[^\p{L}\p{N}_\-\s]/gu, '').replace(/ /g, '-');
const texts = new Map();
const targets = new Map();
const aliasCounts = new Map(groups.map(([, id]) => [id, 1]));
for (const [index, [file, prefix]] of chapters.entries()) {
  const source = path.join(root, file);
  const original = fs.readFileSync(source, 'utf8');
  const body = original.split(navStart)[0].trim();
  const neighbors = [];
  const add = ([target, , title], label) => {
    const relative = path.posix.relative(path.posix.dirname(file), target);
    neighbors.push('[' + label + title + '](' + relative + ')');
  };
  if (index > 1) add(chapters[index - 1], 'Previous: ');
  if (index > 0) add(chapters[0], '');
  if (index < chapters.length - 1) add(chapters[index + 1], 'Next: ');
  const updated = index === 0 ? body + '\n' : [
    body, '', navStart, neighbors.join(' · '), navEnd, '',
  ].join('\n');
  if (check && updated !== original) throw Error('Stale chapter navigation: ' + file);
  if (!check && updated !== original) fs.writeFileSync(source, updated);
  for (const [, id] of body.matchAll(/^<a id="([^"]+)"><\/a>$/gm)) {
    aliasCounts.set(id, (aliasCounts.get(id) ?? 0) + 1);
  }
  const seen = new Map();
  let fenced = false;
  const headings = [];
  for (const line of body.split('\n')) {
    if (line.startsWith(fence)) { fenced = !fenced; continue; }
    if (!fenced && /^#{1,6} /.test(line)) {
      const key = slug(line.replace(/^#+ /, ''));
      const count = seen.get(key) || 0;
      seen.set(key, count + 1);
      headings.push([key + (count ? '-' + count : ''), headings.length ? prefix + '--' + key + (count ? '-' + count : '') : prefix]);
    }
  }
  targets.set(file, new Map(headings));
  texts.set(file, {body, headings});
}
function link(file, url) {
  if (/^(?:https?:|mailto:|data:)/.test(url)) return url;
  const [name, fragment] = url.split('#');
  const target = path.posix.normalize(path.posix.join(path.posix.dirname(file), name || path.posix.basename(file)));
  if (targets.has(target)) {
    if (!fragment) return '#' + targets.get(target).values().next().value;
    const id = targets.get(target).get(decodeURIComponent(fragment));
    if (!id) throw Error('Unknown anchor: ' + file + ' -> ' + url);
    return '#' + id;
  }
  if (target === 'guide.html') return '#read-the-guide';
  const repoPath = path.posix.normalize(path.posix.join('docs', target));
  if (repoPath === '..' || repoPath.startsWith('../')) throw Error('Link escapes repository: ' + url);
  return target + (fragment ? '#' + fragment : '');
}
const header = [
  '---',
  'title: "Powermean"',
  'subtitle: "User guide for proportional regression and fixed effects"',
  'lang: en',
  'format:',
  '  html:',
  '    theme: default',
  '    mainfont: system-ui',
  '    toc: true',
  '    toc-title: "In this guide"',
  '    toc-depth: 2',
  '    toc-location: left',
  '    toc-expand: 1',
  '    number-sections: false',
  '    html-math-method: mathml',
  '    embed-resources: true',
  '    code-copy: true',
  '    code-overflow: wrap',
  '    anchor-sections: true',
  '    smooth-scroll: false',
  '    css: guide.css',
  'execute:',
  '  enabled: false',
  '---',
  '',
  '<!-- Generated from the chapter Markdown files by build-guide.mjs.',
  '     Edit those files, regenerate, then render; do not hand-edit the HTML. -->',
  '',
].join('\n');
const assembleChapter = ([file, prefix], index = false) => {
  const {body, headings} = texts.get(file);
  let headingIndex = 0;
  let fenced = false;
  return body.split('\n').map(line => {
    if (line.startsWith(fence)) { fenced = !fenced; return line; }
    if (fenced) return line;
    const alias = line.match(/^<a id="([^"]+)"><\/a>$/);
    if (alias && aliasCounts.get(alias[1]) > 1) return '<a id="' + prefix + '--' + alias[1] + '"></a>';
    if (/^#{1,6} /.test(line)) {
      const [, id] = headings[headingIndex++];
      if (index) return line + ' {#' + id + ' .unlisted}';
      return '#' + line + ' {#' + id + '}';
    }
    return line.replace(/\]\(([^)\s]+)\)/g, (_, url) => '](' + link(file, url) + ')');
  }).join('\n');
};
const index = texts.get(indexChapter[0]).body;
const indexedGroups = [...index.matchAll(/^## (.+)\n([\s\S]*?)(?=^## |$(?![\s\S]))/gm)];
if (indexedGroups.length !== groups.length) throw Error('Guide outline group count differs from the offline navigation.');
for (const [i, [title, , entries]] of groups.entries()) {
  const [, label, body] = indexedGroups[i];
  const linked = [...body.matchAll(/\]\(([^)#\s]+\.md)\)/g)].map(match => match[1]);
  if (label !== title || linked.join() !== entries.map(([file]) => file).join()) {
    throw Error('Guide outline links differ from the offline navigation in ' + title);
  }
}
const assembled = [
  assembleChapter(indexChapter, true),
  ...groups.map(([heading, id, entries]) => [
    '# ' + heading + ' {#' + id + '}',
    ...entries.map(entry => assembleChapter(entry)),
  ].join('\n\n')),
].join('\n\n');
const output = header + '\n' + assembled + '\n';
const destination = path.join(root, 'guide.qmd');
if (check) {
  if (!fs.existsSync(destination) || fs.readFileSync(destination, 'utf8') !== output)
    throw Error('guide.qmd differs from its chapter sources; regenerate it.');
  console.log('PASS: guide.qmd matches all ' + chapters.length + ' chapter sources.');
} else {
  fs.writeFileSync(destination, output);
  console.log('Assembled ' + chapters.length + ' chapters in docs/guide.qmd. Render it with Quarto.');
}

