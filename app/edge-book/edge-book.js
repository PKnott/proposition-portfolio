/* Edge Book — Match Board and Portfolio Book over one day's export.
 *
 * Two payloads arrive inlined as <script type="application/json"> blocks:
 * predictions (one object per fixture) and portfolios (propositions, prices and
 * the undominated portfolio set). Nothing is fetched and nothing is computed
 * that Python already computed -- stake fractions in particular arrive per leg,
 * because a second implementation of the two split formulas is exactly the kind
 * of quiet disagreement this pipeline keeps designing out.
 */
(() => {
  'use strict';

  const PRED = JSON.parse(document.getElementById('predictions-data').textContent);
  const BOOK = JSON.parse(document.getElementById('portfolios-data').textContent);

  // --- indexes ------------------------------------------------------------

  const fixtures = PRED.fixtures.slice().sort(
    (a, b) => (a.date < b.date ? -1 : a.date > b.date ? 1 : a.event < b.event ? -1 : 1));
  const fixtureByEvent = new Map(fixtures.map(f => [f.event, f]));
  const propById = new Map(BOOK.propositions.map(p => [p.id, p]));
  const propsByEvent = new Map();
  for (const p of BOOK.propositions) {
    if (!propsByEvent.has(p.event)) propsByEvent.set(p.event, []);
    propsByEvent.get(p.event).push(p);
  }
  for (const list of propsByEvent.values()) {
    list.sort((a, b) => b.selection_frequency - a.selection_frequency);
  }
  // Prices are keyed on (event, label) because that pair is what the odds form
  // was filled against; `staking.proposition_label` builds the same string on
  // both sides of the export.
  const priceByKey = new Map(BOOK.prices.map(r => [r.event + '|' + r.label, r]));

  const portfolios = BOOK.portfolios;
  for (const pf of portfolios) pf._set = new Set(pf.picks);

  /* The growth block is nested in the payload, where it belongs -- it is one
     coherent answer to one question, not six loose fields. Everything on this
     page, though, indexes a portfolio by a flat key: `FIELDS`, `BOUNDS`, the
     sort comparator and every fader predicate all do `pf[key]`. Flattening once
     here is a line; teaching all four to walk a path would be four places to get
     it wrong. Portfolios predating the block simply have no growth keys, and the
     bounds below skip fields that are absent everywhere. */
  const HAS_GROWTH = portfolios.some(pf => pf.growth);
  /* The typical leg's chance of landing. A median, not a mean -- one near-certain
     leg among eleven longshots moves a mean and does not change what the
     portfolio is. Payloads written before it existed simply have no column. */
  const HAS_LEG_P = portfolios.some(pf => pf.median_leg_p != null);
  /* Edge capacity: the sum of each leg's squared standalone Sharpe, and the one
     channel leg count enters the model through. Adding a leg can only raise it,
     so it is the number to sort on -- and it is deliberately not a dominance
     criterion, since the largest book always wins it. Absent from older payloads. */
  const HAS_CAPACITY = portfolios.some(pf => pf.capacity != null);
  for (const pf of portfolios) {
    if (!pf.growth) continue;
    pf.growth_p0 = pf.growth.p0;
    pf.growth_f_suggested = pf.growth.f_suggested;
    pf.growth_g_suggested = pf.growth.g_suggested;
    pf.growth_f_drawdown = pf.growth.f_drawdown;
    pf.growth_breakeven_shift = pf.growth.breakeven_shift;
  }

  // --- formatting ---------------------------------------------------------

  const pct = (x, d = 2) => (x == null ? '—' : (x * 100).toFixed(d) + '%');
  const pp = x => (x >= 0 ? '+' : '') + (x * 100).toFixed(1);
  const num = (x, d = 2) => (x == null ? '—' : x.toFixed(d));
  /* A probability that is small but not zero must not print as '0.00%'. Ruin
     probability runs from 2.7e-09 to 0.63 across one day's portfolios, and the
     distinction this column exists to draw -- can this lose everything, yes or
     no -- is destroyed by rounding the yes to look like the no. */
  const pctFloor = (x, d = 2) => {
    if (x == null) return '—';
    if (x === 0) return '0%';
    const lo = Math.pow(10, -d) / 100;
    return x < lo ? '<' + (lo * 100).toFixed(d) + '%' : (x * 100).toFixed(d) + '%';
  };
  const int = x => x.toLocaleString('en-GB');
  const money = x => x.toLocaleString('en-GB', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  const clamp01 = x => Math.max(0, Math.min(1, x));
  const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  // "Corners - Coventry City - Over 2.5" -> "Corners – Coventry City – O2.5",
  // which is how the design writes it and fits a 280px column.
  const shortProp = s => String(s)
    .replace(/^Shots on Target\b/, 'SoT')          // as the design writes it, and it fits
    .replace(/ - Over /, ' – O').replace(/ - /g, ' – ');
  const shortLine = s => String(s).split(' ').pop();

  const el = id => document.getElementById(id);

  /* How a portfolio is named outside this page. Ids restart at 1 on every run,
     so `770` identifies a portfolio only within the file you are looking at;
     `R007-770` is what `fpp.ledger.record_bet` takes. Older payloads carry no
     run code and fall back to the bare id rather than inventing one. */
  const RUN_CODE = BOOK.run_code || null;
  const refOf = pf => RUN_CODE ? `${RUN_CODE}-${pf.id}` : String(pf.id);

  /** The signature component: fill = model probability, tick = market-implied. */
  function edgeBar(p, o) {
    const cls = o == null ? ' none' : p * o < 1 ? ' neg' : '';
    const tick = o ? `<div class="tick" style="left:${clamp01(1 / o) * 100}%"></div>` : '';
    return `<div class="edgebar"><div class="fill${cls}" `
         + `style="width:${clamp01(p) * 100}%"></div>${tick}</div>`;
  }

  // --- filterable fields --------------------------------------------------

  const FIELDS = [
    { key: 'expected_return_pct', label: 'Expected return %', kind: 'pct' },
    { key: 'sd_pct', label: 'SD %', kind: 'pct' },
    { key: 'variance', label: 'Variance', kind: 'var' },
    ...BOOK.thresholds.map(t => ({ key: 'p_over_' + t, label: `P(>${t}%)`, kind: 'pct' })),
    { key: 'legs', label: 'Legs', kind: 'int' },
    ...(HAS_LEG_P ? [{ key: 'median_leg_p', label: 'Median leg P', kind: 'pct' }] : []),
    ...(HAS_CAPACITY ? [
      { key: 'capacity', label: 'Edge capacity C', kind: 'var' },
      { key: 'n_eff', label: 'Effective legs', kind: 'var' },
      { key: 'max_leg_stake', label: 'Largest leg %', kind: 'pct' },
    ] : []),
    // One stake. It is the largest fraction that keeps a bad run unlikely at the
    // risk settings in `BOOK.risk`, shrunk for model error and capped. The old
    // growth-optimal column is gone: it read back its own ceiling on two thirds
    // of portfolios, so it was reporting a constant, not a recommendation.
    ...(HAS_GROWTH ? [
      { key: 'growth_f_suggested', label: 'Suggested stake %', kind: 'pct' },
      { key: 'growth_g_suggested', label: 'Growth % (at that stake)', kind: 'pct' },
      { key: 'growth_p0', label: 'P(returns nothing)', kind: 'pct' },
    ] : []),
  ];
  const FIELD_BY_KEY = new Map(FIELDS.map(f => [f.key, f]));
  const fmtField = (f, v) => f.kind === 'pct' ? pct(v, 1) : f.kind === 'var' ? num(v, 5) : String(v);

  // Bounds come from the data itself, per the brief, so a fader always opens at
  // the full range of what actually exists rather than a guessed scale.
  const BOUNDS = new Map(FIELDS.map(f => {
    let lo = Infinity, hi = -Infinity;
    for (const p of portfolios) { const v = p[f.key]; if (v < lo) lo = v; if (v > hi) hi = v; }
    if (!portfolios.length) { lo = 0; hi = 1; }
    return [f.key, { lo, hi }];
  }));

  const DEFAULT_FADERS = FIELDS.map(f => f.key);
  /** A fader opened to the full range of the field it filters. `BOUNDS` holds
   *  `{lo, hi}` and a fader carries `{min, max}`; spreading one into the other
   *  produced a control whose value was `undefined` and whose predicate was
   *  therefore never true or false but quietly absent. */
  const newFader = (key, custom = false) => {
    const b = BOUNDS.get(key);
    return { key, custom, min: b.lo, max: b.hi };
  };
  const SPLIT_KEYS = Object.keys(BOOK.splits);

  // --- state --------------------------------------------------------------

  const state = {
    view: 'match',
    event: fixtures.length ? fixtures[0].event : null,
    faders: DEFAULT_FADERS.map(key => newFader(key)),
    splits: new Set(),          // empty means "no constraint"
    props: new Map(),           // proposition id -> 'in' | 'out'
    events: new Map(),          // event code   -> 'in' | 'out'
    search: '',
    // Edge capacity, where the payload carries it: it is the single quantity that
    // orders selections, and the stake is proportional to it. Older payloads have
    // no such column and keep the previous default.
    sort: { key: HAS_CAPACITY ? 'capacity' : 'p_over_100', dir: -1 },
    stake: 100,
    open: null,                 // expanded portfolio id
    expanded: new Set(),        // which fixtures show their individual propositions
    // Projection Book. `pot` is deliberately *not* `stake`: the Portfolio Book's
    // stake is what goes on this portfolio once, and the pot is the bankroll a
    // stake fraction is taken out of every round. `stake = f * pot`, so calling
    // them one number would make the two pages quietly contradict each other.
    project: null,              // portfolio id being projected
    pot: 1000,
    customF: null,              // the third stake fraction, read off the curve
    stakeMode: 'suggested',     // which of the two the betting slip prices at
    rounds: 52,
    target: null,
  };

  const isNarrowed = f => {
    const b = BOUNDS.get(f.key);
    return f.min > b.lo || f.max < b.hi;
  };

  // --- hash persistence ---------------------------------------------------
  //
  // The URL, not localStorage: a filter set you spent ten minutes building is
  // worth being able to bookmark or paste. The hash rather than the query
  // string because a bundled page is opened from file://, where the History API
  // is unreliable and the hash is not.

  let restoring = false;
  // The hash this page last wrote. Setting `location.hash` fires `hashchange`,
  // and answering our own event by re-rendering the rail detached the very
  // input the pointer was holding -- so a fader moved one step and then stopped
  // dead, every time. Only a hash somebody *else* set is a reason to re-render.
  let lastWritten = null;

  function writeHash() {
    if (restoring) return;
    const q = new URLSearchParams();
    q.set('v', state.view);
    if (state.event) q.set('e', state.event);
    const fs = state.faders
      .filter(f => f.custom || isNarrowed(f))
      .map(f => `${f.key}:${f.min}:${f.max}`);
    if (fs.length) q.set('f', fs.join(','));
    if (state.splits.size) q.set('s', [...state.splits].join(','));
    const ins = [...state.props].filter(([, v]) => v === 'in').map(([k]) => k);
    const outs = [...state.props].filter(([, v]) => v === 'out').map(([k]) => k);
    if (ins.length) q.set('in', ins.join(','));
    if (outs.length) q.set('out', outs.join(','));
    const evIn = [...state.events].filter(([, v]) => v === 'in').map(([k]) => k);
    const evOut = [...state.events].filter(([, v]) => v === 'out').map(([k]) => k);
    if (evIn.length) q.set('evin', evIn.join(','));
    if (evOut.length) q.set('evout', evOut.join(','));
    q.set('sort', `${state.sort.key}:${state.sort.dir}`);
    if (state.stake !== 100) q.set('stake', String(state.stake));
    if (state.project != null) q.set('p', String(state.project));
    if (state.pot !== 1000) q.set('pot', String(state.pot));
    if (state.customF != null) q.set('cf', String(state.customF));
    if (state.stakeMode !== 'suggested') q.set('sm', state.stakeMode);
    if (state.rounds !== 52) q.set('n', String(state.rounds));
    if (state.target != null) q.set('tgt', String(state.target));
    lastWritten = q.toString();
    location.hash = lastWritten;
  }

  function readHash() {
    const raw = location.hash.replace(/^#/, '');
    if (!raw) return;
    const q = new URLSearchParams(raw);
    restoring = true;
    // A hash describes the whole filter set rather than patching the current
    // one. Merging looked identical on a fresh load and left the previous
    // state's IN list sitting underneath the new one on every back button.
    state.faders = DEFAULT_FADERS.map(key => newFader(key));
    state.splits = new Set();
    state.props = new Map();
    state.events = new Map();
    state.stake = 100;
    state.search = '';
    state.project = null;
    state.pot = 1000;
    state.customF = null;
    state.stakeMode = 'suggested';
    state.rounds = 52;
    state.target = null;
    if (q.get('v')) state.view = q.get('v');
    if (q.get('e') && fixtureByEvent.has(q.get('e'))) state.event = q.get('e');
    if (q.has('f')) {
      const seen = [];
      for (const part of q.get('f').split(',')) {
        const [key, lo, hi] = part.split(':');
        if (!FIELD_BY_KEY.has(key)) continue;
        seen.push({ key, custom: !DEFAULT_FADERS.includes(key) || seen.some(s => s.key === key),
                    min: Number(lo), max: Number(hi) });
      }
      // Defaults are permanent; anything the hash carried beyond them is custom.
      const byKey = new Map();
      for (const s of seen) if (!byKey.has(s.key)) byKey.set(s.key, s);
      state.faders = DEFAULT_FADERS.map(key => {
        const hit = byKey.get(key);
        return hit ? { key, custom: false, min: hit.min, max: hit.max } : newFader(key);
      });
      for (const s of seen) {
        if (byKey.get(s.key) !== s) state.faders.push({ ...s, custom: true });
      }
    }
    if (q.has('s')) state.splits = new Set(q.get('s').split(',').filter(Boolean));
    for (const id of (q.get('in') || '').split(',')) if (propById.has(id)) state.props.set(id, 'in');
    for (const id of (q.get('out') || '').split(',')) if (propById.has(id)) state.props.set(id, 'out');
    for (const ev of (q.get('evin') || '').split(',')) if (propsByEvent.has(ev)) state.events.set(ev, 'in');
    for (const ev of (q.get('evout') || '').split(',')) if (propsByEvent.has(ev)) state.events.set(ev, 'out');
    if (q.has('sort')) {
      const [key, dir] = q.get('sort').split(':');
      if (FIELD_BY_KEY.has(key) || key === 'id' || key === 'split') state.sort = { key, dir: Number(dir) };
    }
    if (q.has('stake')) state.stake = Number(q.get('stake')) || 100;
    // A bookmarked projection must land on a portfolio that is still in today's
    // export -- ids are per run, so yesterday's link would otherwise open the
    // page on an empty chart rather than saying the portfolio is gone.
    if (q.has('p') && portfolioById.has(Number(q.get('p')))) state.project = Number(q.get('p'));
    if (q.has('pot')) state.pot = Number(q.get('pot')) || 1000;
    if (q.has('cf')) state.customF = Number(q.get('cf')) || null;
    if (STAKE_MODES.some(m => m.key === q.get('sm'))) state.stakeMode = q.get('sm');
    if (q.has('n')) state.rounds = Number(q.get('n')) || 52;
    if (q.has('tgt')) state.target = Number(q.get('tgt')) || null;
    restoring = false;
  }

  // --- filtering ----------------------------------------------------------

  /** One predicate per active constraint, each labelled so an empty result can
   *  say which one is most likely to blame rather than showing a blank table. */
  function activeConstraints() {
    const out = [];
    for (const f of state.faders) {
      if (!isNarrowed(f)) continue;
      const field = FIELD_BY_KEY.get(f.key);
      out.push({
        label: `${field.label} ${fmtField(field, f.min)}–${fmtField(field, f.max)}`,
        test: p => p[f.key] >= f.min && p[f.key] <= f.max,
      });
    }
    if (state.splits.size) {
      out.push({ label: 'Split', test: p => state.splits.has(p.split) });
    }
    for (const [id, mode] of state.props) {
      const prop = propById.get(id);
      const name = shortProp(prop ? prop.proposition : id);
      out.push(mode === 'in'
        ? { label: `IN: ${name}`, test: p => p._set.has(id) }
        : { label: `OUT: ${name}`, test: p => !p._set.has(id) });
    }
    // A fixture is *not* the AND of its propositions. A portfolio backs at most
    // one proposition per event, so requiring all twelve of Arsenal v Coventry
    // at once is a filter that can only ever return nothing. IN on a fixture
    // therefore means "backs one of these", which is what anyone clicking it
    // wants; OUT is unambiguous either way and means "backs none of them".
    for (const [ev, mode] of state.events) {
      const ids = (propsByEvent.get(ev) || []).map(x => x.id);
      const f = fixtureByEvent.get(ev);
      const name = f ? `${f.home} v ${f.away}` : ev;
      out.push(mode === 'in'
        ? { label: `fixture IN: ${name}`, test: p => ids.some(id => p._set.has(id)) }
        : { label: `fixture OUT: ${name}`, test: p => !ids.some(id => p._set.has(id)) });
    }
    return out;
  }

  function applyFilters() {
    const cs = activeConstraints();
    const rows = cs.length ? portfolios.filter(p => cs.every(c => c.test(p))) : portfolios.slice();
    const { key, dir } = state.sort;
    rows.sort((a, b) => {
      const x = a[key], y = b[key];
      return (x < y ? -1 : x > y ? 1 : 0) * dir || a.id - b.id;
    });
    return { rows, cs };
  }

  /** The single active constraint that, relaxed alone, would admit the most rows. */
  function likelyCulprit(cs) {
    let best = null, bestN = -1;
    for (const skip of cs) {
      const others = cs.filter(c => c !== skip);
      let n = 0;
      for (const p of portfolios) if (others.every(c => c.test(p))) n++;
      if (n > bestN) { bestN = n; best = skip; }
    }
    return best ? { label: best.label, n: bestN } : null;
  }

  // --- match board --------------------------------------------------------

  function renderTickets() {
    el('tickets').innerHTML = fixtures.map(f => `
      <button type="button" class="ticket${f.event === state.event ? ' sel' : ''}" data-event="${esc(f.event)}">
        <div class="code">${esc(f.event)}</div>
        <div class="fx">${esc(f.home)} v ${esc(f.away)}</div>
        <div class="xg">xG ${num(f.exp_goals)}</div>
      </button>`).join('');
  }

  function heatmap(f) {
    const m = f.scoreline_matrix;
    const max = Math.max(...m.values.flat());
    let top = [0, 0], topV = -1;
    m.values.forEach((row, i) => row.forEach((v, j) => { if (v > topV) { topV = v; top = [i, j]; } }));
    const n = m.cols.length;
    const cells = [`<div class="lab"></div>`,
      ...m.cols.map(c => `<div class="lab">${esc(shortLine(c))}</div>`)];
    m.values.forEach((row, i) => {
      cells.push(`<div class="lab row">${esc(shortLine(m.rows[i]))}</div>`);
      row.forEach((v, j) => {
        const isTop = i === top[0] && j === top[1];
        cells.push(`<div class="cell${isTop ? ' top' : ''}" `
                 + `title="${esc(m.rows[i])} – ${esc(m.cols[j])}: ${pct(v)}">`
                 + `<i style="opacity:${(v / max).toFixed(3)}"></i></div>`);
      });
    });
    return `<div class="comp-label">Scoreline matrix — ${esc(f.home)} (rows) × ${esc(f.away)} (cols)</div>
            <div class="heat" style="grid-template-columns:auto repeat(${n},1fr)">${cells.join('')}</div>`;
  }

  function propsInPlay(f) {
    const list = propsByEvent.get(f.event) || [];
    const body = list.length ? list.map(p => `
      <button type="button" class="prop-jump" data-prop="${esc(p.id)}"
              title="p ${pct(p.p)} · O ${num(p.odds)} · ${esc(p.book || '')} · E ${num(p.e, 4)}">
        <div class="edgebar-row">
          <div class="name" style="width:220px">${esc(shortProp(p.proposition))}</div>
          ${edgeBar(p.p, p.odds)}
          <div class="e">${pct(p.selection_frequency, 0)}</div>
        </div>
      </button>`).join('')
      : `<p class="empty" style="padding:12px 0">No proposition from this fixture cleared
         E ≥ 1 and (P, O) dominance, so it appears in no portfolio.</p>`;
    return `<div class="comp-label">Propositions in play (survived to portfolio search)</div>
            <div class="props-list">${body}</div>
            <div class="edgebar-key"><span class="k-fill">probability fill</span>
            <span class="k-tick">market-implied tick</span></div>`;
  }

  function statCards(f) {
    const lc = f.league_context, m = f.match_markets;
    const rows = [
      ['Home', 'home_win'], ['Draw', 'draw'], ['Away', 'away_win'],
      ['BTTS yes', 'btts_yes'], ['BTTS no', 'btts_no'],
    ];
    return `<div class="cards">${rows.map(([label, key]) => {
      const v = m[key];
      const base = lc && lc.rates ? lc.rates[key] : null;
      const d = base == null ? '' :
        `<div class="d ${v >= base ? 'up' : 'down'}">${pp(v - base)} pp vs league</div>`;
      return `<div class="card"><div class="k">${label}</div><div class="v">${pct(v, 1)}</div>${d}</div>`;
    }).join('')}</div>`;
  }

  /* Five columns, because two quantities share this block: every line has a
     model probability, and only the lines a book actually quoted have an edge.
     Printing whichever one exists into a single column made 0.96 and 72% sit
     under each other meaning different things. */
  function ladderRow(f, line) {
    const price = priceByKey.get(f.event + '|' + line.label);
    const o = price ? price.odds : null;
    const d = line.league_avg == null ? '<div class="avg"></div>' :
      `<div class="avg ${line.p >= line.league_avg ? 'up' : 'down'}">${pp(line.p - line.league_avg)}</div>`;
    return `<div class="edgebar-row" title="${esc(line.label)}${price ? ` · O ${num(o)} · ${esc(price.book || '')}` : ' · not priced'}">
      <div class="name">O${line.line}</div>${edgeBar(line.p, o)}
      <div class="e">${pct(line.p, 0)}</div>
      <div class="e edge${price && price.e >= 1 ? ' up' : price ? ' down' : ''}">${price ? num(price.e, 2) : '·'}</div>
      ${d}</div>`;
  }

  function ladders(f) {
    return `<div class="ladders">${PRED.targets.map(t => `
      <div class="ladder">
        <h4>${esc(t === 'sot' ? 'Shots on target' : t)}</h4>
        ${['home', 'away'].map(scope => `
          <div class="side">
            <div class="team">${esc(f[scope])} · ${num(f.projections[scope][t])} expected</div>
            <div class="edgebar-row lad-head">
              <div class="name">line</div><div class="edgebar head"></div>
              <div class="e">model</div><div class="e">E</div><div class="avg">vs lg</div>
            </div>
            ${(f.lines[scope][t] || []).map(l => ladderRow(f, l)).join('')}
          </div>`).join('')}
      </div>`).join('')}</div>`;
  }

  function renderMatch() {
    renderTickets();
    const f = fixtureByEvent.get(state.event);
    if (!f) { el('match-body').innerHTML = '<p class="empty">No fixture selected.</p>'; return; }
    const lc = f.league_context;
    el('match-body').innerHTML = `
      <div class="match-head">
        <span class="code">${esc(f.event)}</span>
        <h2>${esc(f.home)} v ${esc(f.away)}</h2>
        <span class="code">${esc(f.league)} · ${esc(f.date)}</span>
        <span class="xg">xG ${num(f.exp_goals)}</span>
        ${lc ? `<span class="code">league: ${int(lc.matches)} matches over ${lc.seasons} seasons,
                ${num(lc.avg_goals)} goals/match</span>` : ''}
      </div>
      <div class="match-detail"><div>${heatmap(f)}</div><div>${propsInPlay(f)}</div></div>
      ${statCards(f)}
      <div class="section-title">Lines — model probability, market price where one exists</div>
      ${ladders(f)}`;
  }

  // --- portfolio book: filter rail ---------------------------------------

  function faderHTML(f, i) {
    const field = FIELD_BY_KEY.get(f.key);
    const b = BOUNDS.get(f.key);
    const step = field.kind === 'int' ? 1 : (b.hi - b.lo) / 500 || 1e-6;
    const span = (b.hi - b.lo) || 1;
    const left = ((f.min - b.lo) / span) * 100, right = ((b.hi - f.max) / span) * 100;
    // Two inputs stacked on one track: whichever is painted on top wins the
    // pointer. Once both handles sit in the upper half they overlap, and the
    // `hi` input -- later in the DOM -- swallows every grab at the `lo` one,
    // which reads as a slider that has stopped working. So the one nearer the
    // far end is lifted above the other.
    const loOnTop = f.min > (b.lo + b.hi) / 2;
    return `<div class="fader" data-fader="${i}">
      <div class="row1">
        <span>${esc(field.label)}</span>
        <span><span class="val">${fmtField(field, f.min)} – ${fmtField(field, f.max)}</span>
        ${f.custom ? `<button type="button" class="drop" data-drop="${i}" title="remove filter">×</button>` : ''}</span>
      </div>
      <div class="track">
        <div class="range" style="left:${left}%;right:${right}%"></div>
        <input type="range" data-h="lo" data-i="${i}" min="${b.lo}" max="${b.hi}" step="${step}" value="${f.min}"
               style="z-index:${loOnTop ? 3 : 1}" aria-label="${esc(field.label)} minimum">
        <input type="range" data-h="hi" data-i="${i}" min="${b.lo}" max="${b.hi}" step="${step}" value="${f.max}"
               style="z-index:2" aria-label="${esc(field.label)} maximum">
      </div>
    </div>`;
  }

  /* Twenty fixtures, collapsed, rather than 128 propositions in one flat run.
     The event row is the control most of the time -- a whole fixture in or out
     is one click -- and the individual propositions are there when the question
     is finer than that. */
  function eventMode(ev) {
    const set = state.events.get(ev);
    if (set) return set;
    // No fixture-level setting, but some of its propositions carry their own.
    const all = propsByEvent.get(ev) || [];
    return all.some(p => state.props.has(p.id)) ? 'mixed' : 'none';
  }

  function triButtons(attr, id, mode) {
    return `<div class="tri-btns">
      <button type="button" ${attr}="${esc(id)}" data-mode="in"
        class="${mode === 'in' ? 'active-in' : ''}">IN</button>
      <button type="button" ${attr}="${esc(id)}" data-mode="none"
        class="${mode === 'none' ? 'active-none' : ''}">—</button>
      <button type="button" ${attr}="${esc(id)}" data-mode="out"
        class="${mode === 'out' ? 'active-out' : ''}">OUT</button>
    </div>`;
  }

  function propListHTML() {
    const q = state.search.trim().toLowerCase();
    const blocks = [];
    for (const ev of [...propsByEvent.keys()].sort()) {
      const all = propsByEvent.get(ev);
      const f = fixtureByEvent.get(ev);
      const hay = p => (p.proposition + ' ' + p.fixture + ' ' + p.event).toLowerCase();
      const list = q ? all.filter(p => hay(p).includes(q)) : all;
      if (!list.length) continue;

      // A search is a request to see what matched, so it opens the events it hit.
      const open = q ? true : state.expanded.has(ev);
      const mode = eventMode(ev);
      const top = all.length ? all[0].selection_frequency : 0;   // sorted desc at load

      blocks.push(`<div class="ev-block${mode === 'in' ? ' is-in' : mode === 'out' ? ' is-out' : ''}">
        <div class="tristate ev-head">
          <button type="button" class="ev-toggle" data-toggle="${esc(ev)}"
                  aria-expanded="${open}">
            <span class="chev">${open ? '▾' : '▸'}</span>
            <span class="propname">${f ? esc(f.home) + ' v ' + esc(f.away) : esc(ev)}
              <span class="ev">${esc(ev)} · ${list.length}${q ? ' of ' + all.length : ''} props ·
                most picked ${pct(top, 0)}</span>
            </span>
          </button>
          ${triButtons('data-ev', ev, mode)}
        </div>
        ${open ? `<div class="ev-props">${list.map(p => `
          <div class="tristate">
            <div class="propname">${esc(shortProp(p.proposition))}
              <span class="ev">E ${num(p.e, 2)} · picked ${pct(p.selection_frequency, 0)}</span>
            </div>
            ${triButtons('data-prop-set', p.id, state.props.get(p.id) || 'none')}
          </div>`).join('')}</div>` : ''}
      </div>`);
    }
    return blocks.join('') || '<p class="empty" style="padding:12px 0">Nothing matches that search.</p>';
  }

  /** Repaint just the list, keeping where you were scrolled to in it. */
  function refreshPropList() {
    const box = el('prop-list');
    const top = box.scrollTop;
    box.innerHTML = propListHTML();
    box.scrollTop = top;
  }

  function renderRail() {
    const group = (title, keys) => {
      const items = state.faders
        .map((f, i) => ({ f, i }))
        .filter(({ f }) => keys.includes(f.key) && !f.custom);
      return items.length
        ? `<div><h4>${title}</h4>${items.map(({ f, i }) => faderHTML(f, i)).join('')}</div>` : '';
    };
    const customs = state.faders.map((f, i) => ({ f, i })).filter(({ f }) => f.custom);

    el('rail').innerHTML =
      group('Return &amp; risk', ['expected_return_pct', 'sd_pct', 'variance'])
      + group('Probability of profit', BOOK.thresholds.map(t => 'p_over_' + t))
      + group('Legs', ['legs'])
      + (HAS_LEG_P ? group('Legs&rsquo; own odds', ['median_leg_p']) : '')
      // `group` filters on an explicit key list, so a new FIELDS entry gets state
      // and bounds but no control until it is named here.
      + (HAS_CAPACITY ? group('Edge capacity',
          ['capacity', 'n_eff', 'max_leg_stake']) : '')
      + (HAS_GROWTH ? group('Stake sizing',
          ['growth_f_suggested', 'growth_g_suggested', 'growth_p0']) : '')
      + `<div><h4>Split</h4><div class="split-picks">${SPLIT_KEYS.map(k =>
          `<button type="button" data-split="${k}" class="${state.splits.has(k) ? 'on' : ''}">${esc(BOOK.splits[k])}</button>`
        ).join('')}</div></div>`
      + (customs.length ? `<div><h4>Custom filters</h4>${customs.map(({ f, i }) => faderHTML(f, i)).join('')}</div>` : '')
      + `<div><button type="button" class="add-filter" id="add-filter">+ Add filter</button></div>`
      + `<div><h4>Propositions</h4>
           <input type="search" class="prop-search" id="prop-search" placeholder="search propositions…"
                  value="${esc(state.search)}">
           <div class="prop-list" id="prop-list">${propListHTML()}</div>
         </div>`;
  }

  // --- portfolio book: table ---------------------------------------------

  const COLUMNS = [
    { key: 'id', label: 'ID', cell: p => String(p.id) },   // an id is a name, not a quantity: no thousands separator
    { key: 'split', label: 'Split', cell: p =>
        `<span class="pill ${p.split === 'growth' ? 'growth' : p.split === 'min_variance' ? 'min-var' : 'inv-e'}">${esc(BOOK.splits[p.split])}</span>` },
    { key: 'legs', label: 'Legs', cell: p => String(p.legs) },
    ...(HAS_LEG_P ? [{ key: 'median_leg_p', label: 'Median leg P',
                      cell: p => pct(p.median_leg_p) }] : []),
    { key: 'expected_return_pct', label: 'Return %', cls: 'ret', cell: p => pct(p.expected_return_pct) },
    { key: 'sd_pct', label: 'SD %', cell: p => pct(p.sd_pct) },
    ...BOOK.thresholds.map(t => ({ key: 'p_over_' + t, label: `P(>${t}%)`, cell: p => pct(p['p_over_' + t]) })),
    ...(HAS_CAPACITY ? [
      { key: 'capacity', label: 'C', cell: p => num(p.capacity, 3) },
      { key: 'n_eff', label: 'Eff. legs', cell: p => num(p.n_eff, 1) },
    ] : []),
    ...(HAS_GROWTH ? [
      // The stake sits immediately left of the growth IT produces. There used to
      // be two of each, and the pairing invited reading a growth rate from one
      // stake against the other -- both figures right, the comparison wrong.
      { key: 'growth_f_suggested', label: 'Stake %',
        cell: p => pct(p.growth_f_suggested) },
      { key: 'growth_g_suggested', label: 'Growth %', cell: p => pct(p.growth_g_suggested) },
      { key: 'growth_p0', label: 'P(0)', cell: p => pctFloor(p.growth_p0) },
    ] : []),
  ];

  let visible = [];           // current filtered+sorted rows
  let activeCs = [];          // the constraints that produced it
  let expandedH = 0;          // measured height of the open legs panel

  const ROW_H = 42;

  function rowTop(i) {
    const openIdx = state.open == null ? -1 : visible.findIndex(p => p.id === state.open);
    let y = i * ROW_H;
    if (openIdx >= 0 && i > openIdx) y += expandedH;
    return y;
  }

  function firstVisible(scrollTop) {
    const openIdx = state.open == null ? -1 : visible.findIndex(p => p.id === state.open);
    if (openIdx < 0 || scrollTop <= (openIdx + 1) * ROW_H) return Math.floor(scrollTop / ROW_H);
    return Math.floor(Math.max(0, scrollTop - expandedH) / ROW_H);
  }

  function legsPanel(pf) {
    const legs = pf.picks.map((id, k) => ({ prop: propById.get(id), stake: pf.stakes ? pf.stakes[k] : null }));
    return `<div class="legs-panel">
      <div class="legs-head">
        <span>portfolio ${esc(refOf(pf))} · ${esc(BOOK.splits[pf.split])} · ${pf.legs} legs</span>
        <span>return ${pct(pf.expected_return_pct)} · sd ${pct(pf.sd_pct)} · P(profit) ${pct(pf['p_over_100'])}
          ${pf.projection ? `<button type="button" class="to-match go-proj"
             data-project="${pf.id}">projection →</button>` : ''}</span>
      </div>
      <table><tbody>${legs.map(({ prop, stake }) => prop ? `
        <tr>
          <td>${esc(prop.event)}</td>
          <td class="prop">${esc(prop.proposition)}</td>
          <td class="bar">${edgeBar(prop.p, prop.odds)}</td>
          <td>${num(prop.e, 4)}</td>
          <td>${num(prop.odds)}</td>
          <td>${esc(prop.book || '—')}</td>
          <td class="cash">${stake == null ? '—' : money(stake * state.stake)}</td>
          <td><button type="button" class="to-match" data-goto="${esc(prop.event)}">match board →</button></td>
        </tr>` : '').join('')}</tbody></table>
    </div>`;
  }

  function renderHead() {
    el('pf-head').innerHTML = COLUMNS.map(c => {
      const on = state.sort.key === c.key;
      const dir = on ? `<span class="dir">${state.sort.dir < 0 ? '▾' : '▴'}</span>` : '';
      return `<th tabindex="0" data-sort="${c.key}">${esc(c.label)} ${dir}</th>`;
    }).join('') + '<th></th>';
  }

  /** Only the rows on screen go into the DOM; the rest are two spacer rows.
   *  At five thousand rows and eleven columns a full render is tens of
   *  thousands of nodes on every keystroke in the filter rail. */
  function renderRows() {
    const scroller = el('pf-scroll'), body = el('pf-body');
    const ncols = COLUMNS.length + 1;

    if (!visible.length) {
      const culprit = likelyCulprit(activeCs);
      body.innerHTML = `<tr><td colspan="${ncols}"><div class="empty">
        Nothing matches these filters.
        ${culprit ? `The one most likely responsible is <strong>${esc(culprit.label)}</strong> —
           relaxing just that would show ${int(culprit.n)} portfolios.` : ''}
        <div><button type="button" id="reset-inline">reset filters</button></div>
      </div></td></tr>`;
      return;
    }

    const top = scroller.scrollTop;
    const openIdx = state.open == null ? -1 : visible.findIndex(p => p.id === state.open);
    const total = visible.length * ROW_H + (openIdx >= 0 ? expandedH : 0);
    const first = Math.max(0, firstVisible(top) - 6);
    const last = Math.min(visible.length - 1, firstVisible(top + scroller.clientHeight) + 6);

    const parts = [`<tr class="spacer"><td colspan="${ncols}" style="height:${rowTop(first)}px"></td></tr>`];
    for (let i = first; i <= last; i++) {
      const p = visible[i];
      const open = p.id === state.open;
      parts.push(`<tr class="row${open ? ' open' : ''}" tabindex="0" data-id="${p.id}">`
        + COLUMNS.map(c => `<td${c.cls ? ` class="${c.cls}"` : ''}>${c.cell(p)}</td>`).join('')
        + `<td class="arrow">${open ? '↓' : '→'}</td></tr>`);
      if (open) parts.push(`<tr class="legs"><td colspan="${ncols}">${legsPanel(p)}</td></tr>`);
    }
    parts.push(`<tr class="spacer"><td colspan="${ncols}" style="height:${Math.max(0, total - rowTop(last + 1))}px"></td></tr>`);
    body.innerHTML = parts.join('');

    if (openIdx >= 0) {
      const panel = body.querySelector('tr.legs');
      if (panel) {
        const h = panel.getBoundingClientRect().height;
        if (Math.abs(h - expandedH) > 1) { expandedH = h; renderRows(); }
      }
    }
  }

  function renderBook() {
    const { rows, cs } = applyFilters();
    visible = rows;
    activeCs = cs;
    if (state.open != null && !visible.some(p => p.id === state.open)) { state.open = null; expandedH = 0; }
    el('pf-count').innerHTML =
      `<strong>${int(visible.length)}</strong> of ${int(portfolios.length)} undominated portfolios`;
    renderHead();
    renderRows();
  }

  // --- projection ---------------------------------------------------------
  //
  // The Portfolio Book's columns describe one settlement. This page describes
  // the same portfolio played every week and reinvested, which is a different
  // question -- and the one `fpp.growth` already answers.
  //
  // Everything drawn here is *read* from the payload's `projection` block: the
  // `g(f)` curve, the percentile fan and the outcome histogram all come off one
  // `growth_metrics` call, so the curve passes through the `g_star` printed
  // beside it by construction rather than by two implementations happening to
  // agree. What this file adds is the arithmetic Python could not do, because
  // only the browser knows the pot: `pot * exp(log wealth)`.

  const AXES = BOOK.projection_axes || null;
  /* The settings that decide the suggested stake. Rendered rather than assumed,
     because every one of them is a choice: a reader who cannot see the drawdown
     tolerance cannot tell a recommendation from a house rule. Older payloads
     have no `risk` block and simply say nothing. */
  const RISK = BOOK.risk || null;
  const riskSentence = () => {
    if (!RISK) return '';
    const bits = [`stake keeps P(down ${pct(RISK.drawdown_d, 0)} in ${int(RISK.horizon_rounds)}`
                  + ` rounds) under ${pct(RISK.drawdown_p, 0)}`];
    if (RISK.pessimism_b) bits.push(`less ${pp(RISK.pessimism_b)}pp on every leg`);
    if (RISK.slate_tau) bits.push(`slate wobble ${pp(RISK.slate_tau)}pp`);
    bits.push(`max ${pct(RISK.max_leg_stake, 0)} a leg`);
    return bits.join(' · ');
  };
  const portfolioById = new Map(portfolios.map(pf => [pf.id, pf]));
  const hasProjection = !!AXES && portfolios.some(pf => pf.projection);

  const RATES = [
    { key: 'suggested', fk: 'growth_f_suggested', gk: 'growth_g_suggested',
      label: 'suggested', cls: 'r-prot' },
  ];

  /* The betting slip prices at one of two fractions of the pot. The first is the
     model's own answer and carries the palette it carries everywhere else on this
     page; the second is plain ink because it is a choice, not an answer.
     `custom` reads `state.customF` -- the same fraction the curve marks as
     `yours` and the calculator gives a row to, so the panels can never disagree
     about what the reader asked for.

     There used to be a third, `Max`, priced at the growth optimum. It sat at its
     own ceiling on two thirds of portfolios, so offering it as a choice was
     offering a constant dressed as an answer. */
  const STAKE_MODES = [
    { key: 'suggested', label: 'Suggested', fk: 'growth_f_suggested', cls: 'r-prot' },
    { key: 'custom', label: 'Custom', fk: null, cls: 'r-custom' },
  ];
  const stakeMode = () => STAKE_MODES.find(m => m.key === state.stakeMode) || STAKE_MODES[0];
  const modeF = pf => {
    const m = stakeMode();
    return m.key === 'custom' ? state.customF : pf[m.fk];
  };

  /* House order, consulted only to settle a tie the minimum has already left
     open. The spellings are the display names `staking.BOOKS` writes into the
     payload -- '10bet' and 'BoyleSports', not '10Bet' or 'Boyle Sports'. */
  const BOOK_ORDER = ['10bet', 'BoyleSports', 'BetMGM', 'Virgin Bet', 'Paddy Power', 'Bet365'];
  const bookRank = b => { const i = BOOK_ORDER.indexOf(b); return i < 0 ? BOOK_ORDER.length : i; };
  /** Rank first, then name, so a book nobody listed still sorts the same way twice. */
  const byBook = (a, b) => bookRank(a) - bookRank(b) || (a < b ? -1 : a > b ? 1 : 0);
  /** Lexicographic on rank, for two candidate sets of the same size. Both arrive
   *  already rank-sorted, because `universe` is and `filter` preserves order. */
  const betterRanks = (a, b) => {
    for (let i = 0; i < a.length; i++) { const d = byBook(a[i], b[i]); if (d) return d < 0; }
    return false;
  };

  /** One bookmaker per leg, chosen to open the fewest accounts.
   *
   *  `staking.best_price` names *every* book that matched the top price rather
   *  than picking one, so a leg reading 'BetMGM / Virgin Bet' is a free choice
   *  and this is where it gets made. Minimum accounts decides first and
   *  `BOOK_ORDER` only breaks what is left -- a preference that cost an extra
   *  account would be the preference deciding, which is not what it is for.
   *
   *  What this deliberately cannot do: the payload carries the best price and
   *  who matched it, not the ladder behind it, so no leg is ever moved to a
   *  worse price to save an account. Every price on the slip is still the best
   *  price that leg was quoted.
   */
  function chooseBooks(legs) {
    const cands = legs.map(({ prop }) => !prop || !prop.book ? []
      : String(prop.book).split('/').map(t => t.trim()).filter(Boolean));
    const need = cands.map((c, i) => [i, c]).filter(([, c]) => c.length);
    const universe = [...new Set(cands.flat())].sort(byBook);
    const out = new Map();
    if (!need.length) return out;

    const covers = (set, c) => c.some(b => set.includes(b));
    let chosen;
    if (universe.length <= 16) {
      // Six books on real data, so 63 subsets. Smallest covering set wins;
      // among equals, the one whose ranks read best from the left.
      let best = null;
      for (let m = 1; m < (1 << universe.length); m++) {
        const set = universe.filter((_, b) => m & (1 << b));
        if (best && set.length > best.length) continue;
        if (!need.every(([, c]) => covers(set, c))) continue;
        if (!best || set.length < best.length || betterRanks(set, best)) best = set;
      }
      chosen = best || [];
    } else {
      // Never reached with today's six books, and not a reason to hang if the
      // book list ever grows: take the book covering the most open legs, ties
      // to the preferred one because `universe` is already rank-sorted.
      chosen = [];
      let open = need.map(([, c]) => c);
      while (open.length) {
        let pick = null, n = 0;
        for (const b of universe) {
          const k = open.filter(c => c.includes(b)).length;
          if (k > n) { n = k; pick = b; }
        }
        if (!pick) break;
        chosen.push(pick);
        open = open.filter(c => !covers(chosen, c));
      }
    }

    for (const [i, c] of need) {
      const hit = c.filter(b => chosen.includes(b)).sort(byBook);
      if (hit.length) out.set(i, hit[0]);
    }
    return out;
  }

  /* One inline-SVG line chart, shared by both plots. There is no charting
     library in this bundle and adding one would be its only external
     dependency, so this stays deliberately small: a linear or log y axis, one
     path per series, and a `null` **breaks** the path rather than bridging it.
     That last part is the whole reason the exporter writes nulls -- a bridged
     gap is a line through a stake the model calls terminal. */
  const CH = { w: 760, h: 300, l: 66, r: 14, t: 12, b: 32 };

  function niceTicks(lo, hi, n) {
    const span = hi - lo;
    if (!(span > 0)) return [lo];
    const mag = Math.pow(10, Math.floor(Math.log10(span / n)));
    const step = ([1, 2, 2.5, 5, 10].find(m => m * mag >= span / n) || 10) * mag;
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-6; v += step) out.push(v);
    return out;
  }

  /* Decade ticks, thinned to stay readable. `g*` compounded over 100 rounds
     spans a dozen decades on real portfolios -- which is the honest answer and
     the reason this axis is logarithmic -- so a tick per decade would be 24
     labels down the side. */
  function logTicks(lo, hi, max = 7) {
    const e0 = Math.floor(Math.log10(lo)), e1 = Math.ceil(Math.log10(hi));
    const step = Math.max(1, Math.ceil((e1 - e0) / max));
    const out = [];
    for (let e = e0; e <= e1; e += step) {
      const v = Math.pow(10, e);
      if (v >= lo * 0.999 && v <= hi * 1.001) out.push(v);
    }
    return out.length > 1 ? out : [lo, hi];
  }

  /** Money at any magnitude. `money()` is right for a stake and unreadable for
   *  a hundred rounds of compounding, which runs to twelve figures. */
  const shortMoney = v => {
    const a = Math.abs(v);
    if (!isFinite(v)) return '—';
    if (a >= 1e12) return (v / 1e12).toFixed(1) + 'T';
    if (a >= 1e9) return (v / 1e9).toFixed(1) + 'B';
    if (a >= 1e6) return (v / 1e6).toFixed(1) + 'M';
    if (a >= 1e3) return (v / 1e3).toFixed(a >= 1e4 ? 0 : 1) + 'k';
    return v.toFixed(a >= 10 ? 0 : 2);
  };
  const cash = v => (v == null || !isFinite(v) ? '—' : Math.abs(v) >= 1e7 ? shortMoney(v) : money(v));

  function lineChart({ x, series, areas = [], ylog = false, yfmt = num, xfmt = num,
                       marks = [], rule = null }) {
    const live = v => v != null && isFinite(v) && (!ylog || v > 0);
    const vals = series.flatMap(s => s.values).filter(live)
      .concat(rule != null && live(rule) ? [rule] : []);
    if (!vals.length) return `<div class="chart-empty">nothing to plot here</div>`;
    let lo = Math.min(...vals), hi = Math.max(...vals);
    if (hi === lo) { const pad = Math.abs(lo || 1) * 0.1; hi += pad; lo -= pad; }
    if (ylog && lo <= 0) lo = hi / 1e6;

    const xlo = x[0], xhi = x[x.length - 1];
    const t = ylog ? Math.log : (v => v);
    const px = v => CH.l + (v - xlo) / ((xhi - xlo) || 1) * (CH.w - CH.l - CH.r);
    const py = v => CH.h - CH.b
      - (t(v) - t(lo)) / ((t(hi) - t(lo)) || 1) * (CH.h - CH.t - CH.b);

    const path = vs => {
      let d = '', pen = false;
      vs.forEach((v, i) => {
        if (!live(v)) { pen = false; return; }
        d += `${pen ? 'L' : 'M'}${px(x[i]).toFixed(1)} ${py(v).toFixed(1)} `;
        pen = true;
      });
      return d.trim();
    };
    // An area is only drawn across the run where *both* edges are live, for the
    // same reason a line breaks: a filled band over a gap asserts a range that
    // was never computed.
    const area = (a, b) => {
      let d = '', run = [];
      const flush = () => {
        if (run.length < 2) { run = []; return; }
        d += 'M' + run.map(([xi, ai]) => `${px(xi).toFixed(1)} ${py(ai).toFixed(1)}`).join(' L');
        d += ' L' + run.slice().reverse()
          .map(([xi, , bi]) => `${px(xi).toFixed(1)} ${py(bi).toFixed(1)}`).join(' L') + ' Z ';
        run = [];
      };
      x.forEach((xi, i) => {
        if (live(a[i]) && live(b[i])) run.push([xi, a[i], b[i]]); else flush();
      });
      flush();
      return d.trim();
    };

    /* Marker labels stack rather than overprint. `f*` and a protective stake can
       land one grid step apart -- 1% of a 99% axis -- and two labels on the same
       baseline there are illegible at exactly the point the reader is looking. */
    const placed = marks.map(m => ({ ...m, x: px(m.at) })).sort((a, b) => a.x - b.x);
    let lastX = -Infinity, row = 0;
    for (const m of placed) {
      row = (m.x - lastX) < 52 ? row + 1 : 0;
      m.row = row;
      lastX = m.x;
    }

    const yt = (ylog ? logTicks(lo, hi) : niceTicks(lo, hi, 5)).map(v => `
      <line class="grid" x1="${CH.l}" x2="${CH.w - CH.r}" y1="${py(v).toFixed(1)}" y2="${py(v).toFixed(1)}"/>
      <text class="ax y" x="${CH.l - 8}" y="${(py(v) + 3.5).toFixed(1)}">${esc(yfmt(v))}</text>`).join('');
    const xt = niceTicks(xlo, xhi, 6).filter(v => v >= xlo && v <= xhi).map(v => `
      <text class="ax x" x="${px(v).toFixed(1)}" y="${CH.h - CH.b + 18}">${esc(xfmt(v))}</text>`).join('');

    return `<svg class="chart" viewBox="0 0 ${CH.w} ${CH.h}" role="img">
      ${yt}${xt}
      ${rule != null && live(rule) ? `<line class="rule" x1="${CH.l}" x2="${CH.w - CH.r}"
        y1="${py(rule).toFixed(1)}" y2="${py(rule).toFixed(1)}"/>` : ''}
      ${placed.map(m => `<line class="mark ${m.cls}" x1="${m.x.toFixed(1)}" x2="${m.x.toFixed(1)}"
        y1="${CH.t}" y2="${CH.h - CH.b}"/>
        <text class="mark-l ${m.cls}" x="${m.x.toFixed(1)}"
          y="${CH.t + 10 + m.row * 11}">${esc(m.label)}</text>`).join('')}
      ${areas.map(a => `<path class="area ${a.cls}" d="${area(a.from, a.to)}"/>`).join('')}
      ${series.map(s => `<path class="ln ${s.cls}" d="${path(s.values)}"/>`).join('')}
      <line class="axis" x1="${CH.l}" x2="${CH.w - CH.r}" y1="${CH.h - CH.b}" y2="${CH.h - CH.b}"/>
    </svg>`;
  }

  // --- reading the exported curves ----------------------------------------

  /** `g` at a stake fraction. A read-off, not an interpolation: the exported
   *  grid is `GROWTH_F_FINE`, the same resolution `f*` was chosen at. */
  function gAt(pr, f) {
    const i = Math.round(f * 100) - 1;
    return (i >= 0 && i < pr.g_curve.length) ? pr.g_curve[i] : null;
  }

  /** The **lower** of the two stakes that earn `g`.
   *
   *  `g(f)` is concave through the origin, so every rate below the peak is
   *  earned twice — once climbing and once falling — and the higher stake is
   *  the same growth for strictly more risk. The panel says so rather than
   *  quietly picking one. */
  function fAt(pr, g) {
    const c = pr.g_curve;
    for (let i = 0; i < c.length; i++) {
      if (c[i] == null || c[i] < g) continue;
      if (i === 0) return AXES.f[0];
      const a = c[i - 1], b = c[i];
      return AXES.f[i - 1] + (b === a ? 0 : (g - a) / (b - a)) * (AXES.f[i] - AXES.f[i - 1]);
    }
    return null;
  }

  /** A band's log wealth at any round, straight-line between exported points.
   *
   *  The exporter ships ~25 log-spaced rounds rather than all 100 because log
   *  wealth is close to straight in `n`; this is the interpolation that trade
   *  assumes. Past the last exported round it clamps rather than extrapolating,
   *  and `roundsNote` tells the reader when that has happened. */
  function bandAt(band, key, n) {
    const v = band && band[key];
    if (!v) return null;
    const R = AXES.rounds;
    if (n <= R[0]) return v[0];
    for (let i = 1; i < R.length; i++) {
      if (n <= R[i]) return v[i - 1] + (n - R[i - 1]) / (R[i] - R[i - 1]) * (v[i] - v[i - 1]);
    }
    return v[v.length - 1];
  }

  const potOf = lw => (lw == null ? null : state.pot * Math.exp(lw));

  // --- the panels ---------------------------------------------------------

  function fanPanel(pf, pr) {
    const x = AXES.rounds;
    const series = [], areas = [];
    for (const r of RATES) {
      const b = pr.bands[r.key];
      if (!b) continue;
      areas.push({ cls: r.cls, from: b.p5.map(potOf), to: b.p95.map(potOf) });
      series.push({ cls: `${r.cls} p50`, values: b.p50.map(potOf) });
      series.push({ cls: `${r.cls} p95`, values: b.p95.map(potOf) });
      series.push({ cls: `${r.cls} p5`, values: b.p5.map(potOf) });
    }
    if (!series.length) {
      return `<div class="chart-empty">This portfolio has no plottable stake —
              every fraction it could be played at wipes the pot out on a reachable round.</div>`;
    }
    const legend = RATES.filter(r => pr.bands[r.key]).map(r => `
      <span class="key ${r.cls}"><i></i>${esc(r.label)} · stake ${pct(pf[r.fk], 0)}
      · ${cash(state.pot * pf[r.fk])}/round</span>`).join('');
    return lineChart({
      x, series, areas, ylog: true, rule: state.pot,
      yfmt: shortMoney,
      xfmt: v => v.toFixed(0),
    }) + `<div class="chart-foot"><div class="keys">${legend}</div>
      <span class="axis-note">rounds → · pot on a log scale · flat line is the pot you started with</span></div>`;
  }

  function curvePanel(pf, pr) {
    const marks = RATES.filter(r => pf[r.fk] != null)
      .map(r => ({ at: pf[r.fk], cls: r.cls, label: r.label }));
    if (state.customF != null) marks.push({ at: state.customF, cls: 'r-custom', label: 'yours' });
    const chart = lineChart({
      x: AXES.f, series: [{ cls: 'r-curve', values: pr.g_curve }], marks, rule: 0,
      yfmt: v => (v * 100).toFixed(1) + '%', xfmt: v => (v * 100).toFixed(0) + '%',
    });
    const f = state.customF;
    const g = f == null ? null : gAt(pr, Math.round(f * 100) / 100);
    const read = f == null
      ? `<p class="hint">Type a stake or a growth rate — each fills the other.</p>`
      : `<p class="read"><strong>${pct(f, 0)}</strong> of the pot
         (${cash(state.pot * f)} a round) grows at <strong>${pct(g, 2)}</strong> per round.
         ${g != null && g < 0 ? '<em>Negative — this stake loses money despite the edge.</em>'
           : f > pf.growth_f_suggested
             ? `<em>Above the suggested ${pct(pf.growth_f_suggested, 0)} — more growth, and more drawdown than the risk setting allows.</em>`
             : ''}</p>`;
    return chart + `<div class="chart-foot"><span class="axis-note">stake % of pot →
      · growth per round · curve peaks at g*</span></div>${read}
      <p class="caveat">Every rate below the peak is earned at two stakes. The read-off
      returns the lower one — the higher is the same growth for strictly more risk.</p>`;
  }

  function calcPanel(pf, pr) {
    const n = Math.max(1, state.rounds || 1);
    const target = state.target || null;
    const rows = RATES.map(r => ({ ...r, g: pf[r.gk], band: pr.bands[r.key] }));
    if (state.customF != null) {
      rows.push({ key: 'custom', label: `yours · ${pct(state.customF, 0)}`, cls: 'r-custom',
                  g: gAt(pr, Math.round(state.customF * 100) / 100), band: null });
    }
    const beyond = n > AXES.rounds[AXES.rounds.length - 1];
    const body = rows.map(r => {
      if (r.g == null || !isFinite(r.g)) {
        return `<tr><td class="k ${r.cls}">${esc(r.label)}</td><td colspan="3">—</td></tr>`;
      }
      const med = state.pot * Math.exp(r.g * n);
      const lo = potOf(bandAt(r.band, 'p5', n)), hi = potOf(bandAt(r.band, 'p95', n));
      const need = target && target > 0 && r.g > 0 ? Math.log(target / state.pot) / r.g : null;
      return `<tr>
        <td class="k ${r.cls}">${esc(r.label)}</td>
        <td class="mono">${pct(r.g, 2)}</td>
        <td class="mono cash">${cash(med)}
          ${lo != null ? `<span class="band">${cash(lo)} – ${cash(hi)}</span>`
                       : `<span class="band">median only</span>`}</td>
        <td class="mono">${need == null ? '—' : need <= 0 ? 'already there'
          : need.toFixed(1) + ' rounds'}</td>
      </tr>`;
    }).join('');
    return `<table class="calc"><thead><tr>
        <th>rate</th><th>per round</th><th>pot after ${int(n)}</th><th>rounds to target</th>
      </tr></thead><tbody>${body}</tbody></table>
      <p class="caveat">The pot column is the <strong>median</strong> path, with the p5–p95
      band beneath it — not a forecast, and not an average. Half of all futures land outside
      that band's edges combined.${beyond ? ' Past round '
        + int(AXES.rounds[AXES.rounds.length - 1]) + ' the band is held flat rather than extrapolated.'
        : ''}</p>`;
  }

  /** The portfolio as instructions: what to back, where, and for how much.
   *
   *  Every other panel here describes the portfolio as a series. This one is
   *  the single settlement you are about to place, so it prices off the pot at
   *  the top of the page rather than off the Portfolio Book's stake --
   *  `stake = f * pot`, the identity the state comment sets out.
   */
  function slipPanel(pf) {
    const mode = stakeMode();
    const f = modeF(pf);
    const legs = pf.picks.map((id, k) =>
      ({ prop: propById.get(id), stake: pf.stakes ? pf.stakes[k] : null }));
    const books = chooseBooks(legs);

    /* Rounded to the penny *before* anything is added up. Summing the exact
       fractions and then rounding gives a total that is a penny or two off the
       rows printed above it, and a slip whose column does not add up is a slip
       you have to check by hand. */
    const p2 = v => Math.round(v * 100) / 100;
    const cashOf = st => (st == null || f == null ? null : p2(st * f * state.pot));

    const rows = legs.map(({ prop, stake }, k) => {
      if (!prop) return '';
      const b = books.get(k), c = cashOf(stake);
      return `<tr>
        <td class="sel"><span class="fx">${esc(prop.fixture || prop.event)}</span>
          ${esc(prop.proposition)}</td>
        <td>${num(prop.e, 4)}</td>
        <td>${num(prop.odds)}</td>
        <td class="bk">${b ? esc(b) : '—'}</td>
        <td class="cash">${c == null ? '—' : money(c)}</td>
      </tr>`;
    }).join('');

    const totals = new Map();
    legs.forEach(({ stake }, k) => {
      const b = books.get(k), c = cashOf(stake);
      if (!b || c == null) return;
      const cur = totals.get(b) || { n: 0, sum: 0 };
      totals.set(b, { n: cur.n + 1, sum: p2(cur.sum + c) });
    });
    const accounts = [...totals].sort((a, b) => byBook(a[0], b[0]));
    const total = accounts.reduce((t, [, v]) => p2(t + v.sum), 0);
    const priced = accounts.reduce((n, [, v]) => n + v.n, 0);

    // `legsPanel` guards `pf.stakes` because a portfolio predating the field has
    // none, and a slip that cannot price its legs should say so rather than
    // announce a confident total of nothing.
    const head = f == null || !accounts.length
      ? `<p class="hint">${f == null && mode.key === 'custom'
          ? 'Type a stake % above to price this slip.'
          : f == null ? `This portfolio has no ${esc(mode.label.toLowerCase())} stake.`
          : 'This portfolio carries no stake split, so its legs cannot be priced.'}</p>`
      : `<p class="slip-head"><strong class="${mode.cls}">${esc(mode.label)}</strong> ·
         ${pct(f, 0)} of ${cash(state.pot)} · <strong class="cash">${money(total)}</strong>
         across ${accounts.length} ${accounts.length === 1 ? 'account' : 'accounts'}
         ${RUN_CODE ? `<span class="slip-ref" title="what the ledger calls this portfolio"
           >${esc(refOf(pf))}</span>` : ''}</p>`;

    const acct = !accounts.length ? '' : `
      <div class="slip-sub">What to have where</div>
      <table class="accounts"><tbody>${accounts.map(([b, v]) => `
        <tr><td class="bk">${esc(b)}</td>
          <td>${v.n} ${v.n === 1 ? 'leg' : 'legs'}</td>
          <td class="cash">${money(v.sum)}</td></tr>`).join('')}
        <tr class="tot"><td>total</td><td>${priced} legs</td>
          <td class="cash">${money(total)}</td></tr>
      </tbody></table>`;

    return head + `<table class="slip"><thead><tr>
        <th>selection</th><th>edge</th><th>odds</th><th>book</th><th>stake</th>
      </tr></thead><tbody>${rows}</tbody></table>${acct}
      <p class="caveat">Books are picked to open the <strong>fewest accounts</strong>, never to
      give up a price: a leg quoted the same at two books goes to whichever one the rest of the
      slip already needs. Legs are rounded to the penny before anything is added up, so every
      total is the sum of the rows above it.</p>`;
  }

  function histPanel(pr) {
    const bars = pr.hist, max = Math.max(...bars);
    if (!(max > 0)) return '';
    return `<div class="hist">${bars.map((m, i) => `
      <i style="height:${(m / max * 100).toFixed(1)}%"
         title="return ${((i + 0.5) * pr.hist_step).toFixed(2)}× — ${pct(m, 2)}"></i>`).join('')}</div>
      <div class="hist-ax"><span>0</span><span>one round's return, ×stake</span>
      <span>${num(pr.max_return)}×</span></div>`;
  }

  function statsPanel(pf, pr) {
    const g = pf.growth || {};
    const cell = (k, v, cls = '') => `<div class="stat ${cls}"><dt>${esc(k)}</dt><dd>${v}</dd></div>`;
    const thresholds = BOOK.thresholds
      .map(t => cell(`P(>${t}%)`, pct(pf[`p_over_${t}`]))).join('');
    return `<dl class="stats">
      ${cell('split', esc(BOOK.splits[pf.split]))}
      ${cell('legs', int(pf.legs))}
      ${HAS_LEG_P ? cell('median leg P', pct(pf.median_leg_p)) : ''}
      ${cell('expected return', pct(pf.expected_return_pct))}
      ${cell('spread (sd)', pct(pf.sd_pct))}
      ${thresholds}
      ${cell('best case', `${num(pr.max_return)}×`)}
      ${cell('P(best case)', pctFloor(pr.p_max))}
      ${cell('P(returns nothing)', pctFloor(g.p0), 'warn')}
      ${pf.capacity != null ? cell('edge capacity C', num(pf.capacity, 3)) : ''}
      ${pf.n_eff != null ? cell('effective legs', num(pf.n_eff, 1)) : ''}
      ${pf.max_leg_stake != null ? cell('largest leg', pct(pf.max_leg_stake, 1)) : ''}
      ${cell('suggested stake', pct(pf.growth_f_suggested, 0) + ' · ' + cash(state.pot * pf.growth_f_suggested))}
      ${cell('growth at that stake', pct(pf.growth_g_suggested))}
      ${g.breakeven_shift != null
         ? cell('break-even calibration shift', pp(g.breakeven_shift) + ' pp', 'warn') : ''}
      ${cell(`P(down ${pct(g.drawdown ? g.drawdown.d : null, 0)})`, pct(g.drawdown ? g.drawdown.p : null))}
    </dl>${histPanel(pr)}`;
  }

  function syncInputs(pf, pr) {
    // Never while the pointer is in the box: writing `.value` under a caret
    // moves it to the end, so a two-digit entry becomes unenterable.
    const set = (id, v) => { const n = el(id); if (document.activeElement !== n) n.value = v; };
    set('pot', state.pot);
    set('in-n', state.rounds);
    set('in-target', state.target == null ? '' : state.target);
    set('in-f', state.customF == null ? '' : Math.round(state.customF * 100));
    set('in-slip-f', state.customF == null ? '' : Math.round(state.customF * 100));
    const g = state.customF == null ? null : gAt(pr, Math.round(state.customF * 100) / 100);
    set('in-g', g == null ? '' : (g * 100).toFixed(2));
    // Buttons hold no caret, so unlike the boxes above they are safe to write
    // whether or not one of them has focus.
    for (const b of el('stake-mode').children) {
      b.classList.toggle('on', b.dataset.mode === state.stakeMode);
    }
  }

  function renderProjection() {
    const pf = state.project == null ? null : portfolioById.get(state.project);
    const pr = pf && pf.projection;
    el('proj-wrap').hidden = !pr;
    el('proj-empty').hidden = !!pr;
    if (!pr) {
      el('proj-empty').innerHTML = !hasProjection
        ? `<p>This run was exported without projections.</p>
           <p class="hint">Re-run <span class="mono">06_Split</span> with
           <span class="mono">with_projection=True</span> to build them.</p>`
        : `<p>No portfolio chosen.</p>
           <p class="hint">Open one on the Portfolio Book and follow
           <span class="mono">projection →</span>.</p>`;
      return;
    }
    // Not `int()`: an id is a label, and "portfolio 162,455" reads as a quantity.
    el('proj-id').innerHTML = `<strong>portfolio ${esc(refOf(pf))}</strong> ·
      ${esc(BOOK.splits[pf.split])} · ${pf.legs} legs · return ${pct(pf.expected_return_pct)}
      · sd ${pct(pf.sd_pct)}`;
    el('proj-fan').innerHTML = fanPanel(pf, pr);
    el('proj-curve').innerHTML = curvePanel(pf, pr);
    el('proj-calc').innerHTML = calcPanel(pf, pr);
    el('proj-slip').innerHTML = slipPanel(pf);
    el('proj-stats').innerHTML = statsPanel(pf, pr);
    syncInputs(pf, pr);
  }

  // --- chrome -------------------------------------------------------------

  function renderMeta() {
    if (state.view === 'match') {
      const t = (PRED.generated_at || '').slice(11, 16);
      el('run-meta').textContent =
        `${fixtures.length ? fixtures[0].date : ''} · ${fixtures.length} fixtures${t ? ' · run ' + t : ''}`;
    } else if (state.view === 'projection') {
      const pf0 = state.project == null ? null : portfolioById.get(state.project);
      const pf = pf0 && pf0.projection ? pf0 : null;   // AXES is null when nothing carries one
      el('run-meta').textContent = pf
        ? `pot ${cash(state.pot)} · ${int(AXES.rounds[AXES.rounds.length - 1])} rounds `
          + `· projections under the model's own probabilities`
        : 'no portfolio chosen';
    } else {
      const c = BOOK.counts;
      el('run-meta').textContent =
        `${int(c.scored)} scored → ${int(c.exported)} undominated → ${int(visible.length)} match filters`
        + (RISK ? ` · ${riskSentence()}` : '');
    }
  }

  function render() {
    el('view-match').hidden = state.view !== 'match';
    el('view-book').hidden = state.view !== 'book';
    el('view-projection').hidden = state.view !== 'projection';
    for (const b of el('tabs').children) b.classList.toggle('on', b.dataset.view === state.view);
    el('stake').value = state.stake;
    if (state.view === 'match') renderMatch();
    else if (state.view === 'projection') renderProjection();
    else { renderRail(); renderBook(); }
    renderMeta();
    writeHash();
  }

  function setView(v) { state.view = v; render(); }

  // --- events -------------------------------------------------------------

  el('tabs').addEventListener('click', e => {
    const b = e.target.closest('button[data-view]');
    if (b) setView(b.dataset.view);
  });

  el('view-match').addEventListener('click', e => {
    const ticket = e.target.closest('.ticket');
    if (ticket) { state.event = ticket.dataset.event; state.open = null; render(); return; }
    const jump = e.target.closest('.prop-jump');
    if (jump) {
      // Clicking a proposition ties the two screens together: the Portfolio Book
      // opens already filtered to portfolios containing it.
      state.props.set(jump.dataset.prop, 'in');
      setView('book');
    }
  });

  el('rail').addEventListener('input', e => {
    const inp = e.target.closest('input[type=range]');
    if (inp) {
      const f = state.faders[Number(inp.dataset.i)];
      const v = Number(inp.value);
      if (inp.dataset.h === 'lo') f.min = Math.min(v, f.max); else f.max = Math.max(v, f.min);
      // Repaint the fader in place rather than re-rendering the rail, so the
      // pointer keeps its grip on the thumb mid-drag.
      const wrap = inp.closest('.fader');
      const b = BOUNDS.get(f.key), span = (b.hi - b.lo) || 1;
      const field = FIELD_BY_KEY.get(f.key);
      wrap.querySelector('.val').textContent = `${fmtField(field, f.min)} – ${fmtField(field, f.max)}`;
      const range = wrap.querySelector('.range');
      range.style.left = ((f.min - b.lo) / span) * 100 + '%';
      range.style.right = ((b.hi - f.max) / span) * 100 + '%';
      wrap.querySelector('[data-h=lo]').style.zIndex = f.min > (b.lo + b.hi) / 2 ? 3 : 1;
      // A handle dragged past its partner is clamped in state; put the control
      // back where the state says it is, or the thumb sits somewhere the filter
      // is not.
      inp.value = String(inp.dataset.h === 'lo' ? f.min : f.max);
      renderBook();
      renderMeta();
      writeHash();
      return;
    }
    if (e.target.id === 'prop-search') {
      state.search = e.target.value;
      refreshPropList();
    }
  });

  el('rail').addEventListener('click', e => {
    const drop = e.target.closest('[data-drop]');
    if (drop) { state.faders.splice(Number(drop.dataset.drop), 1); renderRail(); renderBook(); renderMeta(); writeHash(); return; }

    const split = e.target.closest('[data-split]');
    if (split) {
      const k = split.dataset.split;
      state.splits.has(k) ? state.splits.delete(k) : state.splits.add(k);
      renderRail(); renderBook(); renderMeta(); writeHash(); return;
    }

    const setProp = e.target.closest('[data-prop-set]');
    if (setProp) {
      const id = setProp.dataset.propSet, mode = setProp.dataset.mode;
      const prop = propById.get(id);
      if (prop) state.events.delete(prop.event);   // finer beats coarser
      mode === 'none' ? state.props.delete(id) : state.props.set(id, mode);
      refreshPropList();
      renderBook(); renderMeta(); writeHash(); return;
    }

    const toggle = e.target.closest('[data-toggle]');
    if (toggle) {
      const ev = toggle.dataset.toggle;
      state.expanded.has(ev) ? state.expanded.delete(ev) : state.expanded.add(ev);
      refreshPropList();
      return;
    }

    const chip = e.target.closest('[data-ev]');
    if (chip) {
      const ev = chip.dataset.ev, mode = chip.dataset.mode;
      // Setting the fixture clears anything set on its individual propositions,
      // so the two levels can never hold contradictory opinions about the same
      // event -- which would filter to nothing and look like a bug.
      for (const p of propsByEvent.get(ev) || []) state.props.delete(p.id);
      mode === 'none' ? state.events.delete(ev) : state.events.set(ev, mode);
      refreshPropList();
      renderBook(); renderMeta(); writeHash(); return;
    }

    if (e.target.id === 'add-filter') {
      const box = e.target;
      box.outerHTML = `<select class="add-filter" id="add-field" autofocus>
        <option value="">choose a field…</option>
        ${FIELDS.map(f => `<option value="${f.key}">${esc(f.label)}</option>`).join('')}</select>`;
      el('add-field').focus();
    }
  });

  el('rail').addEventListener('change', e => {
    if (e.target.id === 'add-field' && e.target.value) {
      // Duplicates are allowed on purpose -- a second window on a field already
      // shown is simpler to permit than to reason about preventing.
      state.faders.push(newFader(e.target.value, true));
      renderRail(); renderBook(); renderMeta(); writeHash();
    } else if (e.target.id === 'add-field') {
      renderRail();
    }
  });

  el('pf-head').addEventListener('click', e => {
    const th = e.target.closest('[data-sort]');
    if (!th) return;
    const key = th.dataset.sort;
    state.sort = state.sort.key === key ? { key, dir: -state.sort.dir } : { key, dir: -1 };
    renderBook(); writeHash();
  });
  el('pf-head').addEventListener('keydown', e => {
    if ((e.key === 'Enter' || e.key === ' ') && e.target.dataset.sort) { e.preventDefault(); e.target.click(); }
  });

  // Coalesced to one repaint per frame: a fast flick fires scroll far more often
  // than the screen refreshes, and re-rendering the window on every one of those
  // is work nobody sees.
  let scrollQueued = false;
  el('pf-scroll').addEventListener('scroll', () => {
    if (scrollQueued || !visible.length) return;
    scrollQueued = true;
    requestAnimationFrame(() => { scrollQueued = false; renderRows(); });
  });

  el('pf-body').addEventListener('click', e => {
    const goto = e.target.closest('[data-goto]');
    if (goto) { e.stopPropagation(); state.event = goto.dataset.goto; setView('match'); return; }
    const proj = e.target.closest('[data-project]');
    if (proj) {
      e.stopPropagation();
      state.project = Number(proj.dataset.project);
      setView('projection');
      return;
    }
    if (e.target.id === 'reset-inline') { resetFilters(); return; }
    const row = e.target.closest('tr.row');
    if (!row) return;
    const id = Number(row.dataset.id);
    state.open = state.open === id ? null : id;
    expandedH = 0;
    renderRows(); writeHash();
  });
  el('pf-body').addEventListener('keydown', e => {
    if ((e.key === 'Enter' || e.key === ' ') && e.target.classList.contains('row')) { e.preventDefault(); e.target.click(); }
  });

  el('stake').addEventListener('input', e => {
    state.stake = Number(e.target.value) || 0;
    if (state.open != null) renderRows();
    writeHash();
  });

  // --- projection controls ------------------------------------------------
  //
  // Each writes state, re-renders the panels and lets `syncInputs` fill the
  // *other* boxes -- which is why it skips whichever one has focus. The stake
  // and growth fields are two views of one point on `g_curve`, so typing in
  // either has to move the other without moving the caret in the one being used.

  const projChanged = () => { renderProjection(); renderMeta(); writeHash(); };

  el('pot').addEventListener('input', e => {
    state.pot = Math.max(0, Number(e.target.value) || 0);
    projChanged();
  });
  el('in-f').addEventListener('input', e => {
    const v = Number(e.target.value);
    state.customF = v >= 1 && v <= 99 ? Math.round(v) / 100 : null;
    projChanged();
  });
  el('in-g').addEventListener('input', e => {
    const pf = portfolioById.get(state.project);
    const raw = e.target.value;
    if (raw === '' || !pf || !pf.projection) { state.customF = null; projChanged(); return; }
    const f = fAt(pf.projection, Number(raw) / 100);
    // Rounded onto the exported grid, so the stake shown is one the curve was
    // actually evaluated at rather than a point between two of them.
    state.customF = f == null ? null : Math.min(0.99, Math.max(0.01, Math.round(f * 100) / 100));
    projChanged();
  });
  el('in-n').addEventListener('input', e => {
    state.rounds = Math.max(1, Math.round(Number(e.target.value) || 1));
    projChanged();
  });
  el('in-target').addEventListener('input', e => {
    const v = Number(e.target.value);
    state.target = v > 0 ? v : null;
    projChanged();
  });
  el('stake-mode').addEventListener('click', e => {
    const b = e.target.closest('button[data-mode]');
    if (!b) return;
    state.stakeMode = b.dataset.mode;
    // Custom with nothing in it is a blank panel. Seeding from the suggested
    // stake gives the first click something to show and then move.
    if (state.stakeMode === 'custom' && state.customF == null) {
      const pf = portfolioById.get(state.project);
      const f = pf && pf.growth_f_suggested;
      if (f != null) state.customF = Math.min(0.99, Math.max(0.01, Math.round(f * 100) / 100));
    }
    projChanged();
  });
  // The slip's own stake box. It is the same `state.customF` the curve reads, so
  // typing here moves the curve's `yours` marker and adds the calculator's row --
  // but unlike `in-f`, whose job is reading the curve, this box exists only to
  // price the slip, so using it switches the slip to Custom.
  el('in-slip-f').addEventListener('input', e => {
    const v = Number(e.target.value);
    state.customF = v >= 1 && v <= 99 ? Math.round(v) / 100 : null;
    state.stakeMode = 'custom';
    projChanged();
  });
  el('proj-back').addEventListener('click', () => {
    state.open = state.project;
    setView('book');
  });

  function resetFilters() {
    state.faders = DEFAULT_FADERS.map(key => newFader(key));
    state.splits.clear();
    state.props.clear();
    state.events.clear();
    state.search = '';
    renderRail(); renderBook(); renderMeta(); writeHash();
  }
  el('reset-filters').addEventListener('click', resetFilters);

  // The number of rows that fit is a function of the window, so a resize is a
  // reason to rebuild the window even though nothing about the data changed.
  let resizeQueued = false;
  window.addEventListener('resize', () => {
    if (resizeQueued || state.view !== 'book' || !visible.length) return;
    resizeQueued = true;
    requestAnimationFrame(() => { resizeQueued = false; renderRows(); });
  });

  window.addEventListener('hashchange', () => {
    if (location.hash.replace(/^#/, '') === lastWritten) return;   // our own write
    readHash();
    render();
  });

  // --- go -----------------------------------------------------------------

  readHash();
  render();
})();
