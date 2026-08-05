const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, ShadingType, BorderStyle, AlignmentType, Numbering, LevelFormat,
  PageBreak
} = require("docx");

const PAGE_W = 12240, PAGE_H = 15840; // US Letter

function h1(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_1, spacing: { before: 300, after: 150 } });
}
function h2(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_2, spacing: { before: 240, after: 120 } });
}
function p(text, opts = {}) {
  return new Paragraph({
    children: [new TextRun({ text, ...opts })],
    spacing: { after: 160 },
  });
}
function bullet(text, opts = {}) {
  return new Paragraph({
    children: [new TextRun({ text, ...opts })],
    numbering: { reference: "bullets", level: 0 },
    spacing: { after: 80 },
  });
}
function hr() {
  return new Paragraph({
    text: "",
    border: { bottom: { color: "999999", space: 1, style: BorderStyle.SINGLE, size: 6 } },
    spacing: { after: 200 },
  });
}

function ruleBlock(title, body) {
  return [
    new Paragraph({
      children: [new TextRun({ text: title, bold: true, color: "1F4E79" })],
      spacing: { before: 200, after: 60 },
    }),
    p(body),
  ];
}

// ---- Signal key table ----
const sigRows = [
  ["#", "Direction", "Signal Name", "Meaning"],
  ["1", "BUY", "Compression Spike", "Tight consolidation releases into an upward momentum spike"],
  ["2", "SELL", "Compression Spike", "Tight consolidation releases into a downward momentum spike"],
  ["3", "BUY", "Rollover (bottom)", "Stretched weak arc at a low + fast rejection candle confirms"],
  ["4", "SELL", "Rollover (top)", "Stretched weak arc at a high + fast rejection candle confirms"],
  ["5", "BUY", "Failed-Spike Reversal", "A down-spike failed to hold -> reversal long"],
  ["6", "SELL", "Failed-Spike Reversal", "An up-spike failed to hold -> reversal short"],
  ["7", "BUY", "Leg-2 Confirm", "Sharp fall = anchor only; 2nd-leg pullback breaks higher -> entry"],
  ["8", "SELL", "Leg-2 Confirm", "Sharp rally = anchor only; 2nd-leg pullback breaks lower -> entry"],
  ["9", "SELL", "Weakening Domes", "Edge of the last of several progressively weaker rounded tops"],
];
const colWidths = [700, 1200, 2600, 5100];
const tableWidth = colWidths.reduce((a, b) => a + b, 0);

function cell(text, opts = {}) {
  return new TableCell({
    width: { size: opts.width || 1000, type: WidthType.DXA },
    shading: opts.header ? { type: ShadingType.CLEAR, fill: "1F4E79" } : undefined,
    children: [new Paragraph({
      children: [new TextRun({ text, bold: !!opts.header, color: opts.header ? "FFFFFF" : "000000", size: 20 })],
    })],
  });
}

const sigTable = new Table({
  width: { size: tableWidth, type: WidthType.DXA },
  columnWidths: colWidths,
  rows: sigRows.map((row, i) => new TableRow({
    children: row.map((val, j) => cell(val, { header: i === 0, width: colWidths[j] })),
  })),
});

const doc = new Document({
  numbering: {
    config: [{
      reference: "bullets",
      levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } } }],
    }],
  },
  sections: [{
    properties: { page: { size: { width: PAGE_W, height: PAGE_H }, margin: { top: 1080, bottom: 1080, left: 1080, right: 1080 } } },
    children: [
      new Paragraph({
        children: [new TextRun({ text: "Price Action & Market Structure Strategy", bold: true, size: 44, color: "1F4E79" })],
        spacing: { after: 80 },
      }),
      new Paragraph({
        children: [new TextRun({ text: "Working Strategy Compendium — Discretionary Rules Compiled to Date", size: 24, color: "555555", italics: true })],
        spacing: { after: 40 },
      }),
      new Paragraph({
        children: [new TextRun({ text: "Status: In development, not final — a checkpoint snapshot of the rules discussed so far.", size: 20, color: "888888" })],
        spacing: { after: 300 },
      }),
      hr(),

      h1("1. Overview"),
      p("This document compiles a discretionary trading strategy built from personal chart experience, centered on price action and market structure rather than indicators. The core skill is reading the shape, tightness, and momentum of small recurring price formations (“mini-structures”) and using them to anticipate the next directional move."),
      p("The strategy is applied as an intraday scalping approach on the Nifty index, 1-minute chart, with fixed, tight risk parameters (detailed in Section 2)."),

      h1("2. Trading Profile & Risk Parameters"),
      bullet("Instrument: Nifty (index), intraday"),
      bullet("Timeframe: 1-minute chart (scalping)"),
      bullet("Stop-loss: tight, maximum 10 points"),
      bullet("Target: 10 to 30 points per trade, varying by setup"),
      bullet("Frequency: multiple trades per day, no long holding time"),
      bullet("Entry style: precise, “sniper”-level entries rather than broad zone entries"),

      h1("3. Core Unifying Principle"),
      p("Every pattern below is a variation of one underlying mechanic: price structures compete for dominance, and momentum is transferred gradually rather than all at once."),
      p("In the trader’s own words: “It’s all about gaining power gradually, one [structure] over the other — if it has strength it will overpower [the prior structure]; if it doesn’t, it will try to gain power, hitting again and again, gradually gaining momentum until it breaks through.”"),
      p("Concretely: if a new mini-structure has enough strength, it overpowers the resistance/support left behind by the prior structure immediately. If it doesn’t, it must repeatedly test that level — each failed attempt chips away at the old level’s strength — until enough of these attempts have accumulated to finally break through."),

      h1("4. Structure Concepts"),

      ...ruleBlock("4.1 Compression → Spike",
        "A tight, weak, low-momentum consolidation (small candles, no directional push) that fails to make progress is building stored pressure. Once released, it tends to move as a sharp, fast momentum spike in the breakout direction rather than a gradual move."),

      ...ruleBlock("4.2 Repeated Compression Compounds",
        "Multiple tight mini-structures stacked back-to-back (not just a single one) build a larger store of pressure than any one structure alone, and tend to release into a bigger, more sustained spike — observed on both upside and downside breaks."),

      ...ruleBlock("4.3 Stretched Weak Arc → Rollover",
        "The inverse of compression: a long, drawn-out, low-momentum arc made of small, weak candles over many bars signals no real directional intent (“no pressure”). This lazy structure tends to roll over or reverse hard once it finally breaks — the longer and weaker the stretch, the larger the eventual move."),

      ...ruleBlock("4.4 Sequential Weakening Domes",
        "A compounding version of 4.3: a series of rounded tops, each with a lower high and less momentum than the previous one. The valid entry is at the edge of the LAST (weakest) dome in the sequence, not the first — since the earlier domes haven’t yet exhausted themselves."),

      ...ruleBlock("4.5 Support/Resistance Validity = Reaction Speed",
        "A level only counts as real support or resistance if price reacts to it with a fast, aggressive, tightly-packed candle cluster (a quick rejection). A slow, weak, drifting reaction with no aggression means the level is undefended, and price is likely to slide straight through it rather than hold."),

      ...ruleBlock("4.6 Sharp-Top vs. Rounded-Top Mini-Structures",
        "The shape of a mini-structure’s peak changes its meaning even when the overall arc looks similar. A sharp, pointed top signals continued accumulation and buying intent (a bullish, staircase-style build), while a rounded, stretched top signals exhaustion and rollover (bearish) — see 4.3."),

      ...ruleBlock("4.7 Multi-Leg Reversal (Anchor + Confirm)",
        "After a sharp, fast V-shaped move, that first leg is an anchor/base only — not itself a reversal signal. The real, confirmable reversal typically comes from the second leg onward: a pullback that holds a shallower extreme (a higher low after a down-leg, or a lower high after an up-leg) and then breaks its own local high/low. Noted as a conditional pattern, not universal — sometimes only a single leg-and-pullback occurs."),

      ...ruleBlock("4.8 Air Pocket (Unsupported Gap)",
        "Measuring the vertical distance between a current weak/rollover top and the last real base or support level, with no structure in between to catch price on the way down, gives a sense of how large the coming move could be. An unsupported “air pocket” like this allows price to fall (or rise) quickly once it starts moving."),

      ...ruleBlock("4.9 Resistance/Support Inheritance & Decay",
        "The most recent refinement, and the concept that ties everything together (see Section 3). Each new mini-structure must contend with the dominance left behind by the prior one. A tight, strong prior structure isn’t broken on the first attempt — later structures repeatedly test it, each failed attempt weakening it slightly, until it finally gives way. This reframes “repeated compression” as a gradual decay process rather than a simple counter of tight bars."),

      h1("5. Codification Status (TradingView Pine Script v4)"),
      p("The rules above have been translated into a working Pine Script v5 strategy for TradingView, with nine numbered, individually toggleable signal types. Each trade's exit label (target/stop/time-based) carries the same number as its originating signal, so results can be traced back to a specific rule."),
      sigTable,
      new Paragraph({ text: "", spacing: { after: 200 } }),
      p("Exit logic: fixed points (default 10pt stop / 20pt target, adjustable 10-30pt) plus a maximum-bars-in-trade timer, matching the scalping profile in Section 2."),

      h1("6. Not Yet Codified / Open Items"),
      bullet("4.6 Sharp-top vs. rounded-top distinction — not yet an explicit, separate filter in the script."),
      bullet("4.9 Resistance/support decay mechanic — current script only counts consecutive tight bars; it doesn't yet model a level's strength decaying across specific repeated failed attempts."),
      bullet("4.5 Reaction-speed validity — currently only applied to the weak-arc rollover signals (3/4), not to general support/resistance levels elsewhere on the chart."),
      bullet("Quantified win-rate backtest — pending: no direct market-data feed is available in this environment; testing is being done live in TradingView's Strategy Tester instead."),

      h1("7. Status"),
      p("This strategy is explicitly a work in progress (“not final” per the trader). This document is a checkpoint capturing every rule and chart example discussed to date, intended to be updated as new observations and refinements are added."),
    ],
  }],
});

Packer.toBuffer(doc).then(buf => {
  require("fs").writeFileSync("Price_Action_Market_Structure_Strategy.docx", buf);
  console.log("done");
});
