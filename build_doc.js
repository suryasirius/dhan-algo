const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, ShadingType, BorderStyle, AlignmentType, LevelFormat,
} = require("docx");

const TABLE_WIDTH = 9360; // DXA, fits within 1" margins on Letter

function hr() {
  return new Paragraph({
    border: { bottom: { color: "999999", space: 1, style: BorderStyle.SINGLE, size: 6 } },
    spacing: { after: 200 },
  });
}

function h1(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_1, spacing: { before: 300, after: 150 } });
}
function h2(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_2, spacing: { before: 250, after: 120 } });
}
function p(text, opts = {}) {
  return new Paragraph({ children: [new TextRun({ text, ...opts })], spacing: { after: 140 } });
}
function bullet(text, opts = {}) {
  return new Paragraph({
    children: [new TextRun({ text, ...opts })],
    numbering: { reference: "bullets", level: 0 },
    spacing: { after: 80 },
  });
}

function cell(text, { bold = false, width, shade } = {}) {
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    shading: shade ? { type: ShadingType.CLEAR, color: "auto", fill: shade } : undefined,
    margins: { top: 80, bottom: 80, left: 100, right: 100 },
    children: [new Paragraph({ children: [new TextRun({ text, bold })] })],
  });
}

function perfTable(rows) {
  const widths = [2400, 1600, 1600, 1600, 2160];
  const header = new TableRow({
    tableHeader: true,
    children: [
      cell("Test window", { bold: true, width: widths[0], shade: "DDEBF7" }),
      cell("Trades", { bold: true, width: widths[1], shade: "DDEBF7" }),
      cell("Win rate", { bold: true, width: widths[2], shade: "DDEBF7" }),
      cell("Net (pts)", { bold: true, width: widths[3], shade: "DDEBF7" }),
      cell("Notes", { bold: true, width: widths[4], shade: "DDEBF7" }),
    ],
  });
  const body = rows.map(r => new TableRow({
    children: [
      cell(r[0], { width: widths[0] }),
      cell(r[1], { width: widths[1] }),
      cell(r[2], { width: widths[2] }),
      cell(r[3], { width: widths[3] }),
      cell(r[4], { width: widths[4] }),
    ],
  }));
  return new Table({
    width: { size: TABLE_WIDTH, type: WidthType.DXA },
    columnWidths: widths,
    rows: [header, ...body],
  });
}

function paramTable(rows) {
  const widths = [3600, 2880, 2880];
  const header = new TableRow({
    tableHeader: true,
    children: [
      cell("Parameter", { bold: true, width: widths[0], shade: "DDEBF7" }),
      cell("Value", { bold: true, width: widths[1], shade: "DDEBF7" }),
      cell("Meaning", { bold: true, width: widths[2], shade: "DDEBF7" }),
    ],
  });
  const body = rows.map(r => new TableRow({
    children: [
      cell(r[0], { width: widths[0] }),
      cell(r[1], { width: widths[1] }),
      cell(r[2], { width: widths[2] }),
    ],
  }));
  return new Table({
    width: { size: TABLE_WIDTH, type: WidthType.DXA },
    columnWidths: widths,
    rows: [header, ...body],
  });
}

const doc = new Document({
  numbering: {
    config: [{
      reference: "bullets",
      levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT }],
    }],
  },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 } } }, // US Letter
    children: [
      new Paragraph({
        children: [new TextRun({ text: "Nifty Scalping Strategy", bold: true, size: 44 })],
        spacing: { after: 60 },
      }),
      new Paragraph({
        children: [new TextRun({ text: "Active Concepts: #2, #3, #4 — Documentation & Backtest Record", size: 26, color: "555555" })],
        spacing: { after: 300 },
      }),
      hr(),

      h1("1. Trading Profile"),
      p("Discretionary intraday scalping strategy on Nifty 50, traded exclusively through in-the-money (ITM) Call and Put options on the current weekly expiry. Every BUY signal buys the nearest ITM Call; every SELL signal buys the nearest ITM Put. Exits are fixed and mechanical, not discretionary or trailing."),
      bullet("Instrument: Nifty 50 index options, ITM strike only (nearest strike in-the-money)"),
      bullet("Expiry: always the current weekly expiry"),
      bullet("Stop-loss: 5 points on the underlying"),
      bullet("Target: 5 points on the underlying (1:1 risk:reward)"),
      bullet("Time exit: 15 bars (roughly 15 minutes) if neither stop nor target is hit"),
      bullet("Confluence required: 1 vote (a single concept firing is enough to trigger a trade)"),

      h1("2. Why Only Concepts 2, 3, and 4"),
      p("The original strategy had 9 discretionary price-action concepts running side by side. Backtesting across several window lengths (5 days, 2 weeks, 1 month, and 6 months of real Nifty 1-minute data via the Dhan API) was used to see which concepts actually held up versus which were noise or actively harmful."),
      p("Concepts 1 (compression spike buy), 5 (failed-spike reversal buy), 6 (failed-spike reversal sell), and 9 (weakening domes sell) were dropped early after consistently underperforming or showing 0% win rates on real trade samples. Concepts 7 and 8 (leg-1/leg-2 anchor-confirm) never fired a single signal across any tested window, so there isn't enough data to judge them at all."),
      p("Concepts 2, 3, and 4 were the only three that stayed net positive across the shorter windows. Note the honest caveat in Section 6: over a full 6-month test, concept 3 reversed from the best-looking performer to the worst, and concept 2's edge shrank to roughly flat. Concept 4 was the only one of the three that held a (thin) edge across every window tested, including 6 months. None of the three should be considered a fully proven, durable edge yet — see Section 6 before trading real money."),

      h1("3. Concept 2 — Compression Spike (Sell)"),
      h2("The idea"),
      p("Price tightens into a narrow range (compression) as buyers and sellers repeatedly fail to gain control. Once one side finally overpowers the other, it breaks out with a sharp, high-range move (the spike). A downside spike out of compression is treated as the start of a fresh move, and the strategy sells into it -- provided the move isn't already exhausted (see the exhaustion filter below, which hands off to concept 6 instead when a spike has already failed)."),
      h2("The precise rule (as coded)"),
      bullet("Compression built: over the last 10 bars, the high-low range is below 0.55x (10 x ATR) for at least 4 consecutive bars"),
      bullet("Spike bar: the current bar's range exceeds 1.7x ATR(14)"),
      bullet("Spike down confirmed: the spike bar's close is below the prior 10-bar low, and the bar closes below its own open (a real bearish close, not just a long wick)"),
      bullet("Not already exhausted: this is skipped if the market is within 3 bars of a prior upside spike that has already failed (i.e. price has closed back below that spike's breakout level) -- that situation is handed to concept 6 instead"),
      h2("Entry"),
      p("SELL (buy ITM Put) on the close of the bar where compression + spike-down + not-exhausted all confirm."),
      h2("Backtest performance"),
      perfTable([
        ["5 trading days (Jul 20-24)", "19", "52.6%", "+55.10", "1:1.6 R:R, first test"],
        ["1 month (~20 days)", "91", "51.6%", "+69.85", "1:1 R:R, held steady vs 5-day test"],
        ["6 months (Jan-Jul)", "549", "49.4%", "-4.90", "Edge shrank to roughly flat at scale"],
      ]),

      h1("4. Concept 3 — Rollover (Buy)"),
      h2("The idea"),
      p("A stretched, weak trend -- one that has been directionless or grinding for a long stretch (weak ADX for many bars) -- tends to run out of energy near its own recent extreme. When price is sitting near the low of that stretched, weak move and shows a fast rejection candle (a long lower wick with a bullish close), it's read as the weak move rolling over and reversing upward."),
      h2("The precise rule (as coded)"),
      bullet("Weak trend: ADX(14) stays below 20 for at least 30 consecutive bars (the \"stretched weak arc\")"),
      bullet("Near the low: the current bar's low is within 0.5% of the lowest low of the last 30 bars"),
      bullet("Fast rejection wick: the lower wick is at least half the bar's total range, and the bar closes above its open"),
      bullet("Fires once per rollover: the signal only triggers on the bar the rejection condition first becomes true, not on every bar it stays true"),
      h2("Entry"),
      p("BUY (buy ITM Call) on the close of the bar where the stretched weak arc, near-low, and fast rejection wick all confirm together."),
      h2("Backtest performance"),
      perfTable([
        ["5 trading days (Jul 20-24)", "5", "60.0%", "+28.35", "Very small sample"],
        ["1 month (~20 days)", "45", "57.8%", "+80.35", "Best-looking performer at this scale"],
        ["6 months (Jan-Jul)", "194", "44.8%", "-193.85", "Reversed hard -- worst performer at full scale"],
      ]),
      p("This is the concept to watch most closely. It looked like the strongest of the three on every short window, then became the single biggest loser once tested on 6 months of data -- a textbook sign of an edge that may have been fit to a specific recent stretch of market rather than a durable pattern."),

      h1("5. Concept 4 — Rollover (Sell)"),
      h2("The idea"),
      p("The mirror image of concept 3. A stretched, weak trend that is sitting near its own recent high, showing a fast rejection candle downward (a long upper wick with a bearish close), is read as that weak move rolling over and reversing down."),
      h2("The precise rule (as coded)"),
      bullet("Weak trend: ADX(14) stays below 20 for at least 30 consecutive bars"),
      bullet("Near the high: the current bar's high is within 0.5% of the highest high of the last 30 bars"),
      bullet("Fast rejection wick: the upper wick is at least half the bar's total range, and the bar closes below its open"),
      bullet("Fires once per rollover: same edge-triggered behaviour as concept 3"),
      h2("Entry"),
      p("SELL (buy ITM Put) on the close of the bar where the stretched weak arc, near-high, and fast rejection wick all confirm together."),
      h2("Backtest performance"),
      perfTable([
        ["5 trading days (Jul 20-24)", "2", "50.0%", "-16.15", "Only 2 trades -- too small to judge"],
        ["1 month (~20 days)", "34", "61.8%", "+91.10", "Best win rate of the three concepts"],
        ["6 months (Jan-Jul)", "182", "49.5%", "+61.45", "Only concept that stayed net positive at full scale"],
      ]),
      p("Concept 4 is the most consistent of the three -- it's the only one that never flipped to net-negative at any tested scale, though the edge is thin (roughly 0.34 points per trade on average over 6 months) and the win rate sits right around the 49-50% breakeven line for a 1:1 risk:reward setup."),

      h1("6. Honest Caveats Before Trading Real Money"),
      bullet("Short windows are misleading: every concept looked better on 1-2 week and 1-month tests than it did on 6 months. Concept 3 fully reversed. Treat any result under ~100 trades as a rough signal, not proof."),
      bullet("No real historical option premium data: the money simulations done separately during testing used the underlying's point moves scaled by an assumed 0.6 delta, not actual historical ITM option prices (expired weekly contracts drop out of Dhan's instrument list, so exact past premiums aren't retrievable). Real fills will differ."),
      bullet("Backtest fills use closing price only: stop and target hits are checked against each 1-minute bar's close, not true intrabar highs/lows, so real broker fills (especially on sharp spikes) may be worse than the backtest shows."),
      bullet("Confluence = 1 means single-vote entries: any one of these three concepts firing alone is enough to trigger a trade; there's no requirement for agreement between concepts."),
      bullet("None of the three concepts has a full 6-month win rate clearly above 50%, which is the breakeven line for a 1:1 stop/target. Their positive net points came from asymmetric average win/loss sizes, not from a strong win rate edge."),

      h1("7. Reference: Key Parameters"),
      paramTable([
        ["SL_POINTS", "5", "Fixed stop-loss, in Nifty points on the underlying"],
        ["TP_POINTS", "5", "Fixed target, in Nifty points on the underlying (1:1 R:R)"],
        ["MAX_BARS_IN_TRADE", "15", "Time-based exit if neither stop nor target is hit"],
        ["CONFLUENCE_REQUIRED", "1", "Minimum votes needed to trigger a trade"],
        ["COMP_LEN / COMP_THRESHOLD / MIN_TIGHT_BARS", "10 / 0.55 / 4", "Compression detection (concept 2)"],
        ["ATR_LEN / SPIKE_MULT / BREAKOUT_LEN / EXHAUSTION_BARS", "14 / 1.7 / 10 / 3", "Spike + exhaustion detection (concept 2)"],
        ["ADX_LEN / WEAK_ADX_THRESH / WEAK_ARC_BARS", "14 / 20 / 30", "Stretched weak-trend detection (concepts 3 & 4)"],
        ["NEAR_EXTREME_PCT", "0.5%", "How close to the 30-bar extreme counts as \"near\" (concepts 3 & 4)"],
        ["REJECTION_WICK_RATIO", "0.5", "Minimum wick-to-range ratio for a valid rejection candle (concepts 3 & 4)"],
        ["DISABLED_CONCEPTS", "{1, 5, 6, 7, 8, 9}", "All concepts except 2, 3, 4 are switched off"],
        ["ITM_STRIKE_OFFSET", "0", "Nearest ITM strike (one strike in-the-money)"],
      ]),
    ],
  }],
});

Packer.toBuffer(doc).then(buf => require("fs").writeFileSync("Active_Concepts_2_3_4_Documentation.docx", buf));
