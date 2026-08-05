const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  LevelFormat, convertInchesToTwip
} = require("docx");

const GREY = "666666";
const DARK = "1A1A1A";
const ACCENT = "2E5C8A";

const numbering = {
  config: [
    {
      reference: "bullets",
      levels: [
        { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: convertInchesToTwip(0.35), hanging: convertInchesToTwip(0.2) } } } },
      ],
    },
  ],
};

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 320, after: 160 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: ACCENT, space: 4 } },
    children: [new TextRun({ text, bold: true, color: ACCENT, size: 26 })],
  });
}

function body(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 160, line: 276 },
    children: [new TextRun({ text, size: 22, color: DARK, ...opts })],
  });
}

function bullet(text, opts = {}) {
  return new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    spacing: { after: 90, line: 276 },
    children: [new TextRun({ text, size: 22, color: DARK, ...opts })],
  });
}

function labelValueRow(label, value) {
  return new TableRow({
    children: [
      new TableCell({
        width: { size: 3200, type: WidthType.DXA },
        shading: { type: ShadingType.CLEAR, fill: "F2F2F2" },
        margins: { top: 100, bottom: 100, left: 150, right: 150 },
        children: [new Paragraph({ children: [new TextRun({ text: label, bold: true, size: 21, color: DARK })] })],
      }),
      new TableCell({
        width: { size: 5600, type: WidthType.DXA },
        margins: { top: 100, bottom: 100, left: 150, right: 150 },
        children: [new Paragraph({ children: [new TextRun({ text: value, size: 21, color: DARK })] })],
      }),
    ],
  });
}

const doc = new Document({
  numbering,
  sections: [
    {
      properties: {
        page: {
          size: { width: 12240, height: 15840 },
          margin: { top: 1080, bottom: 1080, left: 1080, right: 1080 },
        },
      },
      children: [
        new Paragraph({
          spacing: { after: 40 },
          children: [new TextRun({
            text: "Fixed-Strike Opening-Range Breakout",
            bold: true, size: 34, color: DARK,
          })],
        }),
        new Paragraph({
          spacing: { after: 40 },
          children: [new TextRun({
            text: "5-Point Target / 3-Minute Time Exit Strategy",
            bold: true, size: 30, color: ACCENT,
          })],
        }),
        new Paragraph({
          spacing: { after: 280 },
          children: [new TextRun({
            text: "Concept summarized for review · 30 July 2026",
            italics: true, size: 20, color: GREY,
          })],
        }),

        h1("1. Overview"),
        body(
          "This is a single-strike, single-direction breakout trade: pick one specific option strike ahead of time, mark its own opening 5-minute high/low, buy the call once that high is confirmed broken, and exit on whichever comes first — a fixed 5-point profit target, or 3 minutes in the trade. No trailing stop, no strike re-selection during the day, and no stop-loss as written — flagged below."
        ),

        h1("2. Setup — Fix the Strike & Mark Its Opening Range"),
        bullet("Before or at market open, fix one specific option strike to watch for the day (e.g., 24000 CE). This strike does not change during the day."),
        bullet("On that strike’s own 1-minute premium chart, take the first 5 candles (09:15–09:20) and mark: Strike High = highest high of those 5 candles; Strike Low = lowest low of those 5 candles."),

        h1("3. Entry Trigger"),
        bullet("From 09:20 onward, watch the strike’s own premium against its Strike High."),
        bullet("When price trades above the Strike High, wait for a full 1-minute candle to close above it — a confirmed close, not just an intra-candle touch."),
        bullet("On that confirmation, buy the call option (long CE) at the prevailing premium."),

        h1("4. Exit Rule — 5-Point Target OR 3-Minute Time Exit, Whichever First"),
        body(
          "Two exit conditions run side by side from the moment of entry. The trade closes the instant either one is satisfied — there is no trailing and no partial exit."
        ),
        bullet("Target exit: close the position as soon as the premium is 5 points above the entry price."),
        bullet("Time exit: if the 5-point target has not been hit within 3 minutes of entry, close the position at the prevailing premium at that 3-minute mark, regardless of profit or loss."),
        body(
          "Worked example, as you described it:",
          { bold: true }
        ),

        new Table({
          width: { size: 8800, type: WidthType.DXA },
          columnWidths: [3200, 5600],
          rows: [
            labelValueRow("Strike watched", "24000 CE"),
            labelValueRow("Entry condition met", "1-min candle closes above the opening 5-min high"),
            labelValueRow("Entry price", "₹150"),
            labelValueRow("Target", "₹150 + 5 = ₹155"),
            labelValueRow("Time deadline", "3 minutes after entry"),
            labelValueRow("Exit — target case", "Sell the moment premium hits ₹155, if that happens within 3 minutes"),
            labelValueRow("Exit — time case", "If ₹155 is not reached within 3 minutes, sell at whatever the premium is at the 3-minute mark"),
          ],
        }),

        new Paragraph({ spacing: { before: 240 }, children: [] }),

        h1("5. Open Questions to Confirm Before Testing/Trading"),
        bullet("No stop-loss is specified in this version — as written, a losing trade is only bounded by the 3-minute time exit, not by price. Confirm whether a hard point-stop (e.g., the 8–10-point stops used elsewhere in your other strategies) should also apply within that 3-minute window, since a fast adverse move could still hurt more than intended before the clock runs out."),
        bullet("Strike selection criteria — how is “the one strike” chosen each day (ATM, a fixed offset from spot, previous day’s close, or a strike you pick manually)?"),
        bullet("Downside case — this version only describes a call trade off a high-breakout. Is there a mirrored put trade off a low-breakout, or is this intentionally a call-only strategy?"),
        bullet("Re-entry — if the time exit closes the trade and the premium later breaks a new high, can you re-enter for a second attempt the same day, or is it one trade per day?"),
        bullet("Confirmation candle — “wait for one candle to close above the high” means the candle’s close price, not its high, must exceed the Strike High — confirm this reading is correct."),
        bullet("3-minute measurement — confirm the 3 minutes is measured from the entry fill (the confirmation candle’s close), not from the start of that candle, and that it is checked on 1-minute candle boundaries (the underlying data available is 1-minute bars, so sub-minute precision inside the 3-minute window is not achievable)."),
        bullet("What happens if the high is never broken that day — no trade, and the setup simply expires unused?"),

        h1("6. Disclaimer"),
        body(
          "This is a plain-language summary of the concept as described, for discussion and refinement only. It is not financial advice and hasn’t been tested here. In particular, confirm and add a stop-loss rule before treating this as tradeable — validate on historical data or in a paper account before risking real capital."
        ),
      ],
    },
  ],
});

Packer.toBuffer(doc).then((buf) => {
  require("fs").writeFileSync("/sessions/practical-quirky-hamilton/mnt/outputs/Nifty_Fixed_Strike_5pt_3min_Exit_Strategy_Summary.docx", buf);
  console.log("written");
});
